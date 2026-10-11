import gc
import logging
from methods.Wan21.shared.data.dataset import cycle
from methods.Wan21.shared.data.dataset import LatentLMDBDataset
from methods.Wan21.shared.utils.distributed import (
    EMA_FSDP,
    barrier,
    fsdp_optim_state_dict,
    load_fsdp_optim_state_dict,
    load_fsdp_state_dict,
    fsdp_wrap,
    fsdp_state_dict,
    launch_distributed_job,
)
from methods.Wan21.shared.checkpoint import load_checkpoint_bundle, save_checkpoint_bundle
from methods.Wan21.shared.utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os
from methods.Wan21.shared.model import NaiveConsistency



class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        self.model = NaiveConsistency(config, device=self.device)

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=True
        )

        self.model.generator_ema = fsdp_wrap(
            self.model.generator_ema,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=True
        )

        self.model.teacher = fsdp_wrap(
            self.model.teacher,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            cpu_offload=True
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=True
        )


        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader

        dataset = LatentLMDBDataset(
            config.data_path, max_pair=int(1e8))

        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)


        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        if getattr(config, "resume_checkpoint", False):
            manifest, model_state, training_state = load_checkpoint_bundle(
                config.resume_checkpoint
            )
            if manifest.model_id != config.model_id or manifest.stage != config.stage:
                raise ValueError(
                    "resume checkpoint identity mismatch: "
                    f"checkpoint={manifest.model_id}/{manifest.stage} "
                    f"run={config.model_id}/{config.stage}"
                )
            load_fsdp_state_dict(self.model.generator, model_state["generator"])
            if "generator_ema" in model_state:
                load_fsdp_state_dict(
                    self.model.generator_ema, model_state["generator_ema"]
                )
                if self.generator_ema is not None:
                    self.generator_ema.load_full_state_dict(
                        self.model.generator, model_state["generator_ema"]
                    )
            load_fsdp_optim_state_dict(
                self.model.generator,
                self.generator_optimizer,
                training_state["generator_optimizer"],
            )
            self.step = int(training_state["step"])

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None



    def save(self):
        print("Start gathering distributed model states...")
        state_dict = {"generator": fsdp_state_dict(self.model.generator)}
        optimizer_state = fsdp_optim_state_dict(
            self.model.generator, self.generator_optimizer
        )
        if self.generator_ema is not None:
            state_dict["generator_ema"] = self.generator_ema.full_state_dict(
                self.model.generator
            )
        if self.is_main_process:
            path = save_checkpoint_bundle(
                config=self.config,
                step=self.step,
                model_state=state_dict,
                training_state={
                    "step": self.step,
                    "generator_optimizer": optimizer_state,
                    "seed": int(self.config.seed),
                },
            )
            print("Model saved to", path / "model.pt")


    def fwdbwd_one_step(self, batch, clean_latent=None):
        self.model.eval()

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        generator_loss, generator_log_dict = self.model.generator_loss(
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            ema_model = self.generator_ema
        )
        generator_loss.backward()
        generator_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm_generator)

        generator_log_dict.update({"generator_loss": generator_loss,
                                    "generator_grad_norm": generator_grad_norm})

        return generator_log_dict




    def train(self):
        start_step = self.step

        while True:

            self.generator_optimizer.zero_grad(set_to_none=True)

            batch = next(self.dataloader)
            generator_log_dict = self.fwdbwd_one_step(batch, clean_latent=batch["clean_latent"])


            self.generator_optimizer.step()
            if self.generator_ema is not None:
                self.generator_ema.update(self.model.generator)



            # Increment the step since we finished gradient update
            self.step += 1


            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item()
                        }
                    )



                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log(
                            {"per iteration time": current_time - self.previous_time},
                            step=self.step,
                        )
                    self.previous_time = current_time

            max_steps = int(getattr(self.config, "max_train_steps", 0))
            if max_steps and self.step >= max_steps:
                if not self.config.no_save and self.step % self.config.log_iters != 0:
                    self.save()
                barrier()
                break
