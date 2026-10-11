import gc
import logging
from methods.Wan21.shared.data.dataset import cycle
from methods.Wan21.shared.data.dataset import TextDataset
from methods.Wan21.shared.utils.distributed import (
    EMA_FSDP,
    fsdp_optim_state_dict,
    load_fsdp_optim_state_dict,
    load_fsdp_state_dict,
    fsdp_wrap,
    fsdp_state_dict,
    launch_distributed_job,
)
from methods.Wan21.shared.utils.misc import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
from methods.Wan21.shared.model import DMD
import torch
from methods.Wan21.shared.checkpoint.legacy import load_generator_checkpoint
from methods.Wan21.shared.checkpoint import (
    load_checkpoint_bundle,
    save_checkpoint_bundle,
    save_generator_checkpoint,
)
import wandb
import time
import os

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
        if config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        if getattr(config, "generator_ckpt", False) and not getattr(
            config, "resume_checkpoint", False
        ):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            generator_checkpoint_mmap = bool(
                getattr(config, "generator_checkpoint_mmap", False)
            )
            if self.is_main_process:
                print(
                    "[checkpoint-load] "
                    f"mmap={generator_checkpoint_mmap} path={config.generator_ckpt}",
                    flush=True,
                )
            load_generator_checkpoint(
                self.model.generator,
                config.generator_ckpt,
                mmap=generator_checkpoint_mmap,
            )
            if self.is_main_process:
                print("Parent checkpoint loaded as a model-only warm start at step 0", flush=True)

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=False,
            backward_prefetch=config.fsdp_backward_prefetch,
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            cpu_offload=False,
            backward_prefetch=config.fsdp_backward_prefetch,
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            cpu_offload=False,
            backward_prefetch=config.fsdp_backward_prefetch,
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False),
            backward_prefetch=config.fsdp_backward_prefetch,
        )

        if self.is_main_process:
            print(
                "[dmd-runtime] "
                f"world_size={self.world_size} "
                f"generator={config.model_kwargs.model_name} "
                f"real_score={config.real_name} fake_score={config.fake_name} "
                f"model_load_dtype={getattr(config, 'model_load_dtype', None)} "
                f"backward_prefetch={config.fsdp_backward_prefetch} "
                f"generator_param_dtype={next(self.model.generator.parameters()).dtype} "
                f"real_score_param_dtype={next(self.model.real_score.parameters()).dtype} "
                f"fake_score_param_dtype={next(self.model.fake_score.parameters()).dtype}",
                flush=True,
            )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        dataset = TextDataset(config.data_path)
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
            load_fsdp_state_dict(self.model.fake_score, model_state["fake_score"])
            load_fsdp_optim_state_dict(
                self.model.generator,
                self.generator_optimizer,
                training_state["generator_optimizer"],
            )
            load_fsdp_optim_state_dict(
                self.model.fake_score,
                self.critic_optimizer,
                training_state["critic_optimizer"],
            )
            self.step = int(training_state["step"])
            if "generator_ema" in model_state:
                if self.generator_ema is None:
                    self.generator_ema = EMA_FSDP(
                        self.model.generator, decay=config.ema_weight
                    )
                self.generator_ema.load_full_state_dict(
                    self.model.generator, model_state["generator_ema"]
                )

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None
        self.checkpoint_mode = str(getattr(config, "checkpoint_mode", "full_bundle"))
        if self.checkpoint_mode not in {"full_bundle", "generator_only"}:
            raise ValueError(
                "checkpoint_mode must be 'full_bundle' or 'generator_only', "
                f"got {self.checkpoint_mode!r}"
            )

    def save(self):
        print("Start gathering distributed model states...")
        generator_state = fsdp_state_dict(self.model.generator)
        if self.checkpoint_mode == "generator_only":
            if self.is_main_process:
                path = save_generator_checkpoint(
                    config=self.config,
                    step=self.step,
                    generator_state=generator_state,
                )
                print("Generator-only model saved to", path / "model.pt")
            return

        state_dict = {"generator": generator_state}
        state_dict["fake_score"] = fsdp_state_dict(self.model.fake_score)
        generator_optimizer_state = fsdp_optim_state_dict(
            self.model.generator, self.generator_optimizer
        )
        critic_optimizer_state = fsdp_optim_state_dict(
            self.model.fake_score, self.critic_optimizer
        )
        if self.generator_ema is not None:
            state_dict["generator_ema"] = self.generator_ema.full_state_dict(self.model.generator)

        if self.is_main_process:
            path = save_checkpoint_bundle(
                config=self.config,
                step=self.step,
                model_state=state_dict,
                training_state={
                    "step": self.step,
                    "generator_optimizer": generator_optimizer_state,
                    "critic_optimizer": critic_optimizer_state,
                    "seed": int(self.config.seed),
                },
            )
            print("Model saved to", path / "model.pt")

    def save_critic(self):
        print("Start gathering distributed model states...")

        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)


        state_dict = critic_state_dict

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def fwdbwd_one_step(self, batch, train_generator, clean_latent=None):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        image_latent = None

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
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent
            )

            torch.cuda.empty_cache()
            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent
        )

        torch.cuda.empty_cache()
        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)

        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": critic_grad_norm})

        return critic_log_dict


    def train(self):
        start_step = self.step

        while True:
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)

                batch = next(self.dataloader)
                generator_log_dict = self.fwdbwd_one_step(batch, True)

                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)




            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            batch = next(self.dataloader)
            critic_log_dict = self.fwdbwd_one_step(batch, False)

            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)
                progress_log_iters = int(getattr(self.config, "progress_log_iters", 20))
                if progress_log_iters > 0 and self.step % progress_log_iters == 0:
                    wandb_loss_dict.update(
                        {
                            "peak_allocated_mb": torch.cuda.max_memory_allocated(self.device) / (1024 ** 2),
                            "peak_reserved_mb": torch.cuda.max_memory_reserved(self.device) / (1024 ** 2),
                        }
                    )
                    metrics = " ".join(
                        f"{key}={value:.6g}" for key, value in sorted(wandb_loss_dict.items())
                    )
                    print(f"DMD train step={self.step} {metrics}", flush=True)

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
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            if getattr(self.config, "max_train_steps", 0) and self.step >= self.config.max_train_steps:
                if not self.config.no_save and self.step % self.config.log_iters != 0:
                    self.save()
                break
