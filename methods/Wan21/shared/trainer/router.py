from methods.Wan21.shared.router_objective import router_training_contract, validate_router_resume
import gc
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf

from methods.Wan21.shared.model.router import RouterModel
from methods.Wan21.shared.checkpoint import normalize_generator_state_dict
from methods.Wan21.shared.checkpoint import load_checkpoint_bundle, save_checkpoint_bundle
from methods.Wan21.shared.data.dataset import TextDataset, cycle
from methods.Wan21.shared.utils.distributed import barrier, launch_distributed_job
from methods.Wan21.shared.utils.misc import set_seed
from methods.core.runtime import load_checkpoint_manifest


def _router_contract_mismatches(checkpoint_contract, runtime_contract):
    if not checkpoint_contract:
        return [
            {
                "field": "router_runtime",
                "checkpoint": None,
                "runtime": runtime_contract,
                "reason": "checkpoint_metadata_missing",
            }
        ]
    return [
        {
            "field": key,
            "checkpoint": checkpoint_contract.get(key),
            "runtime": value,
            "reason": "value_mismatch",
        }
        for key, value in runtime_contract.items()
        if checkpoint_contract.get(key) != value
    ]


def _manifest_for_checkpoint(path_value):
    if path_value in (None, "", False):
        return None
    path = Path(str(path_value)).expanduser().resolve(strict=False)
    directory = path if path.is_dir() else path.parent
    if not (directory / "manifest.json").is_file():
        return None
    return load_checkpoint_manifest(directory)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0
        self.attempted_iterations = self.step
        self.optimizer_updates = 0
        self.max_optimizer_updates = int(getattr(config, "max_optimizer_updates", 0))
        self.max_train_attempts = int(getattr(config, "max_train_attempts", 0))
        if bool(getattr(config, "router_reset_head", False)) and getattr(config, "resume_checkpoint", False):
            raise ValueError("reset_router cannot be combined with resume_checkpoint")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.disable_wandb = config.disable_wandb

        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()
        set_seed(config.seed + global_rank)
        if self.is_main_process:
            print(f"Router resolved seed: {int(config.seed)}", flush=True)

        if self.is_main_process and not self.disable_wandb:
            wandb_entity = self._normalize_wandb_value(getattr(config, "wandb_entity", None))
            wandb_project = self._normalize_wandb_value(getattr(config, "wandb_project", None))
            wandb_save_dir = self._normalize_wandb_value(getattr(config, "wandb_save_dir", None))
            wandb_key = self._normalize_wandb_value(getattr(config, "wandb_key", None))
            wandb_run_name = (
                self._normalize_wandb_value(getattr(config, "wandb_run_name", None))
                or config.config_name
            )
            wandb.login(host=config.wandb_host, key=wandb_key)
            wandb.init(
                config=self._wandb_config_container(config),
                name=wandb_run_name,
                mode="online",
                entity=wandb_entity,
                project=wandb_project,
                dir=wandb_save_dir,
            )

        self.output_path = config.logdir
        self.model = RouterModel(config, device=self.device)
        router_runtime = {
            "method": "Wan21",
            "family": "wan",
            "num_train_timesteps": int(
                getattr(
                    config,
                    "num_train_timesteps",
                    getattr(config, "num_train_timestep", 1000),
                )
            ),
            "denoising_steps": [
                float(item) for item in config.denoising_step_list
            ],
            "warp_denoising_steps": bool(config.warp_denoising_step),
            "resolved_timesteps": [
                float(item) for item in self.model.denoising_step_list
            ],
            "max_steps": int(self.model.num_router_steps),
            "cache_clear_timestep": float(self.model.context_noise),
            "cache_clear_counts_toward_nfe": False,
            "num_frame_per_block": int(self.model.num_frame_per_block),
            "first_chunk_step": int(self.model.first_chunk_step),
            "router_head_contract": "shared_binary_v1",
            "router_training_contract": router_training_contract(self.model.router_nfe_weight),
            "transition_noise_protocol": "explicit_index",
            "router_reset_head": bool(getattr(config, "router_reset_head", False)),
            "source_checkpoint": str(getattr(config, "generator_ckpt", "")),
            "checkpoint_step_unit": "optimizer_updates",
            "resume_reproducibility": "optimizer_and_counters_only_not_bitexact",

        }
        config.artifact_manifest["router_runtime"] = router_runtime
        checkpoint_runtime_audit = {"loads": []}
        warm_start_manifest = _manifest_for_checkpoint(
            getattr(config, "generator_ckpt", None)
        )
        if warm_start_manifest is not None:
            checkpoint_contract = warm_start_manifest.metadata.get("router_runtime")
            checkpoint_runtime_audit["loads"].append(
                {
                    "kind": "warm_start",
                    "checkpoint_contract": checkpoint_contract,
                    "runtime_contract": router_runtime,
                    "mismatches": _router_contract_mismatches(
                        checkpoint_contract, router_runtime
                    ),
                }
            )
        # Move the backbone without quantizing an initialized/loaded FP32 router.
        head = self._router_head_module()
        self.model.generator.model.patch_exit_router_head = None
        try:
            self.model.generator = self.model.generator.to(device=self.device, dtype=self.dtype)
        finally:
            self.model.generator.model.patch_exit_router_head = head
        self.model.text_encoder = self.model.text_encoder.to(device=self.device, dtype=self.dtype)
        self._move_router_head()
        self._sync_router_head_parameters()
        self.trainable_params = list(self._router_head_module().parameters())
        if not self.trainable_params:
            raise RuntimeError("router trainer found no trainable parameters")
        if self.is_main_process:
            trainable_count = sum(param.numel() for param in self.trainable_params)
            print(f"Router trainable parameters: {trainable_count}", flush=True)
            print(
                f"[router-config] {self.model.backbone_spec.summary()} "
                f"denoising_timesteps={self.model.denoising_step_list.tolist()} "
                f"chunk_frames={self.model.num_frame_per_block} "
                f"training_frames={self.model.num_training_frames} "
                f"text_encoder_dtype={next(self.model.text_encoder.parameters()).dtype}",
                flush=True,
            )

        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )
        self.max_grad_norm = float(getattr(config, "max_grad_norm_router", 1.0))

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
            self.model.generator.load_state_dict(model_state["generator"], strict=True)
            validate_router_resume(training_state.get("router_training_contract"), self.model.router_nfe_weight)
            self.optimizer.load_state_dict(training_state["router_optimizer"])
            self.step = int(training_state["step"])
            if any(
                key not in training_state for key in ("optimizer_updates", "attempted_iterations")
            ):
                raise ValueError("on-policy resume requires explicit optimizer_updates and attempted_iterations")
            self.optimizer_updates = int(training_state.get("optimizer_updates", 0))
            self.attempted_iterations = int(training_state.get("attempted_iterations", self.step))
            if not 0 <= self.optimizer_updates <= self.attempted_iterations:
                raise ValueError("invalid router resume counters")
            if self.step != self.optimizer_updates:
                raise ValueError("on-policy checkpoint step must equal optimizer_updates")
            checkpoint_contract = manifest.metadata.get("router_runtime")
            checkpoint_runtime_audit["loads"].append(
                {
                    "kind": "resume",
                    "checkpoint_contract": checkpoint_contract,
                    "runtime_contract": router_runtime,
                    "mismatches": _router_contract_mismatches(
                        checkpoint_contract, router_runtime
                    ),
                }
            )
        config.artifact_manifest["checkpoint_runtime_audit"] = checkpoint_runtime_audit
        if self.is_main_process and checkpoint_runtime_audit["loads"]:
            print(
                "[router-checkpoint-runtime] "
                + OmegaConf.to_yaml(OmegaConf.create(checkpoint_runtime_audit)).strip(),
                flush=True,
            )

        effective_total_batch_size = int(config.batch_size) * self.world_size
        declared_total_batch_size = int(
            getattr(config, "total_batch_size", effective_total_batch_size)
        )
        if declared_total_batch_size != effective_total_batch_size:
            raise ValueError(
                "router total_batch_size mismatch: "
                f"declared={declared_total_batch_size} "
                f"effective={effective_total_batch_size} "
                f"(batch_size={config.batch_size}, world_size={self.world_size})"
            )
        if self.is_main_process:
            print(
                f"Router world size: {self.world_size}; "
                f"effective total batch size: {effective_total_batch_size}",
                flush=True,
            )

        dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=getattr(config, "num_workers", 4),
            drop_last=True,
        )
        if self.is_main_process:
            print("DATASET SIZE %d" % len(dataset), flush=True)
        self.dataloader = cycle(dataloader)
        self.previous_time = None

    @staticmethod
    def _normalize_wandb_value(value):
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @staticmethod
    def _should_save_initial_checkpoint(config):
        return (
            not bool(config.no_save)
            and bool(getattr(config, "save_initial_checkpoint", True))
            and not getattr(config, "resume_checkpoint", None)
        )

    @classmethod
    def _wandb_config_container(cls, config):
        wandb_config = OmegaConf.to_container(config, resolve=True)
        wandb_config["wandb_key"] = "***"
        wandb_config["wandb_entity"] = cls._normalize_wandb_value(wandb_config.get("wandb_entity"))
        wandb_config["wandb_project"] = cls._normalize_wandb_value(wandb_config.get("wandb_project"))
        return wandb_config

    def _router_head_module(self):
        return self.model.generator.model.patch_exit_router_head

    def _move_router_head(self):
        self._router_head_module().to(device=self.device, dtype=torch.float32)

    def _sync_router_head_parameters(self):
        if not dist.is_initialized():
            return
        router_head = self._router_head_module()
        for param in router_head.parameters():
            dist.broadcast(param.data, src=0)
        for buffer in router_head.buffers():
            dist.broadcast(buffer.data, src=0)

    def _sync_router_grads(self):
        if not dist.is_initialized():
            return
        for param in self.trainable_params:
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(self.world_size)

    def _assert_finite_across_ranks(self, tensors, stage):
        local_nonfinite = torch.zeros((), device=self.device, dtype=torch.int32)
        for tensor in tensors:
            if tensor is not None and not bool(torch.isfinite(tensor).all()):
                local_nonfinite.fill_(1)
                break
        if dist.is_initialized():
            dist.all_reduce(local_nonfinite, op=dist.ReduceOp.MAX)
        if bool(local_nonfinite.item()):
            raise FloatingPointError(f"non-finite router value detected at {stage}")

    def _distributed_max_float(self, value):
        tensor = torch.tensor(float(value), device=self.device, dtype=torch.float32)
        if dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return tensor

    def save(self):
        if self.is_main_process:
            generator_state_dict = self.model.generator.state_dict()
            generator_state_dict = normalize_generator_state_dict(generator_state_dict)
            state_dict = {"generator": generator_state_dict}
            path = save_checkpoint_bundle(
                config=self.config,
                step=self.step,
                model_state=state_dict,
                training_state={
                    "step": self.step,
                    "router_optimizer": self.optimizer.state_dict(),
                    "seed": int(self.config.seed),
                    "optimizer_updates": self.optimizer_updates,
                    "attempted_iterations": self.attempted_iterations,
                    "router_training_contract": router_training_contract(self.model.router_nfe_weight),
                    "router_runtime": OmegaConf.to_container(self.config.artifact_manifest.router_runtime),
                },
            )
            print("Router checkpoint saved to", path / "model.pt", flush=True)

    def train_one_step(self, batch):
        self.model.eval()
        self._router_head_module().train()

        text_prompts = batch["prompts"]
        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size
        noise = torch.randn(image_or_video_shape, device=self.device, dtype=self.dtype)

        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)

        loss, log_dict = self.model.router_loss(noise=noise, conditional_dict=conditional_dict)
        self._assert_finite_across_ranks(
            [loss, *[value for value in log_dict.values() if torch.is_tensor(value)]],
            "loss",
        )
        skip_iter_tensor = log_dict.get("router_skip_iter", loss.new_zeros(())).detach().to(
            device=self.device,
            dtype=torch.float32,
        )
        global_skip_count = skip_iter_tensor.clone()
        if dist.is_initialized():
            dist.all_reduce(global_skip_count, op=dist.ReduceOp.SUM)
        skip_iter = bool(global_skip_count.item() >= self.world_size)
        if skip_iter:
            log_dict["router_loss"] = loss.detach()
            log_dict["router_grad_norm"] = loss.new_zeros(()).detach()
            log_dict["router_grad_abs_max"] = loss.new_zeros(()).detach()
            log_dict["optimizer_updated"] = False
            return log_dict

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._assert_finite_across_ranks(
            [param.grad for param in self.trainable_params if param.grad is not None],
            "backward",
        )
        self._sync_router_grads()
        self._assert_finite_across_ranks(
            [param.grad for param in self.trainable_params if param.grad is not None],
            "gradient-all-reduce",
        )
        grad_abs_max = loss.new_zeros(())
        for param in self.trainable_params:
            if param.grad is not None:
                grad_abs_max = torch.maximum(grad_abs_max, param.grad.detach().abs().max())
        grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable_params, self.max_grad_norm)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Nonfinite router gradient")
        self.optimizer.step()

        log_dict["optimizer_updated"] = True
        log_dict["router_loss"] = loss.detach()
        log_dict["router_grad_norm"] = grad_norm.detach()
        log_dict["router_grad_abs_max"] = grad_abs_max.detach()
        return log_dict

    def _record_attempt(self, optimizer_updated):
        self.attempted_iterations += 1
        self.optimizer_updates += int(optimizer_updated)
        self.step = self.optimizer_updates

    def _training_complete(self):
        if self.max_optimizer_updates:
            return self.optimizer_updates >= self.max_optimizer_updates
        limit = int(getattr(self.config, "max_train_steps", 0))
        return bool(limit and self.step >= limit)

    def train(self):
        last_saved_step = None
        if self._should_save_initial_checkpoint(self.config):
            self.save()
            last_saved_step = self.step
        metrics_path = Path(getattr(self.config, "metrics_path", None) or (Path(self.output_path) / "training" / "metrics.jsonl"))
        if self.is_main_process:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
        while not self._training_complete():
            if self.max_train_attempts and self.attempted_iterations >= self.max_train_attempts:
                if not self.config.no_save:
                    self.save()
                raise RuntimeError(
                    f"router exhausted {self.attempted_iterations} attempts with "
                    f"{self.optimizer_updates}/{self.max_optimizer_updates} optimizer updates"
                )
            batch = next(self.dataloader)
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            step_start_time = time.time()
            log_dict = self.train_one_step(batch)
            torch.cuda.synchronize(self.device)
            log_dict["router_step_time_sec"] = self._distributed_max_float(time.time() - step_start_time)
            log_dict["router_peak_allocated_mb"] = self._distributed_max_float(
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            )
            log_dict["router_peak_reserved_mb"] = self._distributed_max_float(
                torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
            )
            updated = bool(log_dict.pop("optimizer_updated"))
            self._record_attempt(updated)
            # Skip branches may omit metrics: union keys before collectives so
            # ranks always execute the same reductions in the same order.
            metric_keys = sorted(key for key, value in log_dict.items() if torch.is_tensor(value))
            if dist.is_initialized():
                rank_keys = [None] * self.world_size
                dist.all_gather_object(rank_keys, metric_keys)
                metric_keys = sorted(set().union(*rank_keys))
            metrics = {}
            for key in metric_keys:
                value = log_dict.get(key)
                scalar = (value.detach().float().mean().clone() if value is not None
                          else torch.zeros((), device=self.device, dtype=torch.float32))
                if dist.is_initialized():
                    dist.all_reduce(scalar, op=dist.ReduceOp.SUM)
                    scalar.div_(self.world_size)
                metrics[key] = scalar.item()
            metrics.update(
                attempted_iterations=self.attempted_iterations,
                optimizer_updates=self.optimizer_updates,
                optimizer_updated=updated,
                skipped_iterations=self.attempted_iterations - self.optimizer_updates,
            )
            if self.is_main_process:
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics, allow_nan=False) + "\n")
                if not self.disable_wandb:
                    wandb.log(metrics, step=self.attempted_iterations)
                interval = int(getattr(self.config, "progress_log_iters", 20))
                if interval > 0 and self.attempted_iterations % interval == 0:
                    print("Router train " + " ".join(f"{k}={v:.6g}" for k, v in metrics.items()), flush=True)
            if (not self.config.no_save and self.step != last_saved_step
                    and self.step > 0 and self.step % self.config.log_iters == 0):
                self.save()
                last_saved_step = self.step
            if self.attempted_iterations % self.config.gc_interval == 0:
                gc.collect()
                torch.cuda.empty_cache()
            barrier()
        if not self.config.no_save and self.step != last_saved_step:
            self.save()
