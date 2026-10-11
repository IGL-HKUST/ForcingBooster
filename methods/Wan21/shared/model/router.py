from __future__ import annotations

import torch
import torch.nn.functional as F

from methods.Wan21.shared.rollout.router_training import (
    OnPolicyTrainingRollout, offload_training_history, preserving_exit_map)
from methods.Wan21.shared.adapter.wan_wrapper import WanDiffusionWrapper, WanTextEncoder
from methods.Wan21.shared.adapter.denoising_schedule import resolve_wan_denoising_steps
from methods.Wan21.shared.checkpoint.legacy import load_generator_checkpoint
from methods.Wan21.shared.adapter.wan_cache import (
    configure_cache_geometry,
    make_cross_attention_cache,
    make_self_attention_cache,
)
from methods.Wan21.shared.router_objective import router_objective, validate_router_training_config


class RouterModel(torch.nn.Module):
    """Train a patch-wise binary exit router on top of a frozen adaptive causal generator."""

    def __init__(self, args, device):
        super().__init__()
        self.args = args
        validate_router_training_config(args)
        self.router_reset_head = bool(getattr(args, "router_reset_head", False))
        if self.router_reset_head and getattr(args, "resume_checkpoint", False):
            raise ValueError("reset_router cannot be combined with resume_checkpoint")
        self.device = device
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        self.num_frame_per_block = int(getattr(args, "num_frame_per_block", 3))
        self.num_training_frames = int(getattr(args, "num_training_frames", 21))
        self.frame_seq_length = 0
        self.context_noise = int(getattr(args, "context_noise", 0))
        self.router_nfe_weight = float(getattr(args, "router_nfe_weight", 0.03))
        self.router_exit_threshold = float(getattr(args, "router_exit_threshold", 0.5))
        self.router_initial_exit_bias = float(getattr(args, "router_initial_exit_bias", -2.0))
        configured_first_chunk_step = getattr(args, "first_chunk_step", None)
        self.router_train_chunk_min = int(getattr(args, "router_train_chunk_min", 1))
        self.router_train_chunk_max = int(getattr(args, "router_train_chunk_max", 6))
        configured_train_step_choices = getattr(args, "router_train_step_choices", None)
        model_kwargs = dict(getattr(args, "model_kwargs", {}))
        model_kwargs.setdefault(
            "num_train_timesteps",
            int(
                getattr(
                    args,
                    "num_train_timesteps",
                    getattr(args, "num_train_timestep", 1000),
                )
            ),
        )
        if model_kwargs.get("causal_model_variant", "dense") != "adaptive":
            raise ValueError("RouterModel requires model_kwargs.causal_model_variant='adaptive'")
        self.generator = WanDiffusionWrapper(**model_kwargs, is_causal=True)
        model_name = model_kwargs.get("model_name", "Wan2.1-T2V-1.3B")
        model_path = model_kwargs.get("model_path")
        self.text_encoder = WanTextEncoder(model_name=model_name, model_path=model_path)
        self.backbone_spec = self.generator.backbone_spec
        self.num_transformer_blocks = self.backbone_spec.num_layers
        self.local_attn_size = self.generator.model.local_attn_size
        self._cache_signature = None
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

        self.denoising_step_list = resolve_wan_denoising_steps(
            self.scheduler,
            args.denoising_step_list,
            warp=bool(args.warp_denoising_step),
            device=device,
            minimum_steps=2,
        )
        self.num_router_steps = int(len(self.denoising_step_list))
        self.first_chunk_step = (
            self.num_router_steps
            if configured_first_chunk_step is None
            else int(configured_first_chunk_step)
        )
        self.router_train_step_choices = list(
            range(1, self.num_router_steps)
            if configured_train_step_choices is None
            else (int(step) for step in configured_train_step_choices)
        )
        if not 1 <= self.first_chunk_step <= self.num_router_steps:
            raise ValueError(
                f"first_chunk_step must lie in [1, {self.num_router_steps}]"
            )
        if not self.router_train_step_choices or any(
            not 1 <= step < self.num_router_steps
            for step in self.router_train_step_choices
        ):
            raise ValueError(
                "router_train_step_choices must be a non-empty subset of "
                f"[1, {self.num_router_steps - 1}]"
            )

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self._enable_router_head(args)
        self._load_generator_checkpoint(getattr(args, "generator_ckpt", None))
        if self.router_reset_head:
            # Loading a full generator also loads its old head. Rebuild all head
            # layers afterwards, not merely its output bias.
            self.generator.model.enable_patch_exit_router_prediction(
                hidden_dim=getattr(args, "router_head_dim", None),
                initial_exit_bias=self.router_initial_exit_bias,
            )
        if getattr(args, "gradient_checkpointing", False):
            self.generator.enable_gradient_checkpointing()
        self._freeze_except_router()

        self.rollout = OnPolicyTrainingRollout(
            generator=self.generator,
            scheduler=self.scheduler,
            denoising_step_list=self.denoising_step_list,
            frame_seq_length=self.frame_seq_length,
            context_noise=self.context_noise,
            exit_threshold=self.router_exit_threshold,
        )

    @property
    def router_head(self):
        return self.generator.model.patch_exit_router_head

    def _enable_router_head(self, args):
        if not hasattr(self.generator.model, "enable_patch_exit_router_prediction"):
            raise ValueError("generator model does not support router prediction")
        if self.generator.model.patch_exit_router_head is None:
            self.generator.model.enable_patch_exit_router_prediction(
                hidden_dim=getattr(args, "router_head_dim", None),
                initial_exit_bias=self.router_initial_exit_bias,
            )
        self.generator.model.predict_patch_exit_router = True

    def _load_generator_checkpoint(self, path):
        if not path:
            return
        print(f"Loading pretrained generator from {path}", flush=True)
        load_generator_checkpoint(
            self.generator,
            path,
            allow_missing_prefixes=("model.patch_exit_router_head.",),
        )

    def _freeze_except_router(self):
        self.generator.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.router_head.requires_grad_(True)

    def _initialize_kv_cache(self, batch_size, dtype, device):
        return make_self_attention_cache(
            self.backbone_spec,
            batch_size=batch_size,
            capacity_tokens=self.kv_cache_size,
            dtype=dtype,
            device=device,
        )

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        return make_cross_attention_cache(
            self.backbone_spec, batch_size=batch_size, dtype=dtype, device=device
        )

    def _zero_loss(self):
        zero = None
        for param in self.router_head.parameters():
            term = param.sum() * 0.0
            zero = term if zero is None else zero + term
        if zero is None:
            raise RuntimeError("router head has no parameters")
        return zero

    def _patchwise_mse(self, prediction, reference):
        patch_t, patch_h, patch_w = self.generator.model.patch_size
        diff = (prediction.float() - reference.float()).pow(2).permute(0, 2, 1, 3, 4)
        pooled = F.avg_pool3d(diff, kernel_size=(patch_t, patch_h, patch_w), stride=(patch_t, patch_h, patch_w))
        return pooled.mean(dim=1)

    def _run_prefix_chunk(
        self,
        *,
        noise,
        conditional_dict,
        chunk_index,
        current_start_frame,
        kv_cache,
        crossattn_cache,
    ):
        current_end = current_start_frame + self.num_frame_per_block
        noisy_input = noise[:, current_start_frame:current_end]
        if chunk_index == 0:
            patch_shape = self.rollout.patch_shape(
                noisy_input.shape[0], noisy_input.shape[1], noisy_input.shape[3], noisy_input.shape[4]
            )
            steps_map = torch.full(
                patch_shape,
                self.first_chunk_step,
                device=noisy_input.device,
                dtype=torch.long,
            )
            with torch.no_grad():
                self.rollout.run_chunk_fixed_steps(
                    noisy_input=noisy_input,
                    conditional_dict=conditional_dict,
                    current_start_frame=current_start_frame,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    steps_map=steps_map,
                    write_cache=True,
                )
            return
        with torch.no_grad():
            self.rollout.run_router(
                noisy_input=noisy_input,
                conditional_dict=conditional_dict,
                current_start_frame=current_start_frame,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                chunk_index=chunk_index,
                force_max_steps=False,
            )

    def _sample_chunk_and_step(self, num_chunks):
        min_chunk = max(1, self.router_train_chunk_min)
        max_chunk = min(num_chunks - 1, self.router_train_chunk_max)
        if min_chunk > max_chunk:
            raise ValueError(
                f"Invalid router chunk range [{self.router_train_chunk_min}, "
                f"{self.router_train_chunk_max}] for num_chunks={num_chunks}"
            )
        chunk_index = int(torch.randint(min_chunk, max_chunk + 1, (1,), device=self.device).item())
        step_choices = torch.tensor(self.router_train_step_choices, device=self.device, dtype=torch.long)
        selected_step = int(step_choices[torch.randint(step_choices.numel(), (1,), device=self.device)].item())
        return chunk_index, selected_step

    def router_loss(self, *, noise, conditional_dict):
        self.rollout.clear_training_noise()
        try:
            return self._router_loss(noise=noise, conditional_dict=conditional_dict)
        finally:
            self.rollout.clear_training_noise()

    def _router_loss(self, *, noise, conditional_dict):
        batch_size, num_frames, _, height, width = noise.shape
        if batch_size != 1:
            raise ValueError("RouterModel expects batch_size == 1")
        if num_frames % self.num_frame_per_block != 0:
            raise ValueError("router training expects num_frames divisible by num_frame_per_block")

        configure_cache_geometry(
            self,
            batch_size=batch_size,
            num_frames=max(self.num_training_frames, num_frames),
            latent_height=height,
            latent_width=width,
            dtype=noise.dtype,
            device=noise.device,
        )
        self.rollout.frame_seq_length = self.frame_seq_length

        kv_cache = self._initialize_kv_cache(batch_size, dtype=noise.dtype, device=noise.device)
        crossattn_cache = self._initialize_crossattn_cache(batch_size, dtype=noise.dtype, device=noise.device)
        num_chunks = num_frames // self.num_frame_per_block
        chunk_index, selected_step = self._sample_chunk_and_step(num_chunks)

        current_start_frame = 0
        for prefix_chunk in range(chunk_index):
            self._run_prefix_chunk(
                noise=noise,
                conditional_dict=conditional_dict,
                chunk_index=prefix_chunk,
                current_start_frame=current_start_frame,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
            )
            current_start_frame += self.num_frame_per_block

        offload_training_history(kv_cache, crossattn_cache)
        noisy_input = noise[:, current_start_frame:current_start_frame + self.num_frame_per_block]
        trajectory = self.rollout.record_traj(
            noisy_input=noisy_input, conditional_dict=conditional_dict,
            current_start_frame=current_start_frame, kv_cache=kv_cache,
            crossattn_cache=crossattn_cache, chunk_index=chunk_index,
            force_max_steps=False)
        steps_map = trajectory.steps_map.to(device=noise.device, dtype=torch.long)
        active_mask = steps_map >= selected_step
        total_patches = int(steps_map.numel())
        counts = {
            step: int((steps_map == step).sum().item())
            for step in range(1, self.num_router_steps + 1)
        }
        hard_avg_step = steps_map.float().mean()
        active_ratio = active_mask.float().mean()

        log_dict = {
            "router_chunk_index": torch.tensor(float(chunk_index), device=noise.device),
            "router_selected_step": torch.tensor(float(selected_step), device=noise.device),
            "router_active_ratio": active_ratio.detach(),
            "router_hard_avg_step": hard_avg_step.detach(),
        }
        for step in range(1, self.num_router_steps + 1):
            log_dict[f"router_hard_step{step}_ratio"] = torch.tensor(
                counts[step] / float(total_patches),
                device=noise.device,
                dtype=torch.float32,
            )

        if not bool(active_mask.any()):
            zero = self._zero_loss()
            _, metrics = router_objective(
                logits=zero.reshape(1)[:0], exit_mse=zero.reshape(1)[:0],
                continue_steps=zero.reshape(1)[:0], selected_step=selected_step,
                total_patches=total_patches, nfe_weight=self.router_nfe_weight)
            log_dict.update(metrics)
            return zero, log_dict

        logits, active_indices = self.rollout.logits_at_step(
            noisy_input=noisy_input,
            conditional_dict=conditional_dict,
            current_start_frame=current_start_frame,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            steps_map=steps_map,
            selected_step=selected_step,
        )
        expected_active_indices = active_mask.reshape(-1).nonzero(
            as_tuple=False
        ).flatten().long()
        if not torch.equal(
            active_indices.to(device=noise.device, dtype=torch.long),
            expected_active_indices,
        ):
            raise RuntimeError("hard routing and logits replay active indices differ")

        reference_steps_map = torch.full_like(steps_map, self.num_router_steps)
        exit_steps_map = preserving_exit_map(steps_map, selected_step)
        continue_steps_map = steps_map.clone()
        continue_target_step = min(selected_step + 1, self.num_router_steps)
        continue_steps_map[active_mask] = torch.maximum(
            continue_steps_map[active_mask],
            torch.full_like(continue_steps_map[active_mask], continue_target_step),
        )

        with torch.no_grad():
            exit_result = self.rollout.run_chunk_fixed_steps(
                noisy_input=noisy_input,
                conditional_dict=conditional_dict,
                current_start_frame=current_start_frame,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                steps_map=exit_steps_map,
                write_cache=False,
            )
            reference_result = self.rollout.run_chunk_fixed_steps(
                noisy_input=noisy_input,
                conditional_dict=conditional_dict,
                current_start_frame=current_start_frame,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                steps_map=reference_steps_map,
                write_cache=False,
            )

        exit_mse_map = self._patchwise_mse(exit_result.latents, reference_result.latents).reshape(-1)
        active_indices = active_indices.to(device=noise.device, dtype=torch.long)
        exit_mse = exit_mse_map.index_select(0, active_indices)
        continue_steps = continue_steps_map.reshape(-1).index_select(0, active_indices).float()

        loss, metrics = router_objective(
            logits=logits, exit_mse=exit_mse, continue_steps=continue_steps,
            selected_step=selected_step, total_patches=total_patches,
            nfe_weight=self.router_nfe_weight)
        log_dict.update(metrics)
        return loss, log_dict
