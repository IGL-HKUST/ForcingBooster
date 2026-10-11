"""Two-phase LingBot router objective built on the shared Wan router rollout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from methods.Wan21.shared.adapter.wan_cache import configure_cache_geometry
from methods.Wan21.shared.model.router import RouterModel
from methods.LingBotWorld.shared.adapter.scheduler import resolve_timesteps
from .rollout import LingBotTrainingRollout
from methods.Wan21.shared.rollout.router_training import (
    TrajectoryRecord, offload_training_history, preserving_exit_map)
from methods.Wan21.shared.router_objective import router_objective, validate_router_training_config


@dataclass
class RouterTargetState:
    """Detached hard-routing targets retained until the final logits forward."""

    noisy_input: torch.Tensor
    current_start_frame: int
    kv_cache: list[dict[str, Any]]
    crossattn_cache: list[dict[str, Any]]
    steps_map: torch.Tensor
    selected_step: int
    active_indices: torch.Tensor
    exit_mse: torch.Tensor
    continue_steps: torch.Tensor
    total_patches: int
    metrics: dict[str, torch.Tensor]
    trajectory: TrajectoryRecord
    skip: bool = False


class LingBotRouterTrainingModel(RouterModel):
    """Generator-only FSDP root with target and logits forward phases.

    T5 and the conditioning VAE live in each rank's trainer, outside this
    generator-only FSDP root.
    """

    def __init__(self, args: Any, *, generator: torch.nn.Module, device: torch.device) -> None:
        torch.nn.Module.__init__(self)
        validate_router_training_config(args)
        self.args = args
        self.device = device
        self.dtype = torch.bfloat16 if bool(args.mixed_precision) else torch.float32
        self.num_frame_per_block = int(args.num_frame_per_block)
        self.num_training_frames = int(args.num_training_frames)
        self.frame_seq_length = 0
        self.context_noise = 0
        self.router_nfe_weight = float(args.router_nfe_weight)
        self.router_exit_threshold = float(args.router_exit_threshold)
        self.router_initial_exit_bias = float(args.router_initial_exit_bias)
        self.first_chunk_step = int(args.first_chunk_step)
        self.router_train_chunk_min = int(args.router_train_chunk_min)
        self.router_train_chunk_max = int(args.router_train_chunk_max)
        self.router_train_step_choices = tuple(int(item) for item in args.router_train_step_choices)
        num_router_steps = len(tuple(args.timestep_indices))
        if num_router_steps < 2:
            raise ValueError("LingBot router requires at least two timestep indices")
        if not 1 <= self.first_chunk_step <= num_router_steps:
            raise ValueError(
                f"first_chunk_step must lie in [1, {num_router_steps}]"
            )
        if not self.router_train_step_choices or any(
            not 1 <= item < num_router_steps for item in self.router_train_step_choices
        ):
            raise ValueError(
                "router_train_step_choices must be a non-empty subset of "
                f"[1, {num_router_steps - 1}]"
            )

        self.generator = generator
        self.backbone_spec = generator.backbone_spec
        self.num_transformer_blocks = self.backbone_spec.num_layers
        self.local_attn_size = int(generator.model.local_attn_size)
        self.speculative_cache_frames = int(args.speculative_cache_frames)
        self._cache_signature = None
        self.scheduler = generator.get_scheduler()
        self.denoising_step_list = resolve_timesteps(
            self.scheduler,
            indices=tuple(int(item) for item in args.timestep_indices),
            device=device,
        )
        self.num_router_steps = int(len(self.denoising_step_list))
        generator.model.num_frame_per_block = self.num_frame_per_block
        generator.enable_patch_exit_router_prediction(
            hidden_dim=int(args.router_head_dim),
            initial_exit_bias=self.router_initial_exit_bias,
            force_grad=True,
        )
        generator.requires_grad_(False)
        self.router_head.requires_grad_(True)
        self.rollout = LingBotTrainingRollout(
            generator=generator,
            scheduler=self.scheduler,
            denoising_step_list=self.denoising_step_list,
            frame_seq_length=0,
            context_noise=0,
            exit_threshold=self.router_exit_threshold,
        )

    def forward(self, phase: str, **kwargs: Any) -> Any:
        try:
            if phase == "targets":
                return self.prepare_targets(**kwargs)
            if phase == "logits":
                return self.logits_for_targets(**kwargs)
            raise ValueError(f"unknown LingBot router forward phase {phase!r}")
        finally:
            self.rollout.clear_training_noise()

    @torch.no_grad()
    def prepare_targets(
        self,
        *,
        noise: torch.Tensor,
        conditional_dict: dict[str, torch.Tensor],
        chunk_index: int,
        selected_step: int,
        noise_generator: torch.Generator,
    ) -> RouterTargetState:
        self.rollout.clear_training_noise()
        self.rollout.set_noise_generator(noise_generator)
        batch_size, num_frames, channels, height, width = noise.shape
        if batch_size != 1 or channels != 16:
            raise ValueError("LingBot router noise must have shape [1,F,16,H,W]")
        if num_frames % self.num_frame_per_block:
            raise ValueError("router latent frames must contain complete chunks")
        num_chunks = num_frames // self.num_frame_per_block
        if not 1 <= int(chunk_index) < num_chunks:
            raise ValueError(f"chunk_index={chunk_index} is invalid for {num_chunks} chunks")
        if int(selected_step) not in self.router_train_step_choices:
            raise ValueError(f"selected_step={selected_step} is not enabled")

        configure_cache_geometry(
            self,
            batch_size=batch_size,
            num_frames=max(self.num_training_frames, num_frames),
            latent_height=height,
            latent_width=width,
            dtype=self.dtype,
            device=noise.device,
            physical_capacity_frames=(
                self.local_attn_size + self.speculative_cache_frames
                if self.local_attn_size != -1
                else None
            ),
            attention_window_frames=(
                self.local_attn_size if self.local_attn_size != -1 else None
            ),
        )
        self.rollout.frame_seq_length = self.frame_seq_length
        kv_cache = self._initialize_kv_cache(batch_size, dtype=self.dtype, device=noise.device)
        crossattn_cache = self._initialize_crossattn_cache(
            batch_size, dtype=self.dtype, device=noise.device
        )

        current_start_frame = 0
        for prefix_chunk in range(int(chunk_index)):
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
        noisy_input = noise[:, current_start_frame : current_start_frame + self.num_frame_per_block]
        # Record only after prefix chunks: all target branches share this
        # on-policy history and an explicitly indexed transition-noise bank.
        trajectory = self.rollout.record_traj(
            noisy_input=noisy_input,
            conditional_dict=conditional_dict,
            current_start_frame=current_start_frame,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            chunk_index=int(chunk_index),
            force_max_steps=False,
            commit_cache=False,
        )
        steps_map = torch.as_tensor(
            trajectory.steps_map, device=noise.device, dtype=torch.long
        ).clone()

        active_mask = steps_map >= int(selected_step)
        total_patches = int(steps_map.numel())
        counts = {
            step: int((steps_map == step).sum().item())
            for step in range(1, self.num_router_steps + 1)
        }
        metrics = {
            "router_chunk_index": noise.new_tensor(float(chunk_index), dtype=torch.float32),
            "router_selected_step": noise.new_tensor(float(selected_step), dtype=torch.float32),
            "router_active_ratio": active_mask.float().mean(),
            "router_hard_avg_step": steps_map.float().mean(),
        }
        for step, count in counts.items():
            metrics[f"router_hard_step{step}_ratio"] = noise.new_tensor(
                count / float(total_patches), dtype=torch.float32
            )

        active_indices = active_mask.reshape(-1).nonzero(as_tuple=False).flatten().long()
        local_has_active = bool(active_indices.numel() > 0)
        global_has_active = torch.tensor(
            int(local_has_active), device=noise.device, dtype=torch.int32
        )
        if dist.is_initialized():
            dist.all_reduce(global_has_active, op=dist.ReduceOp.MAX)
        if not bool(global_has_active.item()):
            empty = noise.new_empty((0,), dtype=torch.float32)
            return RouterTargetState(
                noisy_input=noisy_input,
                current_start_frame=current_start_frame,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                steps_map=steps_map,
                selected_step=int(selected_step),
                active_indices=active_indices,
                exit_mse=empty,
                continue_steps=empty,
                total_patches=total_patches,
                metrics=metrics,
                trajectory=trajectory,
                skip=True,
            )

        reference_steps_map = torch.full_like(steps_map, self.num_router_steps)
        exit_steps_map = preserving_exit_map(steps_map, int(selected_step))
        continue_steps_map = steps_map.clone()
        continue_target_step = min(int(selected_step) + 1, self.num_router_steps)
        continue_steps_map[active_mask] = torch.maximum(
            continue_steps_map[active_mask],
            torch.full_like(continue_steps_map[active_mask], continue_target_step),
        )
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
        exit_mse_map = self._patchwise_mse(
            exit_result.latents, reference_result.latents
        ).reshape(-1)
        return RouterTargetState(
            noisy_input=noisy_input,
            current_start_frame=current_start_frame,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            steps_map=steps_map,
            selected_step=int(selected_step),
            active_indices=active_indices,
            exit_mse=exit_mse_map.index_select(0, active_indices).detach(),
            continue_steps=continue_steps_map.reshape(-1)
            .index_select(0, active_indices)
            .float()
            .detach(),
            total_patches=total_patches,
            metrics=metrics,
            trajectory=trajectory,
            skip=not local_has_active,
        )

    def logits_for_targets(
        self,
        *,
        targets: RouterTargetState,
        conditional_dict: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if targets.skip:
            global_has_active = torch.zeros((), device=targets.noisy_input.device, dtype=torch.int32)
            if dist.is_initialized():
                dist.all_reduce(global_has_active, op=dist.ReduceOp.MAX)
            if not bool(global_has_active.item()):
                zero = self._zero_loss().reshape(1, 1)
                return zero, targets.active_indices
        elif dist.is_initialized():
            global_has_active = torch.ones((), device=targets.noisy_input.device, dtype=torch.int32)
            dist.all_reduce(global_has_active, op=dist.ReduceOp.MAX)
        if self.rollout.noise_generator is None:
            raise RuntimeError("router logits replay requires a per-sample noise generator")
        self.rollout.bind_traj(targets.trajectory)
        logits, active_indices = self.rollout.logits_at_step(
            noisy_input=targets.noisy_input,
            conditional_dict=conditional_dict,
            current_start_frame=targets.current_start_frame,
            kv_cache=targets.kv_cache,
            crossattn_cache=targets.crossattn_cache,
            steps_map=targets.steps_map,
            selected_step=targets.selected_step,
        )
        if not torch.equal(active_indices, targets.active_indices):
            raise RuntimeError("target and logits active patch indices differ")
        return logits, active_indices


def compose_router_loss(*, logits, targets, nfe_weight=0.03):
    loss, objective_metrics = router_objective(
        logits=logits.reshape(-1)[:0] if targets.skip else logits,
        exit_mse=targets.exit_mse, continue_steps=targets.continue_steps,
        selected_step=targets.selected_step, total_patches=targets.total_patches,
        nfe_weight=nfe_weight)
    return loss, dict(targets.metrics, **objective_metrics)


__all__ = ["LingBotRouterTrainingModel", "RouterTargetState", "compose_router_loss"]
