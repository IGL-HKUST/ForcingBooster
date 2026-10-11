from typing import List, Optional

import torch
import torch.distributed as dist

from methods.Wan21.shared.rollout.adaptive_causal import AdaptiveCausalRollout
from methods.Wan21.shared.adapter.scheduler import SchedulerInterface
from methods.Wan21.shared.adapter.wan_wrapper import WanDiffusionWrapper
from methods.Wan21.shared.adapter.wan_cache import (
    configure_cache_geometry,
    make_cross_attention_cache,
    make_self_attention_cache,
)


class AdaptiveSelfForcingTrainingPipeline:
    def __init__(
        self,
        denoising_step_list: List[int],
        scheduler: SchedulerInterface,
        generator: WanDiffusionWrapper,
        num_frame_per_block=3,
        independent_first_frame: bool = False,
        same_step_across_blocks: bool = True,
        last_step_only: bool = False,
        num_max_frames: int = 21,
        context_noise: int = 0,
        adaptive_exit_step_weights: Optional[list[float]] = None,
        adaptive_steps_map_path: Optional[str] = None,
        adaptive_first_chunk_exit_step: Optional[int] = None,
        adaptive_exit_sampling_mode: str = "discrete_weighted",
        adaptive_fallback_prob: Optional[list[float]] = None,
        adaptive_verbose: bool = False,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        if independent_first_frame:
            raise ValueError("AdaptiveSelfForcingTrainingPipeline currently supports T2V/chunkwise only")
        if adaptive_exit_sampling_mode != "discrete_weighted":
            raise ValueError(f"Unsupported adaptive_exit_sampling_mode={adaptive_exit_sampling_mode!r}")
        if not hasattr(generator.model, "adaptive_sparse_forward"):
            raise ValueError("AdaptiveSelfForcingTrainingPipeline requires an AdaptiveCausalWanModel generator")

        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = torch.as_tensor(list(denoising_step_list), dtype=torch.float32)
        if len(self.denoising_step_list) != 4:
            raise ValueError("adaptive self-forcing training currently requires exactly 4 denoising steps")

        self.backbone_spec = self.generator.backbone_spec
        self.num_transformer_blocks = self.backbone_spec.num_layers
        self.frame_seq_length = 0
        self.local_attn_size = self.generator.model.local_attn_size
        self._cache_signature = None
        self.num_frame_per_block = int(num_frame_per_block)
        self.generator.model.num_frame_per_block = self.num_frame_per_block
        self.context_noise = int(context_noise)
        self.independent_first_frame = bool(independent_first_frame)
        self.same_step_across_blocks = bool(same_step_across_blocks)
        self.last_step_only = bool(last_step_only)
        self.adaptive_fallback_prob = adaptive_fallback_prob or [0.0, 0.0]
        if len(self.adaptive_fallback_prob) != 2:
            raise ValueError("adaptive_fallback_prob must contain [first_chunk_prob, later_chunk_prob]")
        self.adaptive_fallback_prob = [float(prob) for prob in self.adaptive_fallback_prob]
        if any(prob < 0.0 or prob > 1.0 for prob in self.adaptive_fallback_prob):
            raise ValueError("adaptive_fallback_prob values must be in [0, 1]")
        self.num_max_frames = int(num_max_frames)
        self.kv_cache1 = None
        self.crossattn_cache = None
        self.rollout = AdaptiveCausalRollout(
            generator=generator,
            scheduler=scheduler,
            denoising_step_list=self.denoising_step_list,
            frame_seq_length=self.frame_seq_length,
            context_noise=context_noise,
            exit_step_weights=adaptive_exit_step_weights or [0.1, 0.2, 0.3, 0.4],
            steps_map_path=adaptive_steps_map_path,
            first_chunk_exit_step=adaptive_first_chunk_exit_step,
            ensure_all_exit_steps=True,
            verbose=adaptive_verbose,
        )

    def generate_and_sync_exit_steps(self, num_blocks, device):
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            if self.last_step_only:
                exit_steps = torch.full((num_blocks,), 4, device=device, dtype=torch.long)
            elif self.same_step_across_blocks:
                value = torch.randint(1, 5, (1,), device=device, dtype=torch.long)
                exit_steps = value.expand(num_blocks).clone()
            else:
                exit_steps = torch.randint(1, 5, (num_blocks,), device=device, dtype=torch.long)
        else:
            exit_steps = torch.empty(num_blocks, dtype=torch.long, device=device)
        if dist.is_initialized():
            dist.broadcast(exit_steps, src=0)
        return exit_steps.tolist()

    def generate_and_sync_fallback_flags(self, num_blocks, device):
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            probs = torch.full(
                (num_blocks,),
                self.adaptive_fallback_prob[1],
                device=device,
                dtype=torch.float32,
            )
            if num_blocks > 0:
                probs[0] = self.adaptive_fallback_prob[0]
            fallback_flags = (torch.rand(num_blocks, device=device) < probs).to(dtype=torch.long)
        else:
            fallback_flags = torch.empty(num_blocks, dtype=torch.long, device=device)
        if dist.is_initialized():
            dist.broadcast(fallback_flags, src=0)
        return [bool(value) for value in fallback_flags.tolist()]

    def inference_with_trajectory(
        self,
        noise: torch.Tensor,
        clean_image_or_video: torch.Tensor = None,
        initial_latent: Optional[torch.Tensor] = None,
        unconditional_dict: Optional[dict] = None,
        return_sim_step: bool = False,
        enable_truncated_grad: bool = True,
        **conditional_dict,
    ) -> torch.Tensor:
        del clean_image_or_video, initial_latent, unconditional_dict
        if return_sim_step:
            raise ValueError("AdaptiveSelfForcingTrainingPipeline does not support return_sim_step")

        batch_size, num_frames, num_channels, height, width = noise.shape
        if batch_size != 1:
            raise ValueError("AdaptiveSelfForcingTrainingPipeline currently supports batch_size == 1")
        if num_frames % self.num_frame_per_block != 0:
            raise ValueError("num_frames must be divisible by num_frame_per_block for adaptive rollout")

        num_blocks = num_frames // self.num_frame_per_block
        configure_cache_geometry(
            self,
            batch_size=batch_size,
            num_frames=max(self.num_max_frames, num_frames),
            latent_height=height,
            latent_width=width,
            dtype=noise.dtype,
            device=noise.device,
        )
        self.rollout.frame_seq_length = self.frame_seq_length
        output = torch.zeros_like(noise)
        gradient_mask = torch.zeros_like(noise, dtype=torch.bool) if enable_truncated_grad else None
        self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)

        exit_steps = self.generate_and_sync_exit_steps(num_blocks, device=noise.device)
        fallback_flags = self.generate_and_sync_fallback_flags(num_blocks, device=noise.device)
        selected_from = None
        selected_to = None
        current_start_frame = 0
        for block_index in range(num_blocks):
            current_num_frames = self.num_frame_per_block
            noisy_input = noise[:, current_start_frame:current_start_frame + current_num_frames]
            result = self.rollout.run_chunk(
                noisy_input=noisy_input,
                conditional_dict=conditional_dict,
                current_start_frame=current_start_frame,
                current_num_frames=current_num_frames,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                chunk_index=block_index,
                grad_exit_step=exit_steps[block_index],
                enable_truncated_grad=enable_truncated_grad,
                force_uniform_exit_step=fallback_flags[block_index],
                full_gradient_mask=fallback_flags[block_index],
            )
            output[:, current_start_frame:current_start_frame + current_num_frames] = result.latents
            if result.denoised_timestep_from is not None:
                selected_from = result.denoised_timestep_from
                selected_to = result.denoised_timestep_to
            if enable_truncated_grad and result.gradient_mask is not None:
                gradient_mask[:, current_start_frame:current_start_frame + current_num_frames] = result.gradient_mask
            current_start_frame += current_num_frames

        if gradient_mask is not None and not bool(gradient_mask.any()):
            gradient_mask = None

        if enable_truncated_grad:
            return output, gradient_mask, selected_from, selected_to
        return output, selected_from, selected_to

    def _initialize_kv_cache(self, batch_size, dtype, device):
        self.kv_cache1 = make_self_attention_cache(
            self.backbone_spec,
            batch_size=batch_size,
            capacity_tokens=self.kv_cache_size,
            dtype=dtype,
            device=device,
        )

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        self.crossattn_cache = make_cross_attention_cache(
            self.backbone_spec, batch_size=batch_size, dtype=dtype, device=device
        )
