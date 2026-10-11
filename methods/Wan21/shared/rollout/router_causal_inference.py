from __future__ import annotations

from typing import List, Optional

import torch
import tqdm

from methods.Wan21.shared.adapter.demo_utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
from methods.Wan21.shared.rollout.causal_inference import CausalInferencePipeline
from methods.Wan21.shared.rollout.router_rollout import RouterRollout
from methods.Wan21.shared.adapter.wan_cache import configure_cache_geometry


class RouterCausalInferencePipeline(CausalInferencePipeline):
    """Adaptive sparse causal inference driven by a patch-wise online exit router."""

    def __init__(self, args, device, generator=None, text_encoder=None, vae=None):
        super().__init__(args, device, generator=generator, text_encoder=text_encoder, vae=vae)
        if not hasattr(self.generator.model, "enable_patch_exit_router_prediction"):
            raise ValueError("generator model does not support router prediction")
        if self.generator.model.patch_exit_router_head is None:
            self.generator.model.enable_patch_exit_router_prediction(
                hidden_dim=getattr(args, "router_head_dim", None),
                initial_exit_bias=float(getattr(args, "router_initial_exit_bias", -2.0)),
            )
        self.generator.model.predict_patch_exit_router = True
        self.router_exit_threshold = float(getattr(args, "router_exit_threshold", 0.5))
        self.num_router_steps = int(len(self.denoising_step_list))
        configured_first_chunk_step = getattr(args, "first_chunk_step", None)
        self.first_chunk_step = (
            self.num_router_steps
            if configured_first_chunk_step is None
            else int(configured_first_chunk_step)
        )
        if not 1 <= self.first_chunk_step <= self.num_router_steps:
            raise ValueError(
                f"first_chunk_step must lie in [1, {self.num_router_steps}]"
            )
        self.router_step_logs = []
        self.last_router_summary = None
        self.last_exit_maps = []
        self.last_chunk_times_sec = []
        self.last_vae_chunk_times_sec = []
        self.last_vae_chunk_latent_frames = []
        self.last_vae_chunk_pixel_frames = []
        self.last_vae_chunk_start_offsets_sec = []
        self.last_vae_chunk_ready_offsets_sec = []
        self.rollout = RouterRollout(
            generator=self.generator,
            scheduler=self.scheduler,
            denoising_step_list=self.denoising_step_list,
            frame_seq_length=self.frame_seq_length,
            context_noise=self.args.context_noise,
            exit_threshold=self.router_exit_threshold,
        )

    def _reset_or_initialize_caches(self, batch_size, dtype, device):
        if self.kv_cache1 is None:
            self._initialize_kv_cache(batch_size=batch_size, dtype=dtype, device=device)
            self._initialize_crossattn_cache(batch_size=batch_size, dtype=dtype, device=device)
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            for cache in self.kv_cache1:
                cache["global_end_index"].zero_()
                cache["local_end_index"].zero_()
        self.rollout.reset_cache_state(self.kv_cache1)

    def _build_router_summary(self):
        total_router_nfe = sum(item["router_nfe"] for item in self.router_step_logs)
        total_dense_reference_nfe = sum(
            item["dense_reference_nfe"] for item in self.router_step_logs
        )
        total_nfe_ratio = (
            total_router_nfe / float(total_dense_reference_nfe)
            if total_dense_reference_nfe
            else 1.0
        )
        patch_counts = {
            step: sum(
                int(item["patch_counts"].get(str(step), item["patch_counts"].get(step, 0)))
                for item in self.router_step_logs
            )
            for step in range(1, self.num_router_steps + 1)
        }
        cache_clear_forwards = sum(
            int(item["cache_clear_forwards"]) for item in self.router_step_logs
        )
        cache_clear_tokens = sum(
            int(item["cache_clear_tokens"]) for item in self.router_step_logs
        )
        summary = {
            "mode": "router",
            "exit_threshold": float(self.router_exit_threshold),
            "max_steps": int(self.num_router_steps),
            "resolved_timesteps": [float(item) for item in self.denoising_step_list],
            "num_chunks": len(self.router_step_logs),
            "chunks": self.router_step_logs,
            "router_nfe": int(total_router_nfe),
            "dense_reference_nfe": int(total_dense_reference_nfe),
            "patch_counts": patch_counts,
            "chunk_latent_frames": int(self.num_frame_per_block),
            "cache_clear_forwards": int(cache_clear_forwards),
            "cache_clear_tokens": int(cache_clear_tokens),
            "total_router_nfe": int(total_router_nfe),
            "total_dense_reference_nfe": int(total_dense_reference_nfe),
            "total_nfe_ratio": float(total_nfe_ratio),
            "total_speedup": (
                float(total_dense_reference_nfe / total_router_nfe)
                if total_router_nfe
                else 1.0
            ),
            "total_nfe_reduction": float(1.0 - total_nfe_ratio),
            "selected_steps": [int(item["selected_step"]) for item in self.router_step_logs],
        }
        if self.num_router_steps == 4:
            summary["full4_nfe"] = int(total_dense_reference_nfe)
            summary["total_full4_nfe"] = int(total_dense_reference_nfe)
        return summary

    @torch.no_grad()
    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        rectified_tf: bool = False,
        report_timing: bool = False,
    ) -> torch.Tensor:
        del rectified_tf
        if initial_latent is not None:
            raise ValueError("router inference currently supports T2V only")

        batch_size, num_frames, num_channels, height, width = noise.shape
        if num_frames % self.num_frame_per_block != 0:
            raise ValueError("router inference requires num_frames divisible by num_frame_per_block")
        num_blocks = num_frames // self.num_frame_per_block

        timing_enabled = report_timing or profile
        if timing_enabled:
            torch.cuda.synchronize()
            generation_start_event = torch.cuda.Event(enable_timing=True)
            generation_end_event = torch.cuda.Event(enable_timing=True)
            total_end_event = torch.cuda.Event(enable_timing=True)
            vae_start_event = torch.cuda.Event(enable_timing=True)
            vae_end_event = torch.cuda.Event(enable_timing=True)
            chunk_event_pairs = []
            generation_start_event.record()

        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder,
                target_device=gpu,
                preserved_memory_gb=gpu_memory_preservation,
            )

        output = torch.zeros(
            [batch_size, num_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype,
        )
        streaming_session = None
        if getattr(self.vae, "streaming_decode_enabled", False):
            streaming_session = self.vae.create_streaming_session(output.device)
        self.router_step_logs = []
        self.last_router_summary = None
        self.last_exit_maps = []
        self.last_chunk_times_sec = []
        self.last_vae_chunk_times_sec = []
        self.last_vae_chunk_latent_frames = []
        self.last_vae_chunk_pixel_frames = []
        self.last_vae_chunk_start_offsets_sec = []
        self.last_vae_chunk_ready_offsets_sec = []
        cache_changed, _ = configure_cache_geometry(
            self,
            batch_size=batch_size,
            num_frames=num_frames,
            latent_height=height,
            latent_width=width,
            dtype=noise.dtype,
            device=noise.device,
        )
        self.rollout.frame_seq_length = self.frame_seq_length
        if cache_changed:
            self.kv_cache1 = None
            self.crossattn_cache = None
        self._reset_or_initialize_caches(batch_size=batch_size, dtype=noise.dtype, device=noise.device)

        current_start_frame = 0
        for chunk_index in tqdm.tqdm(range(num_blocks)):
            if timing_enabled:
                chunk_start_event = torch.cuda.Event(enable_timing=True)
                chunk_end_event = torch.cuda.Event(enable_timing=True)
                chunk_start_event.record()

            current_end = current_start_frame + self.num_frame_per_block
            noisy_input = noise[:, current_start_frame:current_end]
            if chunk_index == 0:
                patch_shape = self.rollout.patch_shape(
                    noisy_input.shape[0],
                    noisy_input.shape[1],
                    noisy_input.shape[3],
                    noisy_input.shape[4],
                )
                steps_map = torch.full(
                    patch_shape,
                    self.first_chunk_step,
                    device=noisy_input.device,
                    dtype=torch.long,
                )
                result = self.rollout.run_chunk_fixed_steps(
                    noisy_input=noisy_input,
                    conditional_dict=conditional_dict,
                    current_start_frame=current_start_frame,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    steps_map=steps_map,
                )
                result.chunk_log.update(
                    {"forced": True, "selected_step": self.first_chunk_step}
                )
            else:
                result = self.rollout.run_router(
                    noisy_input=noisy_input,
                    conditional_dict=conditional_dict,
                    current_start_frame=current_start_frame,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    chunk_index=chunk_index,
                    force_max_steps=False,
                )

            output[:, current_start_frame:current_end] = result.latents
            if timing_enabled:
                chunk_end_event.record()
                chunk_event_pairs.append((chunk_start_event, chunk_end_event))
            if streaming_session is not None:
                streaming_session.submit(output[:, current_start_frame:current_end])
            chunk_log = {
                "chunk": int(chunk_index),
                "start_frame": int(current_start_frame),
                "num_frames": int(self.num_frame_per_block),
                "selected_step": int(result.chunk_log["selected_step"]),
                "patch_avg_step": float(result.chunk_log["patch_avg_step"]),
                "patch_counts": {str(k): int(v) for k, v in result.chunk_log["patch_counts"].items()},
                "mean_max_prob": 1.0,
                "forced": bool(chunk_index == 0),
                "max_steps": int(result.chunk_log["max_steps"]),
                "dense_reference_nfe": int(result.chunk_log["dense_reference_nfe"]),
                "router_nfe": int(result.chunk_log["router_nfe"]),
                "nfe_ratio": float(result.chunk_log["nfe_ratio"]),
                "speedup": float(result.chunk_log["speedup"]),
                "cache_clear_forwards": int(result.chunk_log["cache_clear_forwards"]),
                "cache_clear_tokens": int(result.chunk_log["cache_clear_tokens"]),
            }
            if "full4_nfe" in result.chunk_log:
                chunk_log["full4_nfe"] = int(result.chunk_log["full4_nfe"])
            self.router_step_logs.append(chunk_log)
            self.last_exit_maps.append(result.steps_map.clone())

            current_start_frame = current_end

        if timing_enabled:
            generation_end_event.record()

        # The router workspace is only needed during diffusion rollout.  In
        # particular, its per-layer prefix K/V buffers must not remain resident
        # while the full latent sequence is decoded by the VAE.
        result = None
        noisy_input = None
        if self.kv_cache1:
            self.kv_cache1[0].pop("_router_workspace", None)
        torch.cuda.empty_cache()

        if streaming_session is not None:
            expected_frames = self.vae.pixel_frames_for_latents(num_frames)
            video = streaming_session.finish(expected_frames=expected_frames)
        else:
            if timing_enabled:
                vae_start_event.record()
            video = self.vae.decode_to_pixel(output, use_cache=False)
            if timing_enabled:
                vae_end_event.record()
        video = (video * 0.5 + 0.5).clamp(0, 1)
        if timing_enabled:
            total_end_event.record()
            torch.cuda.current_stream(output.device).synchronize()
            generation_time_ms = generation_start_event.elapsed_time(generation_end_event)
            if streaming_session is not None:
                vae_time_ms = streaming_session.elapsed_ms()
                overlap_time_ms = streaming_session.overlap_with(
                    generation_start_event, generation_end_event
                )
            else:
                vae_time_ms = vae_start_event.elapsed_time(vae_end_event)
                overlap_time_ms = 0.0
            total_time_ms = generation_start_event.elapsed_time(total_end_event)
            self.last_chunk_times_sec = [
                start.elapsed_time(end) / 1000.0 for start, end in chunk_event_pairs
            ]
            if streaming_session is not None:
                self.last_vae_chunk_times_sec = [
                    value / 1000.0 for value in streaming_session.chunk_elapsed_ms()
                ]
                self.last_vae_chunk_latent_frames = [
                    chunk.latent_frames for chunk in streaming_session.chunks
                ]
                self.last_vae_chunk_pixel_frames = [
                    chunk.pixel_frames for chunk in streaming_session.chunks
                ]
                self.last_vae_chunk_start_offsets_sec = [
                    generation_start_event.elapsed_time(chunk.start_event) / 1000.0
                    for chunk in streaming_session.chunks
                ]
                self.last_vae_chunk_ready_offsets_sec = [
                    generation_start_event.elapsed_time(chunk.done_event) / 1000.0
                    for chunk in streaming_session.chunks
                ]
            self.last_generation_time = generation_time_ms / 1000.0
            self.first_chunk_time = self.last_chunk_times_sec[0]
            self.last_profile_total_time_sec = total_time_ms / 1000.0
            self.last_profile_diffusion_time_sec = generation_time_ms / 1000.0
            self.last_profile_vae_time_sec = vae_time_ms / 1000.0
            self.last_profile_overlap_time_sec = overlap_time_ms / 1000.0
            if report_timing:
                print(f"First chunk time: {self.first_chunk_time:.2f}s")
            if profile:
                print("Profiling results:")
                print(f"  - Diffusion generation time: {generation_time_ms:.2f} ms")
                print(f"  - VAE decoding time: {vae_time_ms:.2f} ms")
                print(f"  - Diffusion/VAE overlap time: {overlap_time_ms:.2f} ms")
                print(f"  - Total time: {total_time_ms:.2f} ms")
        self.last_exit_maps = [item.cpu() for item in self.last_exit_maps]
        self.last_router_summary = self._build_router_summary()

        print(
            "[router-summary] "
            f"chunks={self.last_router_summary['num_chunks']} "
            f"steps={self.last_router_summary['selected_steps']} "
            f"router_nfe={self.last_router_summary['total_router_nfe']} "
            f"dense_reference_nfe={self.last_router_summary['total_dense_reference_nfe']} "
            f"speedup={self.last_router_summary['total_speedup']:.4f} "
            f"nfe_reduction={self.last_router_summary['total_nfe_reduction']:.4f}",
            flush=True,
        )
        if return_latents:
            return video, output
        return video
