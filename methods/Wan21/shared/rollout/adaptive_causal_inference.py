from typing import List, Optional

import torch
import tqdm

from methods.Wan21.shared.rollout.causal_inference import CausalInferencePipeline
from methods.Wan21.shared.rollout.adaptive_causal import AdaptiveCausalRollout
from methods.Wan21.shared.adapter.wan_cache import configure_cache_geometry


class AdaptiveCausalInferencePipeline(CausalInferencePipeline):
    def __init__(self, args, device, generator=None, text_encoder=None, vae=None):
        if generator is None:
            model_kwargs = getattr(args, "model_kwargs", {})
            causal_model_variant = getattr(model_kwargs, "causal_model_variant", None)
            if causal_model_variant != "adaptive":
                raise ValueError(
                    "AdaptiveCausalInferencePipeline requires "
                    "model_kwargs.causal_model_variant='adaptive', "
                    f"got {causal_model_variant!r}"
                )
        super().__init__(args, device, generator=generator, text_encoder=text_encoder, vae=vae)
        if not hasattr(self.generator.model, "adaptive_sparse_forward"):
            raise ValueError("AdaptiveCausalInferencePipeline requires an AdaptiveCausalWanModel generator")
        self._validate_adaptive_config()
        self.adaptive_rollout = AdaptiveCausalRollout(
            generator=self.generator,
            scheduler=self.scheduler,
            denoising_step_list=self.denoising_step_list,
            frame_seq_length=self.frame_seq_length,
            context_noise=getattr(args, "context_noise", 0),
            exit_step_weights=list(getattr(args, "adaptive_exit_step_weights", [0.1, 0.2, 0.3, 0.4])),
            steps_map_path=getattr(args, "adaptive_steps_map_path", None),
            first_chunk_exit_step=getattr(args, "adaptive_first_chunk_exit_step", None),
            verbose=bool(getattr(args, "adaptive_verbose", False)),
        )

    def _validate_adaptive_config(self) -> None:
        exit_steps = list(getattr(self.args, "adaptive_exit_steps", [1000, 750, 500, 250]))
        if exit_steps != [1000, 750, 500, 250]:
            raise ValueError(
                "adaptive few-step inference only supports adaptive_exit_steps=[1000, 750, 500, 250]"
            )
        if len(self.denoising_step_list) != 4:
            raise ValueError("adaptive few-step inference requires exactly 4 denoising steps")
        if self.independent_first_frame:
            raise ValueError("adaptive few-step inference currently supports T2V only")
        solver = getattr(self, "sample_solver", getattr(self.args, "sample_solver", "euler"))
        if solver not in {None, "euler"}:
            raise ValueError("adaptive few-step inference currently supports only Euler-style causal forcing")
        if bool(getattr(self.args, "rollout_use_cfg", False)):
            raise ValueError("adaptive few-step inference is conditional-only and does not support rollout_use_cfg=True")
        sampling_mode = getattr(self.args, "adaptive_exit_sampling_mode", "discrete_weighted")
        if sampling_mode != "discrete_weighted":
            raise ValueError(f"Unsupported adaptive_exit_sampling_mode={sampling_mode!r}")

    @torch.no_grad()
    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        rectified_tf=False,
        adaptive_seq_mode: Optional[str] = None,
        profile_generation: bool = False,
    ) -> torch.Tensor:
        if initial_latent is not None:
            raise ValueError("adaptive few-step inference currently supports T2V only")
        adaptive_seq_mode = adaptive_seq_mode or getattr(self.args, "adaptive_seq_mode", "adaptive")
        if adaptive_seq_mode != "adaptive":
            raise ValueError(f"AdaptiveCausalInferencePipeline expects adaptive_seq_mode='adaptive', got {adaptive_seq_mode!r}")

        profile = profile or profile_generation
        batch_size, num_frames, num_channels, height, width = noise.shape
        if batch_size != 1:
            raise ValueError("adaptive few-step inference currently supports batch_size == 1")
        if num_frames % self.num_frame_per_block != 0:
            raise ValueError("adaptive few-step inference requires num_frames to be divisible by num_frame_per_block")

        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        output = torch.zeros_like(noise)
        streaming_session = None
        if not rectified_tf and getattr(self.vae, "streaming_decode_enabled", False):
            streaming_session = self.vae.create_streaming_session(output.device)

        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)
            block_event_pairs = []
            init_start.record()

        cache_changed, _ = configure_cache_geometry(
            self,
            batch_size=batch_size,
            num_frames=num_frames,
            latent_height=height,
            latent_width=width,
            dtype=noise.dtype,
            device=noise.device,
        )
        self.adaptive_rollout.frame_seq_length = self.frame_seq_length
        if cache_changed:
            self.kv_cache1 = None
            self.crossattn_cache = None
        self._reset_or_initialize_caches(batch_size=batch_size, dtype=noise.dtype, device=noise.device)

        if profile:
            init_end.record()
            diffusion_start.record()

        current_start_frame = 0
        all_num_frames = [self.num_frame_per_block] * (num_frames // self.num_frame_per_block)
        for chunk_index, current_num_frames in enumerate(tqdm.tqdm(all_num_frames)):
            if profile:
                block_start = torch.cuda.Event(enable_timing=True)
                block_end = torch.cuda.Event(enable_timing=True)
                block_start.record()

            noisy_input = noise[:, current_start_frame:current_start_frame + current_num_frames]
            result = self.adaptive_rollout.run_chunk(
                noisy_input=noisy_input,
                conditional_dict=conditional_dict,
                current_start_frame=current_start_frame,
                current_num_frames=current_num_frames,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                chunk_index=chunk_index,
                enable_truncated_grad=False,
            )
            output[:, current_start_frame:current_start_frame + current_num_frames] = result.latents
            if streaming_session is not None:
                streaming_session.submit(
                    output[:, current_start_frame:current_start_frame + current_num_frames]
                )

            if profile:
                block_end.record()
                block_event_pairs.append((block_start, block_end))

            current_start_frame += current_num_frames

        if profile:
            diffusion_end.record()

        if rectified_tf:
            mean = torch.load("laboratory/mean.pt").to(output.device)
            output -= mean

        if streaming_session is not None:
            expected_frames = self.vae.pixel_frames_for_latents(num_frames)
            video = streaming_session.finish(expected_frames=expected_frames)
        else:
            if profile:
                vae_start.record()
            video = self.vae.decode_to_pixel(output, use_cache=False)
            if profile:
                vae_end.record()
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            total_end.record()
            torch.cuda.current_stream(output.device).synchronize()
            init_time = init_start.elapsed_time(init_end)
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            if streaming_session is not None:
                vae_time = streaming_session.elapsed_ms()
                overlap_time = streaming_session.overlap_with(diffusion_start, diffusion_end)
            else:
                vae_time = vae_start.elapsed_time(vae_end)
                overlap_time = 0.0
            total_time = init_start.elapsed_time(total_end)
            block_times = [start.elapsed_time(end) for start, end in block_event_pairs]
            self.last_chunk_times_sec = [value / 1000.0 for value in block_times]
            self.first_chunk_time = self.last_chunk_times_sec[0]
            self.last_generation_time = diffusion_time / 1000.0
            self.last_profile_total_time_sec = total_time / 1000.0
            self.last_profile_diffusion_time_sec = diffusion_time / 1000.0
            self.last_profile_vae_time_sec = vae_time / 1000.0
            self.last_profile_overlap_time_sec = overlap_time / 1000.0
            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Diffusion/VAE overlap time: {overlap_time:.2f} ms")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        return video

    def _reset_or_initialize_caches(self, batch_size, dtype, device) -> None:
        if self.kv_cache1 is None:
            self._initialize_kv_cache(batch_size=batch_size, dtype=dtype, device=device)
            self._initialize_crossattn_cache(batch_size=batch_size, dtype=dtype, device=device)
            return

        for block_index in range(self.num_transformer_blocks):
            self.crossattn_cache[block_index]["is_init"] = False
        for block_index in range(len(self.kv_cache1)):
            self.kv_cache1[block_index]["global_end_index"] = torch.tensor([0], dtype=torch.long, device=device)
            self.kv_cache1[block_index]["local_end_index"] = torch.tensor([0], dtype=torch.long, device=device)
