from typing import List, Optional
import torch

from methods.Wan21.shared.adapter.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from methods.Wan21.shared.adapter.denoising_schedule import resolve_wan_denoising_steps
from methods.Wan21.shared.adapter.wan_cache import (
    configure_cache_geometry,
    get_wan_backbone_spec,
    make_cross_attention_cache,
    make_self_attention_cache,
    reset_cross_attention_cache,
    reset_self_attention_cache,
    validate_cache_shape,
)

from methods.Wan21.shared.adapter.demo_utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
import tqdm

class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
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
        model_name = getattr(model_kwargs, "model_name", None) or model_kwargs.get(
            "model_name", "Wan2.1-T2V-1.3B"
        )
        model_path = getattr(model_kwargs, "model_path", None) or model_kwargs.get("model_path")
        self.generator = WanDiffusionWrapper(
            **model_kwargs, is_causal=True) if generator is None else generator
        self.text_encoder = (
            WanTextEncoder(model_name=model_name, model_path=model_path)
            if text_encoder is None else text_encoder
        )
        self.vae = (
            WanVAEWrapper(model_name=model_name, model_path=model_path)
            if vae is None else vae
        )

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self._validate_denoising_schedule(args.denoising_step_list, "denoising_step_list")
        self.denoising_step_list = resolve_wan_denoising_steps(
            self.scheduler,
            args.denoising_step_list,
            warp=bool(args.warp_denoising_step),
        )

        # Optional: separate denoising schedule for the first chunk (block 0).
        # If the config does not provide `denoising_step_list_first_chunk`, the
        # first chunk uses the same schedule as the rest (backwards compatible).
        if hasattr(args, "denoising_step_list_first_chunk") and args.denoising_step_list_first_chunk is not None:
            self._validate_denoising_schedule(
                args.denoising_step_list_first_chunk,
                "denoising_step_list_first_chunk",
            )
            self.denoising_step_list_first_chunk = resolve_wan_denoising_steps(
                self.scheduler,
                args.denoising_step_list_first_chunk,
                warp=bool(args.warp_denoising_step),
            )
        else:
            self.denoising_step_list_first_chunk = None

        self.backbone_spec = get_wan_backbone_spec(self.generator)
        self.num_transformer_blocks = self.backbone_spec.num_layers
        self.frame_seq_length = 0
        self._cache_signature = None

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        # Timing state; populated only when `report_timing=True` is passed to
        # `inference()`. Kept as attributes so callers can read them afterward.
        self.last_generation_time = None
        self.first_chunk_time = None
        self.last_chunk_times_sec = []
        self.last_profile_total_time_sec = None
        self.last_profile_diffusion_time_sec = None
        self.last_profile_vae_time_sec = None
        self.last_profile_overlap_time_sec = None
        self.last_vae_chunk_times_sec = []
        self.last_vae_chunk_latent_frames = []
        self.last_vae_chunk_pixel_frames = []
        self.last_vae_chunk_start_offsets_sec = []
        self.last_vae_chunk_ready_offsets_sec = []

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    @staticmethod
    def _validate_denoising_schedule(schedule, name: str) -> None:
        steps = torch.as_tensor(list(schedule), dtype=torch.float64)
        if steps.ndim != 1 or steps.numel() == 0:
            raise ValueError(f"{name} must not be empty")
        if not bool(torch.isfinite(steps).all()):
            raise ValueError(f"{name} must contain only finite values")

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        rectified_tf = False,
        report_timing: bool = False,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        self.last_chunk_times_sec = []
        self.last_profile_total_time_sec = None
        self.last_profile_diffusion_time_sec = None
        self.last_profile_vae_time_sec = None
        self.last_profile_overlap_time_sec = None
        self.last_vae_chunk_times_sec = []
        self.last_vae_chunk_latent_frames = []
        self.last_vae_chunk_pixel_frames = []
        self.last_vae_chunk_start_offsets_sec = []
        self.last_vae_chunk_ready_offsets_sec = []
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            # default here
            # self.independent_first_frame: False
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        if initial_latent is not None and initial_latent.shape[-2:] != noise.shape[-2:]:
            raise ValueError(
                f"initial latent geometry {tuple(initial_latent.shape[-2:])} does not match "
                f"noise geometry {tuple(noise.shape[-2:])}"
            )
        cache_changed, _ = configure_cache_geometry(
            self,
            batch_size=batch_size,
            num_frames=num_output_frames,
            latent_height=height,
            latent_width=width,
            dtype=noise.dtype,
            device=noise.device,
        )
        if cache_changed:
            self.kv_cache1 = None
            self.crossattn_cache = None

        # Optional: start generation timer (excludes VAE decode). Only runs
        # when the caller explicitly opts in.
        if report_timing:
            torch.cuda.synchronize()
            report_generation_start = torch.cuda.Event(enable_timing=True)
            report_generation_end = torch.cuda.Event(enable_timing=True)
            report_generation_start.record()
            first_block_start_event = None
            first_block_end_event = None

        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )
        streaming_session = None
        if not rectified_tf and getattr(self.vae, "streaming_decode_enabled", False):
            streaming_session = self.vae.create_streaming_session(output.device)

        # Set up profiling if requested
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

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            reset_cross_attention_cache(self.crossattn_cache)
            reset_self_attention_cache(self.kv_cache1)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        if streaming_session is not None and num_input_frames > 0:
            streaming_session.submit(output[:, :num_input_frames])

        if profile:
            init_end.record()
            diffusion_start.record()

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for block_index, current_num_frames in enumerate(tqdm.tqdm(all_num_frames)):
            # Optional: time the first block (TTFC). Excludes the KV-cache
            # refresh pass that follows the main denoising.
            if report_timing and block_index == 0:
                first_block_start_event = torch.cuda.Event(enable_timing=True)
                first_block_end_event = torch.cuda.Event(enable_timing=True)
                first_block_start_event.record()

            if profile:
                block_start = torch.cuda.Event(enable_timing=True)
                block_end = torch.cuda.Event(enable_timing=True)
                block_start.record()

            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Select denoising schedule: block 0 may use a dedicated schedule
            # when provided by the config; otherwise all blocks share the same list.
            current_denoising_list = (
                self.denoising_step_list_first_chunk
                if block_index == 0 and self.denoising_step_list_first_chunk is not None
                else self.denoising_step_list
            )

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(current_denoising_list):
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(current_denoising_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    next_timestep = current_denoising_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Record first-chunk latency (denoising only, before KV cache refresh).
            if report_timing and block_index == 0:
                first_block_end_event.record()

            if streaming_session is not None:
                streaming_session.submit(
                    output[:, current_start_frame:current_start_frame + current_num_frames]
                )

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise

            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            if profile:
                block_end.record()
                block_event_pairs.append((block_start, block_end))

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            diffusion_end.record()
        if report_timing:
            report_generation_end.record()
        if rectified_tf:
            mean = torch.load('laboratory/mean.pt').to(output.device)
            std = torch.load('laboratory/std.pt').to(output.device)
            noise = torch.randn_like(output).to(output.device)
            output -= mean

        # Step 4: Decode the output
        if streaming_session is not None:
            expected_frames = self.vae.pixel_frames_for_latents(num_output_frames)
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
        if profile or report_timing:
            torch.cuda.current_stream(output.device).synchronize()

        if report_timing:
            self.last_generation_time = report_generation_start.elapsed_time(report_generation_end) / 1000.0
            if first_block_start_event is not None and first_block_end_event is not None:
                self.first_chunk_time = first_block_start_event.elapsed_time(first_block_end_event) / 1000.0
                print(f"First chunk time: {self.first_chunk_time:.2f}s")

        if profile:
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
            self.last_chunk_times_sec = [float(block_time) / 1000.0 for block_time in block_times]
            self.last_profile_total_time_sec = float(total_time) / 1000.0
            self.last_profile_diffusion_time_sec = float(diffusion_time) / 1000.0
            self.last_profile_vae_time_sec = float(vae_time) / 1000.0
            self.last_profile_overlap_time_sec = float(overlap_time) / 1000.0

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
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        if self.frame_seq_length <= 0 or not hasattr(self, "kv_cache_size"):
            raise RuntimeError("cache geometry must be configured from an input latent before allocation")
        self.kv_cache1 = make_self_attention_cache(
            self.backbone_spec,
            batch_size=batch_size,
            capacity_tokens=self.kv_cache_size,
            dtype=dtype,
            device=device,
        )
        validate_cache_shape(self.kv_cache1, self.backbone_spec, self.kv_cache_size)
        print(
            "[kv-cache] "
            f"layers={len(self.kv_cache1)} self_k_shape={tuple(self.kv_cache1[0]['k'].shape)} "
            f"frame_tokens={self.frame_seq_length} local_attn_frames={self.local_attn_size}",
            flush=True,
        )
        if self.backbone_spec.model_name == "Wan2.1-T2V-14B":
            expected = (40, 40, 128)
            actual = (
                self.backbone_spec.num_layers,
                self.backbone_spec.num_heads,
                self.backbone_spec.head_dim,
            )
            if actual != expected:
                raise RuntimeError(f"Wan2.1-T2V-14B backbone mismatch: expected {expected}, got {actual}")
            if self.frame_seq_length == 1560 and self.kv_cache_size == 32760:
                expected_shape = (batch_size, 32760, 40, 128)
                if tuple(self.kv_cache1[0]["k"].shape) != expected_shape:
                    raise RuntimeError(
                        f"Wan2.1 14B cache shape mismatch: expected {expected_shape}, "
                        f"got {tuple(self.kv_cache1[0]['k'].shape)}"
                    )

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        self.crossattn_cache = make_cross_attention_cache(
            self.backbone_spec,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
