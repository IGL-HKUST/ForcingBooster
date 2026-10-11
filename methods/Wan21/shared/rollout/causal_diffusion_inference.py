from tqdm import tqdm
from typing import List, Optional
import torch

from methods.Wan21.shared.backbone.wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from methods.Wan21.shared.backbone.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from methods.Wan21.shared.adapter.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from methods.Wan21.shared.adapter.wan_cache import (
    configure_cache_geometry,
    get_wan_backbone_spec,
    make_cross_attention_cache,
    make_self_attention_cache,
    reset_cross_attention_cache,
    reset_self_attention_cache,
)


class CausalDiffusionInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None,
            need_vae = True
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
        if need_vae:
            self.vae = (
                WanVAEWrapper(model_name=model_name, model_path=model_path)
                if vae is None else vae
            )

        # Step 2: Initialize scheduler
        self.num_train_timesteps = int(
            getattr(args, "num_train_timesteps", getattr(args, "num_train_timestep", 1000))
        )
        self.sampling_steps = 50
        self.sample_solver = 'unipc'
        self.shift = args.timestep_shift

        self.backbone_spec = get_wan_backbone_spec(self.generator)
        self.num_transformer_blocks = self.backbone_spec.num_layers
        self.frame_seq_length = 0
        self._cache_signature = None

        self.kv_cache_pos = None
        self.kv_cache_neg = None
        self.crossattn_cache_pos = None
        self.crossattn_cache_neg = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        start_frame_index: Optional[int] = 0,
        return_video=True,
        profile: bool = False,
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
            start_frame_index (int): In long video generation, where does the current window start?
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        elif self.independent_first_frame and initial_latent is None:
            # Using a [1, 4, 4, 4, 4, 4] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
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
            self.kv_cache_pos = self.kv_cache_neg = None
            self.crossattn_cache_pos = self.crossattn_cache_neg = None
        timing_enabled = profile or report_timing
        if timing_enabled:
            torch.cuda.synchronize()
            generation_start_event = torch.cuda.Event(enable_timing=True)
            generation_end_event = torch.cuda.Event(enable_timing=True)
            total_end_event = torch.cuda.Event(enable_timing=True)
            vae_start_event = torch.cuda.Event(enable_timing=True)
            vae_end_event = torch.cuda.Event(enable_timing=True)
            block_event_pairs = []
            generation_start_event.record()
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
        unconditional_dict = self.text_encoder(
            text_prompts=[self.args.negative_prompt] * len(text_prompts)
        )

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )
        streaming_session = None
        if return_video and getattr(self.vae, "streaming_decode_enabled", False):
            streaming_session = self.vae.create_streaming_session(output.device)

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache_pos is None:
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
            reset_cross_attention_cache(self.crossattn_cache_pos)
            reset_cross_attention_cache(self.crossattn_cache_neg)
            reset_self_attention_cache(self.kv_cache_pos)
            reset_self_attention_cache(self.kv_cache_neg)

        # Step 2: Cache context feature
        current_start_frame = start_frame_index
        cache_start_frame = 0
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
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                current_start_frame += 1
                cache_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for block_index in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, cache_start_frame:cache_start_frame + self.num_frame_per_block]
                output[:, cache_start_frame:cache_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                current_start_frame += self.num_frame_per_block
                cache_start_frame += self.num_frame_per_block

        if streaming_session is not None and num_input_frames > 0:
            streaming_session.submit(output[:, :num_input_frames])

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for current_num_frames in all_num_frames:
            if timing_enabled:
                block_start_event = torch.cuda.Event(enable_timing=True)
                block_end_event = torch.cuda.Event(enable_timing=True)
                block_start_event.record()
            noisy_input = noise[
                :, cache_start_frame - num_input_frames:cache_start_frame + current_num_frames - num_input_frames]
            latents = noisy_input

            # Step 3.1: Spatial denoising loop
            sample_scheduler = self._initialize_sample_scheduler(noise)
            for _, t in enumerate(tqdm(sample_scheduler.timesteps)):
                latent_model_input = latents
                timestep = t * torch.ones(
                    [batch_size, current_num_frames], device=noise.device, dtype=torch.float32
                )

                flow_pred_cond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                flow_pred_uncond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=unconditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )

                flow_pred = flow_pred_uncond + self.args.guidance_scale * (
                    flow_pred_cond - flow_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    flow_pred,
                    t,
                    latents,
                    return_dict=False)[0]
                latents = temp_x0

            # Step 3.2: record the model's output
            output[:, cache_start_frame:cache_start_frame + current_num_frames] = latents
            if timing_enabled:
                block_end_event.record()
                block_event_pairs.append((block_start_event, block_end_event))
            if streaming_session is not None:
                streaming_session.submit(
                    output[:, cache_start_frame:cache_start_frame + current_num_frames]
                )

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=conditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length
            )
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=unconditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_neg,
                crossattn_cache=self.crossattn_cache_neg,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length
            )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames
            cache_start_frame += current_num_frames

        if timing_enabled:
            generation_end_event.record()

        # Step 4: Decode the output
        if return_video:
            if streaming_session is not None:
                expected_frames = self.vae.pixel_frames_for_latents(num_output_frames)
                video = streaming_session.finish(expected_frames=expected_frames)
            else:
                if timing_enabled:
                    vae_start_event.record()
                video = self.vae.decode_to_pixel(output)
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
                    start.elapsed_time(end) / 1000.0 for start, end in block_event_pairs
                ]
                self.first_chunk_time = self.last_chunk_times_sec[0]
                self.last_generation_time = generation_time_ms / 1000.0
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

            if return_latents:
                return video, output
            else:
                return video
        else:
            return output

    
    def inference_for_cd(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        record_step_indices: List[int],
        initial_latent: Optional[torch.Tensor] = None,
        start_frame_index: int = 0
    ):
        """
        Causal-forcing inference + record selected diffusion steps (per-chunk) for consistency distillation data.
        Record semantics: record xt BEFORE scheduler.step() at the specified progress_id (index in timesteps list).
        Also record the final latent of each chunk after the denoising loop.

        Returns:
            if return_video:
                (video, output_latents, cd_pack)
            else:
                (output_latents, cd_pack)

        cd_pack:
            {
            "record_step_indices": [...],
            "record_t_values": [t_i ...]  # same for all chunks
            "chunks": [
                {
                    "frame_start": int,
                    "frame_len": int,
                    "latents": Tensor [B, R, T, C, H, W]  (R = len(record_step_indices)+1, last one is final)
                }, ...
            ]
            }
        """
        self.sampling_steps = 48
        batch_size, num_frames, num_channels, height, width = noise.shape

        # ---- block counting (same logic as inference) ----
        if (not self.independent_first_frame) or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block

        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames
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
            self.kv_cache_pos = self.kv_cache_neg = None
            self.crossattn_cache_pos = self.crossattn_cache_neg = None

        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        unconditional_dict = self.text_encoder(text_prompts=[self.args.negative_prompt] * len(text_prompts))

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # ---- Step 1: init/reset caches (same as inference) ----
        if self.kv_cache_pos is None:
            self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
            self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        else:
            reset_cross_attention_cache(self.crossattn_cache_pos)
            reset_cross_attention_cache(self.crossattn_cache_neg)
            reset_self_attention_cache(self.kv_cache_pos)
            reset_self_attention_cache(self.kv_cache_neg)

        # ---- validate record indices against scheduler length ----
        sample_scheduler_probe = self._initialize_sample_scheduler(noise)
        T = len(sample_scheduler_probe.timesteps)
        record_step_indices = sorted(set(int(i) for i in record_step_indices))
        if len(record_step_indices) == 0:
            raise ValueError("record_step_indices must be non-empty")
        if record_step_indices[0] < 0 or record_step_indices[-1] >= T:
            raise ValueError(f"record_step_indices out of range: valid=[0,{T-1}], got={record_step_indices}")
        record_set = set(record_step_indices)

        # ---- Step 2: cache context from initial_latent (same as inference) ----
        current_start_frame = start_frame_index
        cache_start_frame = 0

        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            
            # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
            assert num_input_frames % self.num_frame_per_block == 0
            num_input_blocks = num_input_frames // self.num_frame_per_block

            for block_index in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, cache_start_frame:cache_start_frame + self.num_frame_per_block]
                output[:, cache_start_frame:cache_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                current_start_frame += self.num_frame_per_block
                cache_start_frame += self.num_frame_per_block

        # ---- Step 3: causal-forcing denoising per chunk + record ----
        all_num_frames = [self.num_frame_per_block] * num_blocks

        full_chunk_record = []
        for current_num_frames in all_num_frames:
            # noise slice for current window (same as inference)
            noisy_input = noise[:, cache_start_frame - num_input_frames:cache_start_frame + current_num_frames - num_input_frames]
            latents = noisy_input

            # record list for this chunk
            chunk_records = []

            sample_scheduler = self._initialize_sample_scheduler(noise)
            for progress_id, t in enumerate(tqdm(sample_scheduler.timesteps)):
                if progress_id in record_set:
                    print(f'{progress_id}: {t} saved')
                    chunk_records.append(latents.detach().clone())

                timestep = t * torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.float32)

                flow_pred_cond, _ = self.generator(
                    noisy_image_or_video=latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                flow_pred_uncond, _ = self.generator(
                    noisy_image_or_video=latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )

                flow_pred = flow_pred_uncond + self.args.guidance_scale * (flow_pred_cond - flow_pred_uncond)
                latents = sample_scheduler.step(flow_pred, t, latents, return_dict=False)[0]

            # always append final latent of this chunk (like "-2")
            chunk_records.append(latents.detach().clone())
            chunk_records = torch.stack(chunk_records, dim=1)  # [B, R, T, C, H, W]

            full_chunk_record.append(chunk_records)
            # write output
            output[:, cache_start_frame:cache_start_frame + current_num_frames] = latents

            # rerun at t=0 to update cache using clean context (same as inference)
            timestep0 = torch.zeros([batch_size, current_num_frames], device=noise.device, dtype=torch.float32)
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=conditional_dict,
                timestep=timestep0,
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length
            )
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=unconditional_dict,
                timestep=timestep0,
                kv_cache=self.kv_cache_neg,
                crossattn_cache=self.crossattn_cache_neg,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length
            )

            
            current_start_frame += current_num_frames
            cache_start_frame += current_num_frames

        
        full_chunk_record = torch.cat(full_chunk_record, dim=2)
        # ---- Step 4: decode if needed ----
        
        return full_chunk_record
    
    
    def inference_for_genuine_cd(
        self,
        noisy_input: torch.Tensor,
        conditional_dict = None,
        unconditional_dict = None,
        text_prompts = None,
        initial_latent: Optional[torch.Tensor] = None,
        timestep_idx=0,
        sampling_steps=48,
        chunksize = 3
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noisy_input.shape
        assert num_frames == chunksize
        if initial_latent is not None:
            num_input_frames = initial_latent.shape[1]
            assert num_input_frames % chunksize == 0
            num_output_frames = num_frames + num_input_frames
        else:
            num_output_frames = num_frames
        cache_changed, _ = configure_cache_geometry(
            self,
            batch_size=batch_size,
            num_frames=num_output_frames,
            latent_height=height,
            latent_width=width,
            dtype=noisy_input.dtype,
            device=noisy_input.device,
        )
        if cache_changed:
            self.kv_cache_pos = self.kv_cache_neg = None
            self.crossattn_cache_pos = self.crossattn_cache_neg = None
            
        if conditional_dict is None:
            assert text_prompts is not None
            conditional_dict = self.text_encoder(
                text_prompts=text_prompts
            )
            unconditional_dict = self.text_encoder(
                text_prompts=[self.args.negative_prompt] * len(text_prompts)
            )
            
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noisy_input.device,
            dtype=noisy_input.dtype
        )

        if self.kv_cache_pos is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noisy_input.dtype,
                device=noisy_input.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noisy_input.dtype,
                device=noisy_input.device
            )
        else:
            reset_cross_attention_cache(self.crossattn_cache_pos)
            reset_cross_attention_cache(self.crossattn_cache_neg)
            reset_self_attention_cache(self.kv_cache_pos)
            reset_self_attention_cache(self.kv_cache_neg)

        current_start_frame = 0
        cache_start_frame = 0
        timestep = torch.ones([batch_size, 1], device=noisy_input.device, dtype=torch.int64) * 0
        

        if initial_latent is not None:
            num_input_blocks = num_input_frames // chunksize
            for block_index in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, cache_start_frame:cache_start_frame + chunksize]
                output[:, cache_start_frame:cache_start_frame + chunksize] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                current_start_frame += chunksize
                cache_start_frame += chunksize

    
        latents = noisy_input
        sample_scheduler = self._initialize_sample_scheduler(noisy_input, sampling_steps=sampling_steps)
        t = sample_scheduler.timesteps[timestep_idx]
        latent_model_input = latents
        timestep = t * torch.ones(
            [batch_size, chunksize], device=noisy_input.device, dtype=torch.float32
        )
        flow_pred_cond, _ = self.generator(
            noisy_image_or_video=latent_model_input,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=self.kv_cache_pos,
            crossattn_cache=self.crossattn_cache_pos,
            current_start=current_start_frame * self.frame_seq_length,
            cache_start=cache_start_frame * self.frame_seq_length
        )
        flow_pred_uncond, _ = self.generator(
            noisy_image_or_video=latent_model_input,
            conditional_dict=unconditional_dict,
            timestep=timestep,
            kv_cache=self.kv_cache_neg,
            crossattn_cache=self.crossattn_cache_neg,
            current_start=current_start_frame * self.frame_seq_length,
            cache_start=cache_start_frame * self.frame_seq_length
        )
        flow_pred = flow_pred_uncond + self.args.guidance_scale * (
            flow_pred_cond - flow_pred_uncond)

        latents = sample_scheduler.step(
            flow_pred,
            t,
            latents,
            return_dict=False)[0]
        
        return latents

    

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        if self.frame_seq_length <= 0 or not hasattr(self, "kv_cache_size"):
            raise RuntimeError("cache geometry must be configured from an input latent before allocation")
        self.kv_cache_pos = make_self_attention_cache(
            self.backbone_spec,
            batch_size=batch_size,
            capacity_tokens=self.kv_cache_size,
            dtype=dtype,
            device=device,
        )
        self.kv_cache_neg = make_self_attention_cache(
            self.backbone_spec,
            batch_size=batch_size,
            capacity_tokens=self.kv_cache_size,
            dtype=dtype,
            device=device,
        )

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        self.crossattn_cache_pos = make_cross_attention_cache(
            self.backbone_spec, batch_size=batch_size, dtype=dtype, device=device
        )
        self.crossattn_cache_neg = make_cross_attention_cache(
            self.backbone_spec, batch_size=batch_size, dtype=dtype, device=device
        )

    def _initialize_sample_scheduler(self, noise, sampling_steps=-1):
        if sampling_steps == -1:
            sampling_steps = self.sampling_steps
        if self.sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=noise.device, shift=self.shift)
            self.timesteps = sample_scheduler.timesteps
        elif self.sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, self.shift)
            self.timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=noise.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
        return sample_scheduler
