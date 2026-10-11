from typing import Tuple
from einops import rearrange
from torch import nn
import torch.distributed as dist
import torch

from methods.Wan21.shared.rollout import (
    SelfForcingTrainingPipeline,
    AdaptiveSelfForcingTrainingPipeline,
    TeacherForcingTrainingPipeline,
)
from methods.Wan21.shared.utils.loss import get_denoising_loss
from methods.Wan21.shared.adapter.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from methods.Wan21.shared.adapter.denoising_schedule import resolve_wan_denoising_steps


class BaseModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self._initialize_models(args, device)

        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        if hasattr(args, "denoising_step_list"):
            self.denoising_step_list = resolve_wan_denoising_steps(
                self.scheduler,
                args.denoising_step_list,
                warp=bool(args.warp_denoising_step),
                device=self.device,
            )

        # Optional: separate denoising schedule for the first chunk (block 0).
        # If the config does not provide `denoising_step_list_first_chunk`, all
        # blocks share `denoising_step_list` (backwards compatible).
        # This technique is proposed by [ASD](https://github.com/BigAandSmallq/SAD) for 1/2-step DMD. 
        # We thank ASD for its contribution.
        if hasattr(args, "denoising_step_list_first_chunk") and args.denoising_step_list_first_chunk is not None:
            self.denoising_step_list_first_chunk = resolve_wan_denoising_steps(
                self.scheduler,
                args.denoising_step_list_first_chunk,
                warp=bool(args.warp_denoising_step),
                device=self.device,
            )
        else:
            self.denoising_step_list_first_chunk = None

    def _initialize_models(self, args, device):
        
        self.real_model_name = getattr(args, "real_name", "Wan2.1-T2V-1.3B")
        self.fake_model_name = getattr(args, "fake_name", "Wan2.1-T2V-1.3B")
        self.iscausal = getattr(args, "causal", True)
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
        model_load_dtype = getattr(args, "model_load_dtype", None)
        model_kwargs.setdefault("load_dtype", model_load_dtype)
        if getattr(args, "adaptive_seq_mode", "none") == "adaptive":
            causal_model_variant = model_kwargs.get("causal_model_variant")
            if causal_model_variant != "adaptive":
                raise ValueError(
                    "adaptive_seq_mode='adaptive' requires model_kwargs.causal_model_variant='adaptive', "
                    f"got {causal_model_variant!r}"
                )
        self.generator = WanDiffusionWrapper(**model_kwargs, is_causal=self.iscausal)
        self.generator.model.requires_grad_(True)

        model_path = model_kwargs.get("model_path")
        self.real_score = WanDiffusionWrapper(
            model_name=self.real_model_name,
            model_path=args.real_model_path,
            is_causal=False,
            load_dtype=model_load_dtype,
            num_train_timesteps=int(
                getattr(args, "num_train_timesteps", getattr(args, "num_train_timestep", 1000))
            ),
        )
        self.real_score.model.requires_grad_(False)

        self.fake_score = WanDiffusionWrapper(
            model_name=self.fake_model_name,
            model_path=args.fake_model_path,
            is_causal=False,
            load_dtype=model_load_dtype,
            num_train_timesteps=int(
                getattr(args, "num_train_timesteps", getattr(args, "num_train_timestep", 1000))
            ),
        )
        self.fake_score.model.requires_grad_(True)

        component_model_name = model_kwargs.get("model_name", "Wan2.1-T2V-1.3B")
        self.text_encoder = WanTextEncoder(model_name=component_model_name, model_path=model_path)
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper(model_name=component_model_name, model_path=model_path)
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _get_timestep(
            self,
            min_timestep: int,
            max_timestep: int,
            batch_size: int,
            num_frame: int,
            num_frame_per_block: int,
            uniform_timestep: bool = False
    ) -> torch.Tensor:
        """
        Randomly generate a timestep tensor based on the generator's task type. It uniformly samples a timestep
        from the range [min_timestep, max_timestep], and returns a tensor of shape [batch_size, num_frame].
        - If uniform_timestep, it will use the same timestep for all frames.
        - If not uniform_timestep, it will use a different timestep for each block.
        """
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long
            ).repeat(1, num_frame)
            return timestep
        else:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, num_frame],
                device=self.device,
                dtype=torch.long
            )
            # make the noise level the same within every block
            if self.independent_first_frame:
                # the first frame is always kept the same
                timestep_from_second = timestep[:, 1:]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1, num_frame_per_block)
                timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1)
                timestep = torch.cat([timestep[:, 0:1], timestep_from_second], dim=1)
            else:
                timestep = timestep.reshape(
                    timestep.shape[0], -1, num_frame_per_block)
                timestep[:, :, 1:] = timestep[:, :, 0:1]
                timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep


class SelfForcingModel(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict = None,
        clean_latent = None,
        initial_latent: torch.tensor = None,
        enable_truncated_grad: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - initial_latent: a tensor containing the initial latents [B, F, C, H, W].
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
            - denoised_timestep: an integer
        """
        # Step 1: Sample noise and backward simulate the generator's input
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent
        if self.args.i2v:
            noise_shape = [image_or_video_shape[0], image_or_video_shape[1] - 1, *image_or_video_shape[2:]]
        else:
            noise_shape = image_or_video_shape.copy()

        # During training, the number of generated frames is uniformly sampled up to
        # self.num_training_frames, while preserving the legacy 21-frame training window.
        # Shorter runs use all configured frames instead of constructing an invalid range.
        max_num_frames = self.num_training_frames - 1 if self.args.independent_first_frame else self.num_training_frames
        min_num_frames = min(
            20 if self.args.independent_first_frame else 21,
            max_num_frames,
        )
        assert max_num_frames % self.num_frame_per_block == 0
        assert min_num_frames % self.num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.num_frame_per_block
        min_num_blocks = min_num_frames // self.num_frame_per_block
        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)
        dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.num_frame_per_block
        if self.args.independent_first_frame and initial_latent is None:
            num_generated_frames += 1
            min_num_frames += 1
        # Sync num_generated_frames across all processes
        noise_shape[1] = num_generated_frames
        
        clean_image_or_video = None
        if clean_latent is not None:
            clean_image_or_video = clean_latent.to(self.dtype)
            clean_image_or_video = clean_image_or_video.to(self.device)
            assert clean_image_or_video.shape == tuple(noise_shape), f"{clean_image_or_video.shape} != {tuple(noise_shape)}"

        simulation_output = self._consistency_backward_simulation(
            noise=torch.randn(noise_shape,
                              device=self.device, dtype=self.dtype),
            clean_image_or_video=clean_image_or_video,
            unconditional_dict=unconditional_dict,
            enable_truncated_grad=enable_truncated_grad,
            **conditional_dict,
        )
        rollout_gradient_mask = None
        if len(simulation_output) == 4:
            pred_image_or_video, rollout_gradient_mask, denoised_timestep_from, denoised_timestep_to = simulation_output
        elif len(simulation_output) == 3:
            pred_image_or_video, denoised_timestep_from, denoised_timestep_to = simulation_output
        else:
            raise ValueError(f"Unexpected rollout output length: {len(simulation_output)}")
        training_window_num_frames = min(21, self.num_training_frames)
        training_window_tail_num_frames = training_window_num_frames - 1
        if pred_image_or_video.shape[1] > training_window_num_frames:
            with torch.no_grad():
                # Reencode to get image latent
                latent_to_decode = pred_image_or_video[:, :-training_window_tail_num_frames, ...]
                # Deccode to video
                pixels = self.vae.decode_to_pixel(latent_to_decode)
                frame = pixels[:, -1:, ...].to(self.dtype)
                frame = rearrange(frame, "b t c h w -> b c t h w")
                # Encode frame to get image latent
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
            pred_image_or_video_training_window = torch.cat(
                [image_latent, pred_image_or_video[:, -training_window_tail_num_frames:, ...]],
                dim=1,
            )
        else:
            pred_image_or_video_training_window = pred_image_or_video

        if rollout_gradient_mask is not None:
            gradient_mask = rollout_gradient_mask
            if pred_image_or_video.shape[1] > training_window_num_frames:
                gradient_mask = gradient_mask[:, -training_window_num_frames:]
        elif num_generated_frames != min_num_frames:
            # Currently, we do not use gradient for the first chunk, since it contains image latents
            gradient_mask = torch.ones_like(pred_image_or_video_training_window, dtype=torch.bool)
            if self.args.independent_first_frame:
                gradient_mask[:, :1] = False
            else:
                gradient_mask[:, :self.num_frame_per_block] = False
        else:
            gradient_mask = None

        pred_image_or_video_training_window = pred_image_or_video_training_window.to(self.dtype)
        return pred_image_or_video_training_window, gradient_mask, denoised_timestep_from, denoised_timestep_to

    def _consistency_backward_simulation(
        self,
        noise: torch.Tensor,
        clean_image_or_video: torch.Tensor,
        unconditional_dict: dict = None,
        enable_truncated_grad: bool = True,
        **conditional_dict: dict
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(
            noise=noise,
            clean_image_or_video=clean_image_or_video,
            unconditional_dict=unconditional_dict,
            enable_truncated_grad=enable_truncated_grad,
            **conditional_dict
        )
    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        adaptive_seq_mode = getattr(self.args, "adaptive_seq_mode", "none")
        if adaptive_seq_mode == "adaptive":
            self.inference_pipeline = AdaptiveSelfForcingTrainingPipeline(
                denoising_step_list=self.denoising_step_list,
                scheduler=self.scheduler,
                generator=self.generator,
                num_frame_per_block=self.num_frame_per_block,
                independent_first_frame=self.args.independent_first_frame,
                same_step_across_blocks=self.args.same_step_across_blocks,
                last_step_only=self.args.last_step_only,
                num_max_frames=self.num_training_frames,
                context_noise=self.args.context_noise,
                adaptive_exit_step_weights=list(getattr(self.args, "adaptive_exit_step_weights", [0.1, 0.2, 0.3, 0.4])),
                adaptive_steps_map_path=getattr(self.args, "adaptive_steps_map_path", None),
                adaptive_first_chunk_exit_step=getattr(self.args, "adaptive_first_chunk_exit_step", None),
                adaptive_exit_sampling_mode=getattr(self.args, "adaptive_exit_sampling_mode", "discrete_weighted"),
                adaptive_fallback_prob=list(getattr(self.args, "adaptive_fallback_prob", [0.0, 0.0])),
                adaptive_verbose=getattr(self.args, "adaptive_verbose", False),
            )
        else:
            self.inference_pipeline = SelfForcingTrainingPipeline(
                denoising_step_list=self.denoising_step_list,
                denoising_step_list_first_chunk=self.denoising_step_list_first_chunk,
                scheduler=self.scheduler,
                generator=self.generator,
                num_frame_per_block=self.num_frame_per_block,
                independent_first_frame=self.args.independent_first_frame,
                same_step_across_blocks=self.args.same_step_across_blocks,
                last_step_only=self.args.last_step_only,
                num_max_frames=self.num_training_frames,
                context_noise=self.args.context_noise
            )
        



class TeacherForcingModel(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        clean_latent,
        initial_latent: torch.tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - initial_latent: a tensor containing the initial latents [B, F, C, H, W].
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
            - denoised_timestep: an integer
        """
        # Step 1: Sample noise and backward simulate the generator's input
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        if initial_latent is not None: # never met
            conditional_dict["initial_latent"] = initial_latent
        if self.args.i2v: # never met
            noise_shape = [image_or_video_shape[0], image_or_video_shape[1] - 1, *image_or_video_shape[2:]]
        else:
            noise_shape = image_or_video_shape.copy()

        # During training, the number of generated frames is uniformly sampled up to
        # self.num_training_frames, while preserving the legacy 21-frame training window.
        # Shorter runs use all configured frames instead of constructing an invalid range.
        max_num_frames = self.num_training_frames - 1 if self.args.independent_first_frame else self.num_training_frames
        min_num_frames = min(
            20 if self.args.independent_first_frame else 21,
            max_num_frames,
        )
        assert max_num_frames % self.num_frame_per_block == 0
        assert min_num_frames % self.num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.num_frame_per_block
        min_num_blocks = min_num_frames // self.num_frame_per_block
        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)
        dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.num_frame_per_block
        if self.args.independent_first_frame and initial_latent is None: # never met
            num_generated_frames += 1
            min_num_frames += 1
        # Sync num_generated_frames across all processes
        noise_shape[1] = num_generated_frames

        # ========== TF clean video. todo: add noise for RTF. ==========
        clean_image_or_video = clean_latent.to(self.dtype)
        clean_image_or_video = clean_image_or_video.to(self.device)
        assert clean_image_or_video.shape == tuple(noise_shape), f"{clean_image_or_video.shape} != {tuple(noise_shape)}"

        # ==============================================================
        
        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation_tf(
            noise=torch.randn(noise_shape,
                              device=self.device, dtype=self.dtype),
            clean_image_or_video=clean_image_or_video,
            **conditional_dict,
        )
        training_window_num_frames = min(21, self.num_training_frames)
        training_window_tail_num_frames = training_window_num_frames - 1
        if pred_image_or_video.shape[1] > training_window_num_frames:
            with torch.no_grad():
                # Reencode to get image latent
                latent_to_decode = pred_image_or_video[:, :-training_window_tail_num_frames, ...]
                # Deccode to video
                pixels = self.vae.decode_to_pixel(latent_to_decode)
                frame = pixels[:, -1:, ...].to(self.dtype)
                frame = rearrange(frame, "b t c h w -> b c t h w")
                # Encode frame to get image latent
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
            pred_image_or_video_training_window = torch.cat(
                [image_latent, pred_image_or_video[:, -training_window_tail_num_frames:, ...]],
                dim=1,
            )
        else:
            pred_image_or_video_training_window = pred_image_or_video

        if num_generated_frames != min_num_frames:
            # Currently, we do not use gradient for the first chunk, since it contains image latents
            gradient_mask = torch.ones_like(pred_image_or_video_training_window, dtype=torch.bool)
            if self.args.independent_first_frame:
                gradient_mask[:, :1] = False
            else:
                gradient_mask[:, :self.num_frame_per_block] = False
        else:
            gradient_mask = None

        pred_image_or_video_training_window = pred_image_or_video_training_window.to(self.dtype)
        return pred_image_or_video_training_window, gradient_mask, denoised_timestep_from, denoised_timestep_to
        
    def _consistency_backward_simulation_tf(
        self,
        noise: torch.Tensor,
        clean_image_or_video: torch.Tensor,
        **conditional_dict: dict
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - clean_image_or_video: clean GT video latent with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline_tf()

        return self.inference_pipeline.inference_with_trajectory(
            noise=noise,
            clean_image_or_video=clean_image_or_video,
            **conditional_dict
        )
        
    def _initialize_inference_pipeline_tf(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        self.inference_pipeline = TeacherForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=self.num_training_frames,
            context_noise=self.args.context_noise,
            spatial_self=True
        )
