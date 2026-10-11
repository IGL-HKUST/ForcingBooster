"""ForcingBooster runtime facade for LingBot-World v2 Fast.

The public video layout is NTCHW.  The wrapped upstream-compatible backbone
uses lists of CFHW tensors internally.
"""

from __future__ import annotations

import types
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional

import torch
from torch import nn

from methods.Wan21.shared.adapter.scheduler import SchedulerInterface
from methods.Wan21.shared.adapter.wan_cache import WanBackboneSpec
from methods.Wan21.shared.adapter.wan_wrapper import WanDiffusionWrapper

from methods.LingBotWorld.shared.backbone import (
    build_camera_origin_direction_rays,
    load_lingbot_v2_fast_model,
)
from .scheduler import LingBotFastScheduler


class LingBotDiffusionWrapper(WanDiffusionWrapper):
    """NTCHW adapter used by the common RouterRollout and trainers."""

    def __init__(
        self,
        model: nn.Module,
        *,
        model_name: str = "lingbot-world-v2-causal-fast",
        timestep_shift: float = 10.0,
        num_train_timesteps: int = 1000,
    ) -> None:
        # Deliberately bypass WanDiffusionWrapper.__init__, which would load a
        # standard Wan checkpoint and the wrong cross-attention structure.
        nn.Module.__init__(self)
        self.model = model
        self.model_name = model_name
        self.is_causal = True
        self.uniform_timestep = False
        self.load_dtype = next(model.parameters()).dtype
        self.backbone_spec = WanBackboneSpec.from_model(model, model_name=model_name)
        self.local_attn_size = int(model.local_attn_size)
        self.sink_size = int(model.sink_size)
        self.num_train_timesteps = int(num_train_timesteps)
        if self.num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive")
        self.scheduler = LingBotFastScheduler(
            shift=timestep_shift,
            num_train_timesteps=self.num_train_timesteps,
        )
        self.seq_len = None
        self.post_init()

    @classmethod
    def from_pretrained(
        cls,
        root: str | Path,
        *,
        torch_dtype: Optional[torch.dtype] = torch.bfloat16,
        device: Optional[torch.device | str | int] = None,
        local_attn_size: int = 18,
        sink_size: int = 6,
        load_weights: Optional[bool] = None,
        timestep_shift: float = 10.0,
        num_train_timesteps: int = 1000,
        **from_pretrained_kwargs,
    ) -> "LingBotDiffusionWrapper":
        model = load_lingbot_v2_fast_model(
            root,
            torch_dtype=torch_dtype,
            device=device,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            load_weights=load_weights,
            **from_pretrained_kwargs,
        )
        return cls(
            model,
            timestep_shift=timestep_shift,
            num_train_timesteps=num_train_timesteps,
        )

    def get_scheduler(self) -> SchedulerInterface:
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler
        )
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler
        )
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler
        )
        return scheduler

    def enable_patch_exit_router_prediction(
        self,
        *,
        hidden_dim: int,
        initial_exit_bias: float,
        force_grad: bool = False,
    ) -> torch.nn.Module:
        """Attach the router in the released backbone's parameter dtype."""

        reference = next(self.model.parameters())
        if self.model.patch_exit_router_head is None:
            with torch.device(reference.device):
                self.model.enable_patch_exit_router_prediction(
                    hidden_dim=int(hidden_dim),
                    initial_exit_bias=float(initial_exit_bias),
                )
        head = self.model.patch_exit_router_head
        head.to(device=reference.device, dtype=self.load_dtype)
        head.requires_grad_(True)
        head.force_grad = bool(force_grad)
        self.model.predict_patch_exit_router = True
        return head

    def _model_autocast(self, reference: torch.Tensor):
        """Use the released Fast model's outer BF16 CUDA autocast region."""

        if reference.device.type != "cuda" or self.load_dtype not in {
            torch.bfloat16,
            torch.float16,
        }:
            return nullcontext()
        return torch.amp.autocast("cuda", dtype=self.load_dtype)

    def _prepare_prompt_embeds(
        self,
        prompt_embeds: torch.Tensor,
        reference_latent: torch.Tensor,
    ) -> torch.Tensor:
        # The released T5 context is BF16 even though the latent trajectory is
        # deliberately kept in FP32 outside the DiT.
        return prompt_embeds.to(
            device=reference_latent.device,
            dtype=self.load_dtype,
        )

    @staticmethod
    def _condition_video(
        conditional_dict: dict,
        reference: torch.Tensor,
        *,
        start_frame: int,
    ) -> torch.Tensor:
        condition = conditional_dict.get("i2v_condition")
        if condition is None:
            raise KeyError("conditional_dict must contain i2v_condition")
        if condition.ndim != 5:
            raise ValueError(
                f"i2v_condition must use NTCHW layout, got {tuple(condition.shape)}"
            )
        layout = str(conditional_dict.get("i2v_condition_layout", "NTCHW")).upper()
        if layout == "NTCHW":
            if condition.shape[2] != 20:
                raise ValueError(
                    f"NTCHW i2v_condition requires 20 channels, got {condition.shape[2]}"
                )
            frame_count = reference.shape[1]
            end_frame = start_frame + frame_count
            if condition.shape[1] == frame_count:
                condition = condition.permute(0, 2, 1, 3, 4)
            elif end_frame > condition.shape[1]:
                raise ValueError(
                    f"I2V condition has {condition.shape[1]} frames, cannot slice "
                    f"[{start_frame}:{end_frame}]"
                )
            else:
                condition = condition[:, start_frame:end_frame].permute(0, 2, 1, 3, 4)
        elif layout == "NCTHW":
            if condition.shape[1] != 20:
                raise ValueError(
                    f"NCTHW i2v_condition requires 20 channels, got {condition.shape[1]}"
                )
            frame_count = reference.shape[1]
            end_frame = start_frame + frame_count
            if condition.shape[2] == frame_count:
                pass
            elif end_frame > condition.shape[2]:
                raise ValueError(
                    f"I2V condition has {condition.shape[2]} frames, cannot slice "
                    f"[{start_frame}:{end_frame}]"
                )
            else:
                condition = condition[:, :, start_frame:end_frame]
        else:
            raise ValueError(f"unsupported i2v_condition_layout={layout!r}")
        expected = (reference.shape[0], 20, reference.shape[1], *reference.shape[-2:])
        if tuple(condition.shape) != expected:
            raise ValueError(
                f"i2v_condition shape {tuple(condition.shape)} does not match expected {expected}"
            )
        return condition.to(device=reference.device, dtype=reference.dtype)

    @staticmethod
    def _normalize_camera_layout(camera: torch.Tensor) -> torch.Tensor:
        if isinstance(camera, (list, tuple)):
            normalized = [LingBotDiffusionWrapper._normalize_camera_layout(item) for item in camera]
            return torch.cat(normalized, dim=0)
        if camera.ndim == 4:
            camera = camera.unsqueeze(0)
        if camera.ndim != 5:
            raise ValueError(f"camera rays must be five-dimensional, got {tuple(camera.shape)}")
        if camera.shape[1] in (6, 384):
            return camera
        if camera.shape[2] in (6, 384):
            return camera.permute(0, 2, 1, 3, 4).contiguous()
        if camera.shape[-1] == 6:
            return camera.permute(0, 4, 1, 2, 3).contiguous()
        raise ValueError(
            "camera rays must be NCFHW, NTCHW, or NFHWC with 6 raw or 384 packed channels, "
            f"got {tuple(camera.shape)}"
        )

    def _camera_condition(
        self,
        conditional_dict: dict,
        reference: torch.Tensor,
        *,
        start_frame: int,
    ) -> dict:
        camera = conditional_dict.get("camera_rays")
        internal_name = "camera_rays"
        if camera is None:
            camera = conditional_dict.get("c2ws_plucker_emb")
            internal_name = "c2ws_plucker_emb"
        if camera is None:
            c2ws = conditional_dict.get("camera_c2ws")
            intrinsics = conditional_dict.get("camera_intrinsics")
            if c2ws is None or intrinsics is None:
                raise KeyError(
                    "conditional_dict requires camera_rays, c2ws_plucker_emb, or both "
                    "camera_c2ws and camera_intrinsics"
                )
            frame_count = reference.shape[1]
            end_frame = start_frame + frame_count
            c2ws_time_dim = 0 if c2ws.ndim == 3 else 1
            static_intrinsics = intrinsics.ndim == 1 or (
                intrinsics.ndim == 2 and tuple(intrinsics.shape) == (3, 3)
            )
            intrinsics_time_dim = 0 if intrinsics.ndim == 2 else 1
            c2ws_frame_count = c2ws.shape[c2ws_time_dim]
            if c2ws_frame_count == frame_count:
                pass
            elif c2ws_frame_count < end_frame:
                raise ValueError(
                    f"camera_c2ws has {c2ws_frame_count} frames, cannot slice "
                    f"[{start_frame}:{end_frame}]"
                )
            else:
                c2ws = c2ws.narrow(c2ws_time_dim, start_frame, frame_count)
            if static_intrinsics:
                # A single vector/matrix applies to every frame.
                pass
            elif intrinsics.shape[intrinsics_time_dim] == 1:
                pass
            elif intrinsics.shape[intrinsics_time_dim] >= end_frame:
                # The released implementation applies the trajectory's first
                # intrinsics to every frame before splitting camera rays into
                # chunks.  Preserve that global-first rule here.
                intrinsics = intrinsics.narrow(
                    intrinsics_time_dim, 0, 1
                )
            else:
                raise ValueError(
                    f"camera_intrinsics has {intrinsics.shape[intrinsics_time_dim]} frames, "
                    f"cannot slice [{start_frame}:{end_frame}]"
                )
            camera = build_camera_origin_direction_rays(
                c2ws.to(device=reference.device, dtype=reference.dtype),
                intrinsics.to(device=reference.device, dtype=reference.dtype),
                height=reference.shape[-2] * self.model.camera_spatial_stride[0],
                width=reference.shape[-1] * self.model.camera_spatial_stride[1],
            )
            internal_name = "camera_rays"
        camera = self._normalize_camera_layout(camera)
        end_frame = start_frame + reference.shape[1]
        # Pose/intrinsics inputs were already sliced before ray construction.
        if camera.shape[2] != reference.shape[1]:
            if end_frame > camera.shape[2]:
                raise ValueError(
                    f"camera rays have {camera.shape[2]} frames, cannot slice "
                    f"[{start_frame}:{end_frame}]"
                )
            camera = camera[:, :, start_frame:end_frame]
        return {
            internal_name: camera.to(device=reference.device, dtype=self.load_dtype)
        }

    def _chunk_start_frame(
        self,
        *,
        token_offset: Optional[int],
        latent_height: int,
        latent_width: int,
    ) -> int:
        token_offset = int(token_offset or 0)
        frame_tokens = self.backbone_spec.tokens_per_frame(latent_height, latent_width)
        if token_offset < 0 or token_offset % frame_tokens:
            raise ValueError(
                f"token offset {token_offset} is not aligned to {frame_tokens} tokens/frame"
            )
        return token_offset // frame_tokens

    def _model_timestep(
        self,
        timestep: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        return self._prepare_model_timestep(
            self._prepare_input_timestep(timestep, reference)
        )

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        *,
        cache_start: Optional[int] = None,
        adaptive_forward_mode: Optional[str] = None,
        rope_positions: Optional[torch.Tensor] = None,
        active_token_indices: Optional[torch.Tensor] = None,
        clear_kv_cache: Optional[List[dict]] = None,
        collect_token_indices: Optional[torch.Tensor] = None,
        collect_active_cache: bool = False,
        full_timestep: Optional[torch.Tensor] = None,
        write_cache: bool = True,
        return_patch_exit_logits: bool = False,
        kv_workspace=None,
        cache_only: bool = False,
        precomputed_sigma: Optional[torch.Tensor] = None,
    ):
        if noisy_image_or_video.ndim != 5 or noisy_image_or_video.shape[2] != 16:
            raise ValueError(
                "noisy video must use NTCHW with 16 latent channels, got "
                f"{tuple(noisy_image_or_video.shape)}"
            )
        batch, frames, _, height, width = noisy_image_or_video.shape
        self.seq_len = self.backbone_spec.sequence_length(frames, height, width)
        token_offset = cache_start if adaptive_forward_mode == "sparse" else current_start
        start_frame = self._chunk_start_frame(
            token_offset=token_offset,
            latent_height=height,
            latent_width=width,
        )
        prompt_embeds = self._prepare_prompt_embeds(
            conditional_dict["prompt_embeds"], noisy_image_or_video
        )
        i2v_condition = self._condition_video(
            conditional_dict,
            noisy_image_or_video,
            start_frame=start_frame,
        )
        camera_condition = self._camera_condition(
            conditional_dict,
            noisy_image_or_video,
            start_frame=start_frame,
        )
        model_timestep = self._model_timestep(timestep, noisy_image_or_video)

        if adaptive_forward_mode is not None:
            if adaptive_forward_mode != "sparse":
                raise ValueError(f"invalid adaptive_forward_mode={adaptive_forward_mode!r}")
            if rope_positions is None or active_token_indices is None:
                raise ValueError("adaptive sparse forward requires rope_positions and active_token_indices")
            with self._model_autocast(noisy_image_or_video):
                model_output = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=model_timestep,
                    context=prompt_embeds,
                    seq_len=self.seq_len,
                    y=i2v_condition,
                    dit_cond_dict=camera_condition,
                    adaptive_forward_mode="sparse",
                    rope_positions=rope_positions,
                    active_token_indices=active_token_indices,
                    clear_kv_cache=clear_kv_cache,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    cache_end=int(cache_start or 0),
                    collect_token_indices=collect_token_indices,
                    collect_active_cache=collect_active_cache,
                    full_timestep=(
                        self._model_timestep(full_timestep, noisy_image_or_video)
                        if full_timestep is not None
                        else None
                    ),
                    return_patch_exit_logits=return_patch_exit_logits,
                    kv_workspace=kv_workspace,
                    cache_only=cache_only,
                )
            return self._convert_sparse_output_to_ntchw(
                model_output,
                has_cache=collect_active_cache or collect_token_indices is not None,
                has_logits=return_patch_exit_logits,
            )

        with self._model_autocast(noisy_image_or_video):
            model_output = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=model_timestep,
                context=prompt_embeds,
                seq_len=self.seq_len,
                y=i2v_condition,
                dit_cond_dict=camera_condition,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=int(current_start or 0),
                cache_start=cache_start,
                write_cache=write_cache,
                return_patch_exit_logits=return_patch_exit_logits,
            )
        if return_patch_exit_logits:
            flow_pred, patch_exit_logits = model_output
        else:
            flow_pred = model_output
            patch_exit_logits = None
        flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
        pred_x0 = self._convert_flow_pred_to_x0_video(
            flow_pred=flow_pred,
            xt=noisy_image_or_video,
            timestep=timestep,
            precomputed_sigma=precomputed_sigma,
        )
        if patch_exit_logits is not None:
            return flow_pred, pred_x0, patch_exit_logits
        return flow_pred, pred_x0

    @staticmethod
    def _convert_sparse_output_to_ntchw(output, *, has_cache: bool, has_logits: bool):
        def as_video(value):
            return None if value is None else value.permute(0, 2, 1, 3, 4)

        if has_cache:
            if has_logits:
                video, cache, logits = output
                return as_video(video), cache, logits
            video, cache = output
            return as_video(video), cache
        if has_logits:
            video, logits = output
            return as_video(video), logits
        return as_video(output)

    def write_sparse_chunk_cache(
        self,
        kv_cache: List[dict],
        sparse_kv_cache: List[dict],
        chunk_tokens: int,
        cache_start: int,
    ) -> None:
        self.model.write_sparse_chunk_cache(
            kv_cache=kv_cache,
            sparse_kv_cache=sparse_kv_cache,
            chunk_tokens=chunk_tokens,
            cache_start=cache_start,
        )


__all__ = ["LingBotDiffusionWrapper", "LingBotFastScheduler"]
