import types
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import torch
from torch import nn

from methods.Wan21.shared.adapter.scheduler import SchedulerInterface, FlowMatchScheduler
from methods.Wan21.shared.backbone.wan.modules.tokenizers import HuggingfaceTokenizer
from methods.Wan21.shared.backbone.wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock
from methods.Wan21.shared.backbone.wan.modules.vae import _video_vae
from methods.Wan21.shared.backbone.wan.modules.t5 import umt5_xxl
from methods.Wan21.shared.backbone.wan.modules.causal_model import CausalWanModel
from methods.Wan21.shared.backbone.wan.modules.causal_adaptive_model import AdaptiveCausalWanModel
from methods.Wan21.shared.adapter.wan_cache import WanBackboneSpec


DEFAULT_WAN_MODEL_NAME = "Wan2.1-T2V-1.3B"


_MODEL_LOAD_DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def resolve_model_load_dtype(value):
    """Resolve the optional public model-load dtype without changing defaults."""

    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    key = str(value).strip().lower()
    try:
        return _MODEL_LOAD_DTYPES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(_MODEL_LOAD_DTYPES))
        raise ValueError(
            f"Unsupported Wan model load dtype {value!r}; expected one of: {supported}"
        ) from exc


class WanTextEncoder(torch.nn.Module):
    def __init__(
        self,
        model_name: str = DEFAULT_WAN_MODEL_NAME,
        model_path: str | os.PathLike[str] | None = None,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        model_path = Path(model_path)

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load(model_path / "models_t5_umt5-xxl-enc-bf16.pth",
                       map_location='cpu', weights_only=False)
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=str(model_path / "google" / "umt5-xxl"), seq_len=512, clean='whitespace')

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


@dataclass
class WanStreamingDecodeChunk:
    pixels: torch.Tensor
    start_event: torch.cuda.Event
    done_event: torch.cuda.Event
    latent_frames: int
    pixel_frames: int


class WanStreamingDecodeSession:
    """Stateful Wan VAE cached decoder running on a dedicated CUDA stream."""

    def __init__(
        self,
        wrapper: "WanVAEWrapper",
        device: torch.device | str | int,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        self.wrapper = wrapper
        if isinstance(device, int):
            self.device = torch.device("cuda", device)
        else:
            self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if self.device.type != "cuda":
            raise ValueError("Streaming Wan VAE decode requires a CUDA device")
        if wrapper.device != self.device:
            raise ValueError(
                f"Wan VAE model must be on {self.device} before streaming decode; "
                f"got {wrapper.device}"
            )

        self.stream = stream if stream is not None else torch.cuda.Stream(device=self.device)
        self._chunks: list[WanStreamingDecodeChunk] = []
        self._submitted_latents = 0
        self._closed = False
        self._decode_start_event: Optional[torch.cuda.Event] = None
        self._decode_end_event: Optional[torch.cuda.Event] = None
        self._finish_start_event: Optional[torch.cuda.Event] = None
        self.wrapper.model.clear_cache()

    @property
    def chunks(self) -> tuple[WanStreamingDecodeChunk, ...]:
        return tuple(self._chunks)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def submitted_latents(self) -> int:
        return self._submitted_latents

    @property
    def expected_pixel_frames(self) -> int:
        return self.wrapper.pixel_frames_for_latents(self._submitted_latents)

    def submit(self, latent_chunk: torch.Tensor) -> WanStreamingDecodeChunk:
        if self._closed:
            raise RuntimeError("Cannot submit to a finished Wan VAE streaming session")
        if latent_chunk.ndim != 5:
            raise ValueError(f"Expected NTCHW latent chunk, got shape={tuple(latent_chunk.shape)}")
        if latent_chunk.device != self.device:
            raise ValueError(f"Expected latent chunk on {self.device}, got {latent_chunk.device}")
        if latent_chunk.shape[0] != 1:
            raise ValueError("Streaming Wan VAE decode currently requires batch size 1")
        if latent_chunk.shape[1] == 0:
            raise ValueError("Cannot submit an empty latent chunk")
        if latent_chunk.shape[2] != self.wrapper.model.z_dim:
            raise ValueError(
                f"Expected {self.wrapper.model.z_dim} latent channels, "
                f"got {latent_chunk.shape[2]}"
            )

        producer_stream = torch.cuda.current_stream(self.device)
        ready_event = torch.cuda.Event()
        ready_event.record(producer_stream)
        previous_total = self.expected_pixel_frames
        latent_frames = int(latent_chunk.shape[1])

        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            self.stream.wait_event(ready_event)
            latent_chunk.record_stream(self.stream)
            chunk_start_event = torch.cuda.Event(enable_timing=True)
            chunk_start_event.record(self.stream)
            if self._decode_start_event is None:
                self._decode_start_event = chunk_start_event

            pixels = self.wrapper.decode_to_pixel(latent_chunk, use_cache=True)
            self._submitted_latents += latent_frames
            expected_chunk_frames = self.expected_pixel_frames - previous_total
            if pixels.shape[1] != expected_chunk_frames:
                raise RuntimeError(
                    "Wan VAE streaming frame-count mismatch: "
                    f"expected {expected_chunk_frames}, got {pixels.shape[1]}"
                )

            done_event = torch.cuda.Event(enable_timing=True)
            done_event.record(self.stream)

        chunk = WanStreamingDecodeChunk(
            pixels=pixels,
            start_event=chunk_start_event,
            done_event=done_event,
            latent_frames=latent_frames,
            pixel_frames=int(pixels.shape[1]),
        )
        self._chunks.append(chunk)
        self._decode_end_event = done_event
        return chunk

    def collect(
        self,
        chunk: WanStreamingDecodeChunk,
        consumer_stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        if consumer_stream is None:
            consumer_stream = torch.cuda.current_stream(self.device)
        consumer_stream.wait_event(chunk.done_event)
        chunk.pixels.record_stream(consumer_stream)
        return chunk.pixels

    def release(self, chunk: WanStreamingDecodeChunk) -> None:
        self._chunks.remove(chunk)

    def finish(self, expected_frames: Optional[int] = None) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("Wan VAE streaming session has already been finished")
        if not self._chunks:
            raise RuntimeError("Cannot finish a Wan VAE session with no submitted latents")

        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            self._finish_start_event = torch.cuda.Event(enable_timing=True)
            self._finish_start_event.record(self.stream)
            video = torch.cat([chunk.pixels for chunk in self._chunks], dim=1)
            done_event = torch.cuda.Event(enable_timing=True)
            done_event.record(self.stream)

        self._decode_end_event = done_event
        self._closed = True
        consumer_stream = torch.cuda.current_stream(self.device)
        consumer_stream.wait_event(done_event)
        video.record_stream(consumer_stream)

        expected_frames = self.expected_pixel_frames if expected_frames is None else expected_frames
        if video.shape[1] != expected_frames:
            raise RuntimeError(f"Expected {expected_frames} decoded frames, got {video.shape[1]}")
        return video

    def synchronize(self) -> None:
        if self._decode_end_event is not None:
            self._decode_end_event.synchronize()

    def elapsed_ms(self) -> float:
        if not self._chunks:
            return 0.0
        self.synchronize()
        elapsed = float(sum(
            chunk.start_event.elapsed_time(chunk.done_event)
            for chunk in self._chunks
        ))
        if self._finish_start_event is not None and self._decode_end_event is not None:
            elapsed += float(self._finish_start_event.elapsed_time(self._decode_end_event))
        return elapsed

    def chunk_elapsed_ms(self) -> list[float]:
        if not self._chunks:
            return []
        self.synchronize()
        return [
            float(chunk.start_event.elapsed_time(chunk.done_event))
            for chunk in self._chunks
        ]

    def span_ms(self) -> float:
        if self._decode_start_event is None or self._decode_end_event is None:
            return 0.0
        self.synchronize()
        return float(self._decode_start_event.elapsed_time(self._decode_end_event))

    def overlap_with(
        self,
        producer_start: torch.cuda.Event,
        producer_end: torch.cuda.Event,
    ) -> float:
        if self._decode_start_event is None or self._decode_end_event is None:
            return 0.0
        self.synchronize()
        producer_end.synchronize()
        producer_duration = float(producer_start.elapsed_time(producer_end))
        overlap = 0.0
        for chunk in self._chunks:
            decoder_start = float(producer_start.elapsed_time(chunk.start_event))
            decoder_end = float(producer_start.elapsed_time(chunk.done_event))
            overlap += max(
                0.0,
                min(producer_duration, decoder_end) - max(0.0, decoder_start),
            )
        return overlap

    def abort(self) -> None:
        self.stream.synchronize()
        self.wrapper.model.clear_cache()
        self._chunks.clear()
        self._closed = True


class WanVAEWrapper(torch.nn.Module):
    supports_streaming_decode = True

    def __init__(
        self,
        streaming_decode: bool = False,
        model_name: str = DEFAULT_WAN_MODEL_NAME,
        model_path: str | os.PathLike[str] | None = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.streaming_decode_enabled = bool(streaming_decode)
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = _video_vae(
            pretrained_path=str(Path(model_path) / "Wan2.1_VAE.pth"),
            z_dim=16,
        ).eval().requires_grad_(False)
        self._active_session: Optional[WanStreamingDecodeSession] = None
        self._decode_stream: Optional[torch.cuda.Stream] = None
        self._decode_stream_device: Optional[torch.device] = None

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @staticmethod
    def pixel_frames_for_latents(latent_frames: int) -> int:
        return 0 if latent_frames == 0 else latent_frames * 4 - 3

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def create_streaming_session(
        self,
        device: Optional[torch.device | str | int] = None,
    ) -> WanStreamingDecodeSession:
        if self._active_session is not None and not self._active_session.closed:
            self._active_session.abort()
        target_device = self.device if device is None else device
        if isinstance(target_device, int):
            target_device = torch.device("cuda", target_device)
        else:
            target_device = torch.device(target_device)
        if target_device.type == "cuda" and target_device.index is None:
            target_device = torch.device("cuda", torch.cuda.current_device())
        if target_device.type != "cuda":
            raise ValueError("Streaming Wan VAE decode requires a CUDA device")
        if self._decode_stream is None or self._decode_stream_device != target_device:
            self._decode_stream = torch.cuda.Stream(device=target_device)
            self._decode_stream_device = target_device
        self._active_session = WanStreamingDecodeSession(
            self,
            target_device,
            stream=self._decode_stream,
        )
        return self._active_session

    def clear_cache(self) -> None:
        if self._active_session is not None:
            if self._active_session.closed:
                self._active_session.synchronize()
                self.model.clear_cache()
            else:
                self._active_session.abort()
        else:
            self.model.clear_cache()
        self._active_session = None


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name=DEFAULT_WAN_MODEL_NAME,
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0,
            causal_model_variant="dense",
            model_path=None,
            load_dtype=None,
            num_train_timesteps=1000,
    ):
        super().__init__()

        self.is_causal = is_causal
        self.model_name = model_name
        self.load_dtype = resolve_model_load_dtype(load_dtype)
        self.num_train_timesteps = int(num_train_timesteps)
        if self.num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive")
        model_path = Path(model_path)
        pretrained_kwargs = {}
        if self.load_dtype is not None:
            pretrained_kwargs["torch_dtype"] = self.load_dtype
        if is_causal:
            if causal_model_variant not in {"dense", "adaptive"}:
                raise ValueError(f"Unsupported causal_model_variant={causal_model_variant!r}")
            causal_model_cls = {
                "dense": CausalWanModel,
                "adaptive": AdaptiveCausalWanModel,
            }[causal_model_variant]
            self.model = causal_model_cls.from_pretrained(
                str(model_path),
                local_attn_size=local_attn_size,
                sink_size=sink_size,
                **pretrained_kwargs,
            )
        else:
            if causal_model_variant != "dense":
                raise ValueError("causal_model_variant is only supported for causal models")
            self.model = WanModel.from_pretrained(str(model_path), **pretrained_kwargs)
        self.model.eval()
        self.backbone_spec = WanBackboneSpec.from_model(self.model, model_name=model_name)

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            num_inference_steps=self.num_train_timesteps,
            num_train_timesteps=self.num_train_timesteps,
            shift=timestep_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.scheduler.set_timesteps(self.num_train_timesteps, training=True)

        # Retained as observable runtime state for older callers. Each forward
        # recomputes it from the actual latent tensor before invoking the model.
        self.seq_len = None
        self.post_init()

    def _prepare_input_timestep(
        self,
        timestep: torch.Tensor,
        noisy_image_or_video: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.uniform_timestep:
            if timestep.ndim == 2:
                if noisy_image_or_video is not None:
                    batch_size, num_frame, _, height, width = noisy_image_or_video.shape
                    patch_t, patch_h, patch_w = self.model.patch_size
                    expected_tokens = (num_frame // patch_t) * (height // patch_h) * (width // patch_w)
                    if timestep.shape == (batch_size, expected_tokens):
                        return timestep
                return timestep[:, 0]
            return timestep
        return timestep

    @staticmethod
    def _prepare_model_timestep(input_timestep: torch.Tensor) -> torch.Tensor:
        if input_timestep.ndim == 4:
            return input_timestep.reshape(input_timestep.shape[0], -1)
        return input_timestep

    @staticmethod
    def _prepare_prompt_embeds(
        prompt_embeds: torch.Tensor,
        reference_latent: torch.Tensor,
    ) -> torch.Tensor:
        # FSDP mixed precision keeps FP32 master parameters outside the nested
        # module's forward and materializes BF16 compute weights only on entry.
        # Reading ``next(model.parameters()).dtype`` here therefore observes
        # the wrong dtype.  The latent is already normalized by the trainer and
        # is the stable compute-dtype reference for every execution mode.
        return prompt_embeds.to(
            device=reference_latent.device,
            dtype=reference_latent.dtype,
        )

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def adding_cls_branch(
        self,
        atten_dim=None,
        num_class=4,
        time_embed_dim=0,
        gan_ffn_dim=8192,
    ) -> None:
        atten_dim = self.backbone_spec.dim if atten_dim is None else int(atten_dim)
        self._cls_pred_branch = nn.Sequential(
            # Input: [B, 384, 21, 60, 104]
            nn.LayerNorm(atten_dim * 3 + time_embed_dim),
            nn.Linear(atten_dim * 3 + time_embed_dim, atten_dim),
            nn.SiLU(),
            nn.Linear(atten_dim, num_class)
        )
        self._cls_pred_branch.requires_grad_(True)
        num_registers = 3
        self._register_tokens = RegisterTokens(num_registers=num_registers, dim=atten_dim)
        self._register_tokens.requires_grad_(True)

        gan_ca_blocks = []
        for _ in range(num_registers):
            block = GanAttentionBlock(
                dim=atten_dim,
                ffn_dim=gan_ffn_dim,
                num_heads=self.backbone_spec.num_heads,
            )
            gan_ca_blocks.append(block)
        self._gan_ca_blocks = nn.ModuleList(gan_ca_blocks)
        self._gan_ca_blocks.requires_grad_(True)
        # self.has_cls_branch = True

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    def _expand_patch_timestep_to_latent(self, timestep: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        batch_size, num_frame, _, height, width = latent.shape
        patch_t, patch_h, patch_w = self.model.patch_size
        patch_frame = num_frame // patch_t
        patch_height = height // patch_h
        patch_width = width // patch_w
        expected_tokens = patch_frame * patch_height * patch_width
        if timestep.ndim == 4:
            expected = (batch_size, patch_frame, patch_height, patch_width)
            if tuple(timestep.shape) != expected:
                raise ValueError(f"Expected patch timestep shape {expected}, got {tuple(timestep.shape)}")
            patch_values = timestep
        elif timestep.ndim == 2 and timestep.shape[1] == expected_tokens:
            patch_values = timestep.view(batch_size, patch_frame, patch_height, patch_width)
        else:
            raise ValueError(
                f"Expected patch timestep shape [B, {expected_tokens}] or "
                f"[B, {patch_frame}, {patch_height}, {patch_width}], got {tuple(timestep.shape)}"
            )
        return (
            patch_values
            .repeat_interleave(patch_t, dim=1)
            .repeat_interleave(patch_h, dim=2)
            .repeat_interleave(patch_w, dim=3)
        )

    def _convert_flow_pred_to_x0_video(
        self,
        flow_pred: torch.Tensor,
        xt: torch.Tensor,
        timestep: torch.Tensor,
        precomputed_sigma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if precomputed_sigma is not None:
            original_dtype = flow_pred.dtype
            if precomputed_sigma.ndim in {2, 4}:
                precomputed_sigma = self._expand_patch_timestep_to_latent(precomputed_sigma, xt)
            while precomputed_sigma.ndim < flow_pred.ndim:
                precomputed_sigma = precomputed_sigma.unsqueeze(2)
            sigma = precomputed_sigma.to(device=flow_pred.device, dtype=torch.float64)
            return (xt.double() - sigma * flow_pred.double()).to(original_dtype)
        if timestep.ndim == 4:
            return self._convert_patch_flow_pred_to_x0_video(flow_pred, xt, timestep)
        if timestep.ndim == 2 and timestep.shape[1] == flow_pred.shape[1]:
            return self._convert_flow_pred_to_x0(
                flow_pred=flow_pred.flatten(0, 1),
                xt=xt.flatten(0, 1),
                timestep=timestep.flatten(0, 1)
            ).unflatten(0, flow_pred.shape[:2])

        if timestep.ndim == 1:
            timestep = timestep[:, None].expand(-1, flow_pred.shape[1])
            return self._convert_flow_pred_to_x0(
                flow_pred=flow_pred.flatten(0, 1),
                xt=xt.flatten(0, 1),
                timestep=timestep.flatten(0, 1)
            ).unflatten(0, flow_pred.shape[:2])

        return self._convert_patch_flow_pred_to_x0_video(flow_pred, xt, timestep)

    def _convert_patch_flow_pred_to_x0_video(
        self,
        flow_pred: torch.Tensor,
        xt: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        original_dtype = flow_pred.dtype
        expanded_patch_timestep = self._expand_patch_timestep_to_latent(timestep, xt)
        flow_pred, xt, expanded_patch_timestep, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [
                flow_pred,
                xt,
                expanded_patch_timestep,
                self.scheduler.sigmas,
                self.scheduler.timesteps,
            ],
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - expanded_patch_timestep.flatten().unsqueeze(1)).abs(),
            dim=1,
        )
        patch_sigma = sigmas[timestep_id].reshape(expanded_patch_timestep.shape)
        while patch_sigma.ndim < flow_pred.ndim:
            patch_sigma = patch_sigma.unsqueeze(2)
        return (xt - patch_sigma * flow_pred).to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        
        classify_mode: Optional[bool] = False, # DF
        concat_time_embeddings: Optional[bool] = False, #DF
        clean_x: Optional[torch.Tensor] = None, # TF
        aug_t: Optional[torch.Tensor] = None, # for TF clean GT, if it's also noisy and needs denoising by the model, aug_t is its timestep
        
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
    ) -> torch.Tensor:
        _, num_frames, _, latent_height, latent_width = noisy_image_or_video.shape
        seq_len = self.backbone_spec.sequence_length(num_frames, latent_height, latent_width)
        self.seq_len = seq_len
        if adaptive_forward_mode is not None:
            if adaptive_forward_mode == "sparse":
                if rope_positions is None or active_token_indices is None:
                    raise ValueError("adaptive sparse forward requires rope_positions and active_token_indices")
                return self._adaptive_sparse_forward_impl(
                    noisy_image_or_video=noisy_image_or_video,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    rope_positions=rope_positions,
                    active_token_indices=active_token_indices,
                    clear_kv_cache=clear_kv_cache,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    cache_end=cache_start,
                    collect_token_indices=collect_token_indices,
                    collect_active_cache=collect_active_cache,
                    full_timestep=full_timestep,
                    return_patch_exit_logits=return_patch_exit_logits,
                    kv_workspace=kv_workspace,
                    cache_only=cache_only,
                )
            raise ValueError(f"Invalid adaptive_forward_mode={adaptive_forward_mode}")

        prompt_embeds = self._prepare_prompt_embeds(
            conditional_dict["prompt_embeds"],
            noisy_image_or_video,
        )
        input_timestep = self._prepare_input_timestep(timestep, noisy_image_or_video)
        model_timestep = self._prepare_model_timestep(input_timestep)

        logits = None
        patch_exit_logits = None
        # X0 prediction
        if kv_cache is not None:
            model_output = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=model_timestep, context=prompt_embeds,
                seq_len=seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                write_cache=write_cache,
                return_patch_exit_logits=return_patch_exit_logits,
            )
            if return_patch_exit_logits:
                flow_pred, patch_exit_logits = model_output
            else:
                flow_pred = model_output
            flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
        else:
            if return_patch_exit_logits:
                raise ValueError("router logits require kv_cache inference")
            if clean_x is not None:
                # teacher forcing
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4), # => [B, C, F, H, W]
                    t=model_timestep, context=prompt_embeds,
                    seq_len=seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4), # => [B, C, F, H, W]
                    aug_t=aug_t,
                ).permute(0, 2, 1, 3, 4)
            else:
                # diffusion forcing or bidirectional
                if classify_mode:
                    flow_pred, logits = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=model_timestep, context=prompt_embeds,
                        seq_len=seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    flow_pred = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=model_timestep, context=prompt_embeds,
                        seq_len=seq_len
                    ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0_video(
            flow_pred=flow_pred,
            xt=noisy_image_or_video,
            timestep=timestep,
            precomputed_sigma=precomputed_sigma,
        )

        if logits is not None:
            return flow_pred, pred_x0, logits
        if patch_exit_logits is not None:
            return flow_pred, pred_x0, patch_exit_logits

        return flow_pred, pred_x0

    def _adaptive_sparse_forward_impl(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        rope_positions: torch.Tensor,
        active_token_indices: torch.Tensor,
        clear_kv_cache: Optional[List[dict]],
        kv_cache: List[dict],
        crossattn_cache: Optional[List[dict]],
        cache_end: int,
        collect_token_indices: Optional[torch.Tensor] = None,
        collect_active_cache: bool = False,
        full_timestep: Optional[torch.Tensor] = None,
        return_patch_exit_logits: bool = False,
        kv_workspace=None,
        cache_only: bool = False,
    ):
        if not hasattr(self.model, "adaptive_sparse_forward"):
            raise ValueError("adaptive_sparse_forward requires causal_model_variant='adaptive'")
        model_output = self.model(
            noisy_image_or_video.permute(0, 2, 1, 3, 4),
            t=self._prepare_model_timestep(self._prepare_input_timestep(timestep, noisy_image_or_video)),
            context=self._prepare_prompt_embeds(
                conditional_dict["prompt_embeds"], noisy_image_or_video
            ),
            seq_len=self.backbone_spec.sequence_length(
                noisy_image_or_video.shape[1],
                noisy_image_or_video.shape[-2],
                noisy_image_or_video.shape[-1],
            ),
            adaptive_forward_mode="sparse",
            rope_positions=rope_positions,
            active_token_indices=active_token_indices,
            clear_kv_cache=clear_kv_cache,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            cache_end=cache_end,
            collect_token_indices=collect_token_indices,
            collect_active_cache=collect_active_cache,
            full_timestep=(
                self._prepare_model_timestep(self._prepare_input_timestep(full_timestep, noisy_image_or_video))
                if full_timestep is not None
                else None
            ),
            return_patch_exit_logits=return_patch_exit_logits,
            kv_workspace=kv_workspace,
            cache_only=cache_only,
        )
        def as_video(output):
            return None if output is None else output.permute(0, 2, 1, 3, 4)

        if collect_active_cache or collect_token_indices is not None:
            if return_patch_exit_logits:
                flow_pred, sparse_cache, patch_exit_logits = model_output
            else:
                flow_pred, sparse_cache = model_output
            flow_pred = as_video(flow_pred)
            if return_patch_exit_logits:
                return flow_pred, sparse_cache, patch_exit_logits
            return flow_pred, sparse_cache
        if return_patch_exit_logits:
            flow_pred, patch_exit_logits = model_output
            flow_pred = as_video(flow_pred)
            return flow_pred, patch_exit_logits
        return as_video(model_output)

    def write_sparse_chunk_cache(
        self,
        kv_cache: List[dict],
        sparse_kv_cache: List[dict],
        chunk_tokens: int,
        cache_start: int,
    ):
        if not hasattr(self.model, "write_sparse_chunk_cache"):
            raise ValueError("write_sparse_chunk_cache requires causal_model_variant='adaptive'")
        self.model.write_sparse_chunk_cache(
            kv_cache=kv_cache,
            sparse_kv_cache=sparse_kv_cache,
            chunk_tokens=chunk_tokens,
            cache_start=cache_start,
        )

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()
