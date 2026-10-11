"""Adaptive causal LingBot-World v2 backbone.

Architecture and upstream parameter names follow Robbyant's LingBot-World v2
``wan/modules/model_fast.py`` at commit
``2648877f763a06cc743bcd919936da4d25f12e7b`` (CC BY-NC-SA 4.0).  Runtime
cache, sparse-token, and patch-router mechanics reuse this repository's Wan
components.
"""

from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as torch_F
from diffusers.configuration_utils import register_to_config

from methods.Wan21.shared.backbone.wan.distributed.ulysses import (
    gather_forward,
    shard_sequence,
)
from methods.Wan21.shared.backbone.wan.modules.causal_adaptive_model import (
    AdaptiveCausalWanModel,
)
from methods.Wan21.shared.backbone.wan.modules.causal_model import (
    CausalWanAttentionBlock,
    _module_param_device,
    _module_param_dtype,
)
from methods.Wan21.shared.backbone.wan.modules.causal_sparse_utils import ChunkKVWorkspace

from .camera import pack_camera_rays


def _official_fp32_autocast(reference: torch.Tensor):
    """Match the released Fast model's FP32 autocast islands on CUDA."""

    if reference.device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.float32)
    return nullcontext()


class LingBotV2CausalAttentionBlock(CausalWanAttentionBlock):
    """Wan causal block with LingBot camera modulation between its attentions."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
    ) -> None:
        # LingBot's I2V condition is concatenated at the DiT input.  Its
        # cross-attention remains the plain text-only Wan cross-attention.
        super().__init__(
            "t2v_cross_attn",
            dim,
            ffn_dim,
            num_heads,
            local_attn_size,
            sink_size,
            qk_norm,
            cross_attn_norm,
            eps,
        )
        # Keep these names byte-for-byte compatible with the official state
        # dict.  Four Linear modules contribute eight keys per block.
        self.cam_injector_layer1 = nn.Linear(dim, dim)
        self.cam_injector_layer2 = nn.Linear(dim, dim)
        self.cam_scale_layer = nn.Linear(dim, dim)
        self.cam_shift_layer = nn.Linear(dim, dim)

    def _inject_camera(
        self,
        x: torch.Tensor,
        camera_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if camera_tokens is None:
            return x
        if camera_tokens.shape != x.shape:
            raise ValueError(
                "encoded camera tokens must align with visual tokens: "
                f"camera={tuple(camera_tokens.shape)} visual={tuple(x.shape)}"
            )
        camera_hidden = self.cam_injector_layer2(
            torch_F.silu(self.cam_injector_layer1(camera_tokens))
        )
        camera_hidden = camera_hidden + camera_tokens
        return (1.0 + self.cam_scale_layer(camera_hidden)) * x + self.cam_shift_layer(
            camera_hidden
        )

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        dit_cond_dict=None,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        write_cache=True,
        grid_shapes=None,
        sparse_forward=None,
        sparse_forward_kwargs=None,
    ):
        # This dispatch must remain inside forward: block-level FSDP only
        # materializes sharded Linear weights while this boundary is active.
        if sparse_forward is not None:
            return sparse_forward(
                block=self,
                x=x,
                e=e,
                context=context,
                context_lens=context_lens,
                **(sparse_forward_kwargs or {}),
            )

        time_axis = e.shape[1]
        per_token = time_axis == x.shape[1]
        if not per_token:
            num_frames = time_axis
            if x.shape[1] % num_frames:
                raise ValueError(
                    f"token count {x.shape[1]} is not divisible by timestep count {num_frames}"
                )
            frame_seqlen = x.shape[1] // num_frames
        # The released Fast block deliberately keeps timestep modulation and
        # residual streams in FP32.  Linear/attention operators still execute
        # under the wrapper's outer BF16 autocast region.
        with _official_fp32_autocast(x):
            e = (self.modulation.unsqueeze(1) + e.float()).chunk(6, dim=2)
        if per_token:
            e = tuple(value.squeeze(2) for value in e)
            attn_input = self.norm1(x).float() * (1 + e[1]) + e[0]
        else:
            attn_input = (
                self.norm1(x)
                .float()
                .unflatten(1, (num_frames, frame_seqlen))
                * (1 + e[1])
                + e[0]
            ).flatten(1, 2)

        y = self.self_attn(
            attn_input,
            seq_lens,
            grid_sizes,
            freqs,
            block_mask,
            kv_cache,
            current_start,
            cache_start,
            write_cache,
            grid_shapes,
        )
        with _official_fp32_autocast(x):
            if per_token:
                x = x.float() + y * e[2]
            else:
                x = x.float() + (
                    y.unflatten(1, (num_frames, frame_seqlen)) * e[2]
                ).flatten(1, 2)

        camera_tokens = None
        if dit_cond_dict is not None:
            camera_tokens = dit_cond_dict.get("c2ws_plucker_emb")
        x = self._inject_camera(x, camera_tokens)

        x = x + self.cross_attn(
            self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache
        )
        if per_token:
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            with _official_fp32_autocast(x):
                return x.float() + y * e[5]
        y = self.ffn(
            (
                self.norm2(x)
                .float()
                .unflatten(1, (num_frames, frame_seqlen))
                * (1 + e[4])
                + e[3]
            ).flatten(1, 2)
        )
        with _official_fp32_autocast(x):
            return x.float() + (
                y.unflatten(1, (num_frames, frame_seqlen)) * e[5]
            ).flatten(1, 2)


class LingBotV2FastAdaptiveModel(AdaptiveCausalWanModel):
    """Official-key-compatible LingBot v2 Fast backbone plus patch routing."""

    _no_split_modules = ["LingBotV2CausalAttentionBlock"]

    @classmethod
    def load_config(cls, pretrained_model_name_or_path, **kwargs):
        """Read the snapshot-root config for HF's split LingBot layout.

        The fixed release stores ``config.json`` at the snapshot root while
        its index and shards live in ``transformers/``.  Diffusers normally
        searches for both in the subfolder, so redirect only config lookup to
        the root and leave weight resolution untouched.
        """

        subfolder = kwargs.get("subfolder")
        if subfolder:
            root = Path(pretrained_model_name_or_path).expanduser()
            if (
                root.is_dir()
                and not (root / str(subfolder) / cls.config_name).is_file()
                and (root / cls.config_name).is_file()
            ):
                kwargs = dict(kwargs)
                kwargs["subfolder"] = None
        return super().load_config(pretrained_model_name_or_path, **kwargs)

    @register_to_config
    def __init__(
        self,
        model_type: str = "i2v",
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 36,
        dim: int = 5120,
        ffn_dim: int = 13824,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 40,
        num_layers: int = 40,
        local_attn_size: int = 18,
        sink_size: int = 6,
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        control_dim: int = 6,
        camera_spatial_stride: tuple[int, int] = (8, 8),
    ) -> None:
        if model_type != "i2v":
            raise ValueError(f"LingBot v2 Fast only supports model_type='i2v', got {model_type!r}")
        if in_dim != 36:
            raise ValueError(
                "LingBot v2 Fast expects 36 input channels (16 noisy + 20 I2V condition), "
                f"got {in_dim}"
            )
        if control_dim != 6:
            raise ValueError(f"LingBot camera rays require six channels, got {control_dim}")
        if tuple(patch_size)[0] != 1:
            raise ValueError("causal LingBot inference requires temporal patch_size == 1")

        # Construct the reusable Wan pieces without I2V CLIP modules or the
        # WanI2VCrossAttention k_img/v_img keys; neither exists upstream here.
        super().__init__(
            model_type="t2v",
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
        )
        self.model_type = "i2v"
        self.sink_size = int(sink_size)
        self.control_dim = int(control_dim)
        self.camera_spatial_stride = tuple(map(int, camera_spatial_stride))

        self.blocks = nn.ModuleList(
            [
                LingBotV2CausalAttentionBlock(
                    dim,
                    ffn_dim,
                    num_heads,
                    local_attn_size,
                    sink_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )
        camera_patch_dim = (
            control_dim
            * math.prod(self.camera_spatial_stride)
            * math.prod(tuple(map(int, patch_size)))
        )
        # These three modules provide the six non-block camera keys in the
        # official checkpoint.
        self.patch_embedding_wancamctrl = nn.Linear(camera_patch_dim, dim)
        self.c2ws_hidden_states_layer1 = nn.Linear(dim, dim)
        self.c2ws_hidden_states_layer2 = nn.Linear(dim, dim)

        # Reinitialize the replaced blocks and newly added camera modules using
        # Wan's initializer.  Official weights overwrite these values on load.
        self.init_weights()

    def _time_tokens(self, t: torch.Tensor, dtype: torch.dtype):
        """Reproduce upstream FP32 time embeddings irrespective of DiT dtype."""

        from methods.Wan21.shared.backbone.wan.modules.model import sinusoidal_embedding_1d

        with _official_fp32_autocast(t):
            embedding = sinusoidal_embedding_1d(self.freq_dim, t.flatten()).float()
            e = self.time_embedding(embedding)
            e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(
                dim=0, sizes=t.shape
            )
        return e0.float(), e.unflatten(dim=0, sizes=t.shape).unsqueeze(2).float()

    def _official_head(self, tokens: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Run the released Fast head's FP32 modulation and projection."""

        with _official_fp32_autocast(tokens):
            modulation = (
                self.head.modulation.unsqueeze(1) + e.float()
            ).chunk(2, dim=2)
            time_axis = e.shape[1]
            per_token = time_axis == tokens.shape[1]
            if per_token:
                modulation = tuple(value.squeeze(2) for value in modulation)
                hidden = self.head.norm(tokens).float() * (1 + modulation[1]) + modulation[0]
            else:
                num_frames = time_axis
                if tokens.shape[1] % num_frames:
                    raise ValueError(
                        f"token count {tokens.shape[1]} is not divisible by timestep count {num_frames}"
                    )
                frame_seqlen = tokens.shape[1] // num_frames
                hidden = (
                    self.head.norm(tokens)
                    .float()
                    .unflatten(1, (num_frames, frame_seqlen))
                    * (1 + modulation[1])
                    + modulation[0]
                ).flatten(1, 2)
            return self.head.head(hidden).float()

    @staticmethod
    def _as_sample_list(value, *, name: str) -> list[torch.Tensor]:
        if isinstance(value, torch.Tensor):
            if value.ndim == 4:
                return [value]
            if value.ndim == 5:
                return list(value.unbind(0))
            raise ValueError(f"{name} must be [C,F,H,W] or [B,C,F,H,W], got {tuple(value.shape)}")
        if isinstance(value, (list, tuple)) and value:
            samples = []
            for sample in value:
                if not isinstance(sample, torch.Tensor):
                    raise TypeError(f"{name} entries must be tensors")
                if sample.ndim == 5 and sample.shape[0] == 1:
                    sample = sample.squeeze(0)
                if sample.ndim != 4:
                    raise ValueError(f"{name} entries must be [C,F,H,W], got {tuple(sample.shape)}")
                samples.append(sample)
            return samples
        raise TypeError(f"{name} must be a tensor or non-empty tensor sequence")

    def _join_i2v_inputs(self, x, y) -> list[torch.Tensor]:
        noisy = self._as_sample_list(x, name="x")
        condition = self._as_sample_list(y, name="y")
        if len(noisy) != len(condition):
            raise ValueError(f"x/y batch mismatch: {len(noisy)} versus {len(condition)}")
        joined = []
        for noisy_sample, condition_sample in zip(noisy, condition):
            if noisy_sample.shape[0] != 16 or condition_sample.shape[0] != 20:
                raise ValueError(
                    "LingBot I2V expects 16 noisy channels and 20 condition channels, "
                    f"got {noisy_sample.shape[0]} and {condition_sample.shape[0]}"
                )
            if noisy_sample.shape[1:] != condition_sample.shape[1:]:
                raise ValueError(
                    "x/y latent geometry mismatch: "
                    f"{tuple(noisy_sample.shape[1:])} versus {tuple(condition_sample.shape[1:])}"
                )
            joined.append(torch.cat((noisy_sample, condition_sample), dim=0))
        return joined

    def _video_tokens(self, x):
        """Patchify a variable-list batch without relying on ``list.device``."""

        samples = self._as_sample_list(x, name="video")
        patch_device = _module_param_device(self.patch_embedding)
        patch_dtype = _module_param_dtype(self.patch_embedding)
        if self.freqs.device != patch_device:
            self.freqs = self.freqs.to(patch_device)
        embedded = [
            self.patch_embedding(
                sample.to(device=patch_device, dtype=patch_dtype).unsqueeze(0)
            )
            for sample in samples
        ]
        grid_shapes = [tuple(int(value) for value in sample.shape[2:]) for sample in embedded]
        grid_sizes = torch.tensor(grid_shapes, dtype=torch.long, device=patch_device)
        tokens = torch.cat([sample.flatten(2).transpose(1, 2) for sample in embedded], dim=0)
        return tokens, grid_sizes, grid_shapes

    @staticmethod
    def _camera_value(dit_cond_dict: Optional[dict]):
        if dit_cond_dict is None:
            return None
        if "camera_rays" in dit_cond_dict:
            return dit_cond_dict["camera_rays"]
        return dit_cond_dict.get("c2ws_plucker_emb")

    def _camera_tensor(self, value, *, batch_size: int) -> Optional[torch.Tensor]:
        if value is None:
            return None
        samples = self._as_sample_list(value, name="camera condition")
        if len(samples) != batch_size:
            raise ValueError(
                f"camera batch size {len(samples)} does not match visual batch size {batch_size}"
            )
        camera = torch.stack(samples, dim=0)
        if camera.shape[1] == self.control_dim:
            camera = pack_camera_rays(
                camera,
                spatial_stride=self.camera_spatial_stride,
            )
        expected_channels = self.control_dim * math.prod(self.camera_spatial_stride)
        if camera.shape[1] != expected_channels:
            raise ValueError(
                "camera condition must be raw six-channel rays or upstream-packed "
                f"{expected_channels}-channel rays, got {camera.shape[1]} channels"
            )
        return camera

    def _encode_camera_tokens(
        self,
        dit_cond_dict: Optional[dict],
        *,
        batch_size: int,
        grid_shapes: list[tuple[int, int, int]],
    ) -> Optional[torch.Tensor]:
        camera = self._camera_tensor(self._camera_value(dit_cond_dict), batch_size=batch_size)
        if camera is None:
            return None
        expected_input_grid = tuple(
            grid_shapes[0][axis] * self.patch_size[axis] for axis in range(3)
        )
        if any(shape != grid_shapes[0] for shape in grid_shapes):
            raise ValueError("camera-conditioned LingBot batches require equal latent geometry")
        if tuple(camera.shape[2:]) != expected_input_grid:
            raise ValueError(
                "packed camera geometry must match the pre-patch latent grid: "
                f"camera={tuple(camera.shape[2:])} expected={expected_input_grid}"
            )
        patch_t, patch_h, patch_w = self.patch_size
        batch, channels, frames, height, width = camera.shape
        camera = (
            camera.view(
                batch,
                channels,
                frames // patch_t,
                patch_t,
                height // patch_h,
                patch_h,
                width // patch_w,
                patch_w,
            )
            .permute(0, 2, 4, 6, 1, 3, 5, 7)
            .reshape(
                batch,
                (frames // patch_t) * (height // patch_h) * (width // patch_w),
                channels * patch_t * patch_h * patch_w,
            )
        )
        camera = camera.to(
            device=_module_param_device(self.patch_embedding_wancamctrl),
            dtype=_module_param_dtype(self.patch_embedding_wancamctrl),
        )
        camera = self.patch_embedding_wancamctrl(camera)
        camera_hidden = self.c2ws_hidden_states_layer2(
            torch_F.silu(self.c2ws_hidden_states_layer1(camera))
        )
        return camera + camera_hidden

    @staticmethod
    def _prepare_timesteps(t: torch.Tensor, *, batch_size: int, token_count: int) -> torch.Tensor:
        if t.ndim == 0:
            t = t.view(1)
        if t.shape[0] != batch_size:
            if t.numel() == 1:
                t = t.reshape(1).expand(batch_size)
            else:
                raise ValueError(f"timestep batch mismatch: expected {batch_size}, got {tuple(t.shape)}")
        if t.ndim == 1:
            t = t[:, None].expand(batch_size, token_count)
        return t

    def _ephemeral_kv_cache(self, tokens: torch.Tensor) -> list[dict]:
        batch, token_count = tokens.shape[:2]
        shape = (batch, token_count, self.num_heads, self.dim // self.num_heads)
        return [
            {
                "k": tokens.new_zeros(shape),
                "v": tokens.new_zeros(shape),
                "global_end_index": torch.zeros(1, dtype=torch.long, device=tokens.device),
                "local_end_index": torch.zeros(1, dtype=torch.long, device=tokens.device),
            }
            for _ in self.blocks
        ]

    def _isolated_router_grad(self, return_patch_exit_logits: bool) -> bool:
        """Whether this forward should retain autograd only for the router head."""

        head = self.patch_exit_router_head
        return bool(
            return_patch_exit_logits
            and head is not None
            and getattr(head, "force_grad", False)
        )

    def _dense_forward(
        self,
        x,
        t,
        context,
        seq_len,
        *,
        y,
        dit_cond_dict=None,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        write_cache=True,
        max_attention_size=None,
        frame_seqlen=None,
        return_patch_exit_logits=False,
    ):
        isolated_router_grad = self._isolated_router_grad(return_patch_exit_logits)
        joined = self._join_i2v_inputs(x, y)
        tokens, grid_sizes, grid_shapes = self._video_tokens(joined)
        if isolated_router_grad:
            # Keep the outer/root FSDP forward in grad mode so it installs its
            # backward hook, while dropping the patch-embedding graph here.
            tokens = tokens.detach()
        batch_size, token_count = tokens.shape[:2]
        if token_count > int(seq_len):
            raise ValueError(f"sequence length {token_count} exceeds seq_len={seq_len}")
        actual_frame_seqlen = math.prod(grid_shapes[0][1:])
        if frame_seqlen is not None and int(frame_seqlen) != actual_frame_seqlen:
            raise ValueError(
                f"frame_seqlen={frame_seqlen} does not match latent geometry {actual_frame_seqlen}"
            )
        context, context_lens = self._context_tokens(context)
        t = self._prepare_timesteps(t, batch_size=batch_size, token_count=token_count)
        e0, eh = self._time_tokens(t, tokens.dtype)
        camera_tokens = self._encode_camera_tokens(
            dit_cond_dict,
            batch_size=batch_size,
            grid_shapes=grid_shapes,
        )
        if isolated_router_grad:
            context = context.detach()
            e0 = e0.detach()
            eh = eh.detach()
            if camera_tokens is not None:
                camera_tokens = camera_tokens.detach()
        sp_size, sp_rank = self._sequence_parallel_state()
        if sp_size > 1:
            # Ulysses: every rank keeps the full-sequence contracts (grid shapes,
            # context, cursors) and only carries its own slice through the blocks.
            # eh stays whole because the head runs on the gathered sequence.
            if e0.shape[1] != token_count:
                # The router's dense stage collapses per-token timesteps back to
                # one value per frame, which the released model's BF16 numerics
                # depend on.  A rank's shard does not have to land on a frame
                # boundary, so expand the already-computed embedding to one entry
                # per token: the time MLP still ran per frame, and the modulation
                # arithmetic downstream is elementwise-identical either way.
                if token_count % e0.shape[1]:
                    raise ValueError(
                        "Ulysses sequence parallelism needs timesteps that divide "
                        f"the sequence: timestep entries={e0.shape[1]} tokens={token_count}"
                    )
                e0 = e0.repeat_interleave(token_count // e0.shape[1], dim=1)
            tokens = shard_sequence(tokens, sp_size, sp_rank)
            e0 = shard_sequence(e0, sp_size, sp_rank)
            if camera_tokens is not None:
                camera_tokens = shard_sequence(camera_tokens, sp_size, sp_rank)
        encoded_condition = (
            None if camera_tokens is None else {"c2ws_plucker_emb": camera_tokens}
        )

        if kv_cache is None:
            if current_start not in (None, 0):
                raise ValueError("a non-zero current_start requires a persistent KV cache")
            kv_cache = self._ephemeral_kv_cache(tokens)
        if len(kv_cache) != len(self.blocks):
            raise ValueError(f"expected {len(self.blocks)} KV cache layers, got {len(kv_cache)}")
        seq_lens = torch.full(
            (batch_size,), token_count, dtype=torch.long, device=tokens.device
        )
        for block_index, block in enumerate(self.blocks):
            # Runtime cache configuration owns the semantic attention window.
            # An explicit override remains available for standalone parity
            # tests, but the default must not widen an 18-frame window to the
            # historical one-million-token sentinel.
            if max_attention_size is not None:
                block.self_attn.max_attention_size = int(max_attention_size)
            if getattr(block.self_attn, "frame_length", None) is None:
                block.self_attn.frame_length = actual_frame_seqlen
                block.self_attn.block_length = (
                    actual_frame_seqlen * int(getattr(self, "num_frame_per_block", 1))
                )
            tokens = block(
                tokens,
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                context_lens=context_lens,
                block_mask=None,
                dit_cond_dict=encoded_condition,
                kv_cache=kv_cache[block_index],
                crossattn_cache=(
                    crossattn_cache[block_index] if crossattn_cache is not None else None
                ),
                current_start=int(current_start or 0),
                cache_start=cache_start,
                write_cache=write_cache,
                grid_shapes=grid_shapes,
            )
            if isolated_router_grad:
                # A root flat parameter may make a nominally frozen block build
                # a graph. Drop it after each block instead of retaining all 40.
                tokens = tokens.detach()

        if sp_size > 1:
            # Router logits and the flow head run on the whole sequence so every
            # rank reaches the same decisions and the head code stays unchanged.
            tokens = gather_forward(tokens, dim=1, group=self.sp_group)[:, :token_count]

        patch_exit_logits = None
        if return_patch_exit_logits:
            if self.patch_exit_router_head is None:
                raise ValueError("return_patch_exit_logits=True requires patch_exit_router_head")
            patch_exit_logits = self.patch_exit_router_head(tokens, grid_sizes)
        flow_head_context = torch.no_grad() if isolated_router_grad else nullcontext()
        with flow_head_context:
            output = torch.stack(
                self.unpatchify(self._official_head(tokens, eh), grid_sizes)
            ).float()
        if return_patch_exit_logits:
            return output, patch_exit_logits
        return output

    def _run_lingbot_sparse_block(
        self,
        *,
        block,
        x: torch.Tensor,
        e: torch.Tensor,
        context: torch.Tensor,
        context_lens,
        rope_positions: torch.Tensor,
        camera_tokens: Optional[torch.Tensor],
        kv_cache: Optional[dict],
        crossattn_cache: Optional[dict],
        cache_end: Optional[int],
        prefix_cache: Optional[dict] = None,
        collect_indices: Optional[torch.Tensor] = None,
        kv_workspace: Optional[ChunkKVWorkspace] = None,
        block_index: Optional[int] = None,
    ):
        with _official_fp32_autocast(x):
            modulation = (block.modulation.unsqueeze(1) + e.float()).chunk(6, dim=2)
        modulation = tuple(value.squeeze(2) for value in modulation)
        y, selected_cache = self._self_attn_sparse(
            block=block,
            x_attn=block.norm1(x).float() * (1 + modulation[1]) + modulation[0],
            rope_positions=rope_positions,
            kv_cache=kv_cache,
            cache_end=cache_end,
            prefix_cache=prefix_cache,
            collect_indices=collect_indices,
            kv_workspace=kv_workspace,
            block_index=block_index,
        )
        with _official_fp32_autocast(x):
            x = x.float() + y * modulation[2]
        # Camera modulation deliberately remains before cross-attention, even
        # when only a subset of patches is active.
        x = block._inject_camera(x, camera_tokens)
        x = x + block.cross_attn(
            block.norm3(x), context, context_lens, crossattn_cache=crossattn_cache
        )
        y = block.ffn(
            block.norm2(x).float() * (1 + modulation[4]) + modulation[3]
        )
        with _official_fp32_autocast(x):
            return x.float() + y * modulation[5], selected_cache

    def adaptive_sparse_forward(
        self,
        x,
        t,
        context,
        seq_len,
        rope_positions,
        active_token_indices,
        clear_kv_cache,
        kv_cache,
        crossattn_cache,
        cache_end: int,
        *,
        y,
        dit_cond_dict=None,
        collect_token_indices=None,
        collect_active_cache: bool = False,
        return_patch_exit_logits: bool = False,
        full_timestep=None,
        kv_workspace: Optional[ChunkKVWorkspace] = None,
        cache_only: bool = False,
    ):
        isolated_router_grad = self._isolated_router_grad(return_patch_exit_logits)
        joined = self._join_i2v_inputs(x, y)
        tokens, grid_sizes, grid_shapes = self._video_tokens(joined)
        if isolated_router_grad:
            tokens = tokens.detach()
        if tokens.shape[0] != 1:
            raise ValueError("adaptive sparse inference currently supports batch_size == 1")
        full_camera_tokens = self._encode_camera_tokens(
            dit_cond_dict,
            batch_size=1,
            grid_shapes=grid_shapes,
        )
        if isolated_router_grad and full_camera_tokens is not None:
            full_camera_tokens = full_camera_tokens.detach()
        active_token_indices = active_token_indices.to(device=tokens.device, dtype=torch.long)
        if active_token_indices.ndim != 1:
            raise ValueError("active_token_indices must be one-dimensional")
        if active_token_indices.numel() and (
            int(active_token_indices[0]) < 0 or int(active_token_indices[-1]) >= tokens.shape[1]
        ):
            raise ValueError("active_token_indices fall outside the current chunk")
        if active_token_indices.numel() > 1 and bool(
            (active_token_indices[1:] <= active_token_indices[:-1]).any()
        ):
            raise ValueError("active_token_indices must be strictly increasing")
        tokens = tokens.index_select(1, active_token_indices)
        camera_tokens = (
            None
            if full_camera_tokens is None
            else full_camera_tokens.index_select(1, active_token_indices)
        )
        rope_positions = rope_positions.to(device=tokens.device, dtype=torch.long)
        active_rope_positions = rope_positions.index_select(1, active_token_indices)
        if tokens.shape[1] > int(seq_len):
            raise ValueError(f"sequence length {tokens.shape[1]} exceeds seq_len={seq_len}")
        context, context_lens = self._context_tokens(context)
        e0, eh = self._time_tokens_for_active_grid(
            t,
            tokens.dtype,
            active_token_indices=active_token_indices,
            grid_shapes=grid_shapes,
            full_timestep=full_timestep,
        )
        if isolated_router_grad:
            context = context.detach()
            e0 = e0.detach()
            eh = eh.detach()

        collect_positions = None
        cache_indices = None
        if collect_active_cache:
            collect_positions = torch.arange(tokens.shape[1], device=tokens.device)
            cache_indices = active_token_indices
        elif collect_token_indices is not None:
            collect_token_indices = collect_token_indices.to(device=tokens.device, dtype=torch.long)
            collect_positions = torch.searchsorted(active_token_indices, collect_token_indices)
            valid_positions = collect_positions < active_token_indices.numel()
            if not bool(valid_positions.all()) or not torch.equal(
                active_token_indices.index_select(0, collect_positions), collect_token_indices
            ):
                raise ValueError("collect_token_indices must be a subset of active_token_indices")
            cache_indices = collect_token_indices

        if kv_workspace is not None and cache_indices is not None:
            kv_workspace.begin_piece(cache_indices)
        sp_size, sp_rank = self._sequence_parallel_state()
        active_tokens = tokens.shape[1]
        if sp_size > 1:
            # Ulysses: shard only the active tokens carried through the blocks.
            # active_rope_positions stays whole -- it is the attention's reference
            # for the full active set -- and so do the collect/cache indices.
            if e0.shape[1] != active_tokens:
                raise ValueError(
                    "Ulysses sequence parallelism requires per-token timesteps: "
                    f"timestep tokens={e0.shape[1]} active tokens={active_tokens}"
                )
            tokens = shard_sequence(tokens, sp_size, sp_rank)
            e0 = shard_sequence(e0, sp_size, sp_rank)
            if camera_tokens is not None:
                camera_tokens = shard_sequence(camera_tokens, sp_size, sp_rank)
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)
        if clear_kv_cache is None:
            clear_kv_cache = [None] * len(self.blocks)
        collected_cache = []
        for block_index, block in enumerate(self.blocks):
            prefix_cache = (
                kv_workspace.prefix_cache(block_index)
                if kv_workspace is not None
                else clear_kv_cache[block_index]
            )
            tokens, selected_cache = block(
                tokens,
                e=e0,
                seq_lens=None,
                grid_sizes=None,
                freqs=None,
                context=context,
                context_lens=context_lens,
                block_mask=None,
                sparse_forward=self._run_lingbot_sparse_block,
                sparse_forward_kwargs={
                    "rope_positions": active_rope_positions,
                    "camera_tokens": camera_tokens,
                    "kv_cache": kv_cache[block_index],
                    "crossattn_cache": (
                        crossattn_cache[block_index] if crossattn_cache is not None else None
                    ),
                    "cache_end": cache_end,
                    "prefix_cache": prefix_cache,
                    "collect_indices": collect_positions,
                    "kv_workspace": kv_workspace,
                    "block_index": block_index,
                },
            )
            if isolated_router_grad:
                tokens = tokens.detach()
                if selected_cache is not None:
                    selected_cache = {
                        key: value.detach() if isinstance(value, torch.Tensor) else value
                        for key, value in selected_cache.items()
                    }
            if cache_indices is not None and kv_workspace is None:
                collected_cache.append({**selected_cache, "indices": cache_indices})
        if kv_workspace is not None and cache_indices is not None:
            kv_workspace.finish_piece()
        if sp_size > 1:
            tokens = gather_forward(tokens, dim=1, group=self.sp_group)[:, :active_tokens]

        patch_exit_logits = None
        if return_patch_exit_logits:
            if self.patch_exit_router_head is None:
                raise ValueError("return_patch_exit_logits=True requires patch_exit_router_head")
            patch_exit_logits = self.patch_exit_router_head(tokens, grid_sizes)
        output = None
        if not cache_only:
            flow_head_context = torch.no_grad() if isolated_router_grad else nullcontext()
            with flow_head_context:
                output = self._unpatchify_sparse(
                    self._official_head(tokens, eh), active_token_indices, grid_shapes
                ).float()
        if cache_indices is not None:
            sparse_cache = None if kv_workspace is not None else collected_cache
            if return_patch_exit_logits:
                return output, sparse_cache, patch_exit_logits
            return output, sparse_cache
        if return_patch_exit_logits:
            return output, patch_exit_logits
        return output

    def forward(self, *args, adaptive_forward_mode: Optional[str] = None, **kwargs):
        if adaptive_forward_mode is None:
            return self._dense_forward(*args, **kwargs)
        if adaptive_forward_mode == "sparse":
            return self.adaptive_sparse_forward(*args, **kwargs)
        raise ValueError(f"invalid adaptive_forward_mode={adaptive_forward_mode!r}")


def resolve_lingbot_transformer_directory(root: str | Path) -> Path:
    """Resolve either a HF snapshot root or its ``transformers`` subfolder."""

    root = Path(root).expanduser().resolve()
    nested = root / "transformers"
    if nested.is_dir():
        return nested
    return root


def resolve_lingbot_snapshot_root(root: str | Path) -> Path:
    """Resolve a snapshot root from either supported checkpoint path."""

    root = Path(root).expanduser().resolve()
    if root.name == "transformers" and (root.parent / "config.json").is_file():
        return root.parent
    return root


def load_lingbot_v2_fast_model(
    root: str | Path,
    *,
    architecture: Optional[dict] = None,
    transformer_index: str | Path | None = None,
    torch_dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device | str | int] = None,
    local_attn_size: int = 18,
    sink_size: int = 6,
    load_weights: Optional[bool] = None,
) -> LingBotV2FastAdaptiveModel:
    """Construct from configuration and load the selected sharded weights.

    Non-loading ranks keep parameters on meta for FSDP initialization.  The
    unregistered complex RoPE table is always rebuilt on CPU, since FSDP does
    not materialize or broadcast plain tensor attributes.
    """
    snapshot_root = resolve_lingbot_snapshot_root(root)
    transformer_dir = resolve_lingbot_transformer_directory(root)
    if architecture is None:
        architecture = json.loads((snapshot_root / "config.json").read_text())
    constructor_kwargs = {
        key: value for key, value in dict(architecture).items()
        if not key.startswith("_")
    }
    constructor_kwargs.update(local_attn_size=local_attn_size, sink_size=sink_size)
    target_device = None if device is None else torch.device(device)
    if load_weights is None:
        load_weights = target_device is None or target_device.type != "meta"
    with torch.device("meta"):
        model = LingBotV2FastAdaptiveModel(**constructor_kwargs)
    if torch_dtype is not None:
        model.to(dtype=torch_dtype)

    from methods.Wan21.shared.backbone.wan.modules.model import rope_params

    head_dim = int(model.dim) // int(model.num_heads)
    with torch.device("cpu"):
        model.freqs = torch.cat(
            [
                rope_params(1024, head_dim - 4 * (head_dim // 6)),
                rope_params(1024, 2 * (head_dim // 6)),
                rope_params(1024, 2 * (head_dim // 6)),
            ],
            dim=1,
        )
    if load_weights:
        from accelerate.utils import load_checkpoint_in_model

        # Accelerate accepts an explicit index or discovers the index in a
        # directory. It loads one shard at a time without a second full copy.
        checkpoint = transformer_dir
        if transformer_index is not None:
            checkpoint = Path(transformer_index).expanduser()
            if not checkpoint.is_absolute():
                checkpoint = transformer_dir / checkpoint
        load_checkpoint_in_model(
            model,
            checkpoint=str(checkpoint),
            device_map={"": "cpu"},
            dtype=torch_dtype,
            offload_state_dict=True,
        )
    if target_device is not None and target_device.type != "meta":
        model.to(target_device)
    return model


__all__ = [
    "LingBotV2CausalAttentionBlock",
    "LingBotV2FastAdaptiveModel",
    "load_lingbot_v2_fast_model",
    "resolve_lingbot_snapshot_root",
    "resolve_lingbot_transformer_directory",
]
