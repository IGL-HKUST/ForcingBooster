from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


@dataclass(frozen=True)
class WanBackboneSpec:
    """Architecture metadata read from an instantiated Wan backbone."""

    model_name: str
    num_layers: int
    num_heads: int
    head_dim: int
    dim: int
    patch_size: tuple[int, int, int]
    text_len: int

    @classmethod
    def from_model(cls, model, model_name: str | None = None) -> "WanBackboneSpec":
        blocks = getattr(model, "blocks", None)
        if blocks is None:
            raise ValueError("Wan backbone does not expose blocks")
        dim = int(getattr(model, "dim"))
        num_heads = int(getattr(model, "num_heads"))
        if dim % num_heads != 0:
            raise ValueError(f"backbone dim={dim} is not divisible by num_heads={num_heads}")
        patch_size = tuple(int(value) for value in getattr(model, "patch_size"))
        if len(patch_size) != 3:
            raise ValueError(f"expected a 3D Wan patch size, got {patch_size}")
        return cls(
            model_name=model_name or str(getattr(model, "model_name", model.__class__.__name__)),
            num_layers=len(blocks),
            num_heads=num_heads,
            head_dim=dim // num_heads,
            dim=dim,
            patch_size=patch_size,
            text_len=int(getattr(model, "text_len")),
        )

    def tokens_per_frame(self, latent_height: int, latent_width: int) -> int:
        patch_t, patch_h, patch_w = self.patch_size
        if patch_t != 1:
            raise ValueError(
                "causal per-frame KV caching requires temporal patch_size == 1; "
                f"got patch_size={self.patch_size}"
            )
        if latent_height % patch_h or latent_width % patch_w:
            raise ValueError(
                f"latent geometry {(latent_height, latent_width)} is not divisible by "
                f"spatial patch size {(patch_h, patch_w)}"
            )
        return (int(latent_height) // patch_h) * (int(latent_width) // patch_w)

    def sequence_length(self, num_frames: int, latent_height: int, latent_width: int) -> int:
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        return int(num_frames) * self.tokens_per_frame(latent_height, latent_width)

    def summary(self) -> str:
        return (
            f"model={self.model_name} layers={self.num_layers} heads={self.num_heads} "
            f"head_dim={self.head_dim} dim={self.dim} patch_size={self.patch_size} "
            f"text_len={self.text_len}"
        )


def get_wan_backbone_spec(generator) -> WanBackboneSpec:
    spec = getattr(generator, "backbone_spec", None)
    if spec is None:
        spec = WanBackboneSpec.from_model(
            generator.model,
            model_name=getattr(generator, "model_name", None),
        )
        generator.backbone_spec = spec
    return spec


def self_attention_capacity_tokens(
    spec: WanBackboneSpec,
    *,
    num_frames: int,
    latent_height: int,
    latent_width: int,
    local_attn_size: int = -1,
) -> int:
    capacity_frames = int(local_attn_size) if int(local_attn_size) != -1 else int(num_frames)
    if capacity_frames <= 0:
        raise ValueError(f"attention cache capacity must be positive, got {capacity_frames} frames")
    return spec.sequence_length(capacity_frames, latent_height, latent_width)


def configure_causal_attention_capacity(
    model,
    capacity_tokens: int,
    frame_tokens: int | None = None,
) -> None:
    """Keep each causal attention layer's rolling window consistent with its cache."""

    capacity_tokens = int(capacity_tokens)
    for block in model.blocks:
        self_attn = getattr(block, "self_attn", None)
        if self_attn is not None and hasattr(self_attn, "max_attention_size"):
            self_attn.max_attention_size = capacity_tokens
        if self_attn is not None and frame_tokens is not None and hasattr(self_attn, "frame_length"):
            self_attn.frame_length = int(frame_tokens)
            self_attn.block_length = int(frame_tokens) * int(getattr(model, "num_frame_per_block", 1))


def make_self_attention_cache(
    spec: WanBackboneSpec,
    *,
    batch_size: int,
    capacity_tokens: int,
    dtype: torch.dtype,
    device: torch.device | str | int,
    num_heads: int | None = None,
) -> list[dict]:
    """Allocate the causal self-attention cache.

    Under Ulysses sequence parallelism each rank only ever holds its own heads,
    so callers pass ``num_heads=spec.num_heads // sp_size``.  Cross-attention
    keeps every head and therefore keeps using the spec value.
    """

    heads = spec.num_heads if num_heads is None else int(num_heads)
    shape = (int(batch_size), int(capacity_tokens), heads, spec.head_dim)
    return [
        {
            "k": torch.zeros(shape, dtype=dtype, device=device),
            "v": torch.zeros(shape, dtype=dtype, device=device),
            "global_end_index": torch.zeros(1, dtype=torch.long, device=device),
            "local_end_index": torch.zeros(1, dtype=torch.long, device=device),
        }
        for _ in range(spec.num_layers)
    ]


def make_cross_attention_cache(
    spec: WanBackboneSpec,
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device | str | int,
) -> list[dict]:
    shape = (int(batch_size), spec.text_len, spec.num_heads, spec.head_dim)
    return [
        {
            "k": torch.zeros(shape, dtype=dtype, device=device),
            "v": torch.zeros(shape, dtype=dtype, device=device),
            "is_init": False,
        }
        for _ in range(spec.num_layers)
    ]


def reset_self_attention_cache(caches: Iterable[dict]) -> None:
    for cache in caches:
        cache["global_end_index"].zero_()
        cache["local_end_index"].zero_()


def reset_cross_attention_cache(caches: Iterable[dict]) -> None:
    for cache in caches:
        cache["is_init"] = False


def cache_signature(
    spec: WanBackboneSpec,
    *,
    batch_size: int,
    capacity_tokens: int,
    dtype: torch.dtype,
    device: torch.device | str | int,
) -> tuple:
    return (
        spec.num_layers,
        spec.num_heads,
        spec.head_dim,
        int(batch_size),
        int(capacity_tokens),
        dtype,
        str(torch.device(device) if not isinstance(device, int) else torch.device("cuda", device)),
    )


def configure_cache_geometry(
    owner,
    *,
    batch_size: int,
    num_frames: int,
    latent_height: int,
    latent_width: int,
    dtype: torch.dtype,
    device: torch.device | str | int,
    physical_capacity_frames: int | None = None,
    attention_window_frames: int | None = None,
) -> tuple[bool, int]:
    """Derive cache geometry without conflating storage and attention semantics.

    ``physical_capacity_frames`` may reserve non-committing scratch space beyond
    the model's local-attention window.  ``attention_window_frames`` continues
    to bound the tokens visible to attention, so speculative cache writes do not
    silently widen the model's temporal context.
    """

    spec = get_wan_backbone_spec(owner.generator)
    frame_tokens = spec.tokens_per_frame(latent_height, latent_width)
    configured_window_frames = int(getattr(owner, "local_attn_size", -1))
    storage_frames = (
        int(physical_capacity_frames)
        if physical_capacity_frames is not None
        else configured_window_frames
    )
    capacity_tokens = self_attention_capacity_tokens(
        spec,
        num_frames=num_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        local_attn_size=storage_frames,
    )
    visible_frames = (
        int(attention_window_frames)
        if attention_window_frames is not None
        else configured_window_frames
    )
    attention_tokens = self_attention_capacity_tokens(
        spec,
        num_frames=num_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        local_attn_size=visible_frames,
    )
    signature = cache_signature(
        spec,
        batch_size=batch_size,
        capacity_tokens=capacity_tokens,
        dtype=dtype,
        device=device,
    )
    changed = getattr(owner, "_cache_signature", None) != signature
    owner.backbone_spec = spec
    owner.num_transformer_blocks = spec.num_layers
    owner.frame_seq_length = frame_tokens
    owner.kv_cache_size = capacity_tokens
    owner.attention_window_size = attention_tokens
    owner._cache_signature = signature
    configure_causal_attention_capacity(owner.generator.model, attention_tokens, frame_tokens)
    return changed, capacity_tokens


def validate_cache_shape(caches: Sequence[dict], spec: WanBackboneSpec, capacity_tokens: int) -> None:
    if len(caches) != spec.num_layers:
        raise RuntimeError(f"expected {spec.num_layers} KV layers, got {len(caches)}")
    expected_tail = (int(capacity_tokens), spec.num_heads, spec.head_dim)
    for layer, cache in enumerate(caches):
        if tuple(cache["k"].shape[1:]) != expected_tail or tuple(cache["v"].shape[1:]) != expected_tail:
            raise RuntimeError(
                f"KV cache layer {layer} shape mismatch: k={tuple(cache['k'].shape)} "
                f"v={tuple(cache['v'].shape)} expected tail={expected_tail}"
            )
