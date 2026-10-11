"""Sparse video queries with full visible clean/history keys and native PRoPE.

Derived from SolarWM sgf_attention.py at the pinned commit (Apache-2.0).
Packed prefix rows retain their upstream attention visibility exactly.
"""
from dataclasses import dataclass
from typing import Any
import torch


@dataclass(frozen=True)
class SparseControl:
    layout: Any
    viewmats: torch.Tensor
    intrinsics: torch.Tensor
    mode: str = "rollout"
    chunk_index: int = 0
    cache: Any = None
    commit_cache: bool = False
    prefix_rotary: Any = None
    window_rotaries: Any = None
    compute_ids: Any = None
    refresh_ids: Any = None
    workspace: Any = None


def sparse_attention(q, k, v, *, control, layer, apply_rotary, attend, prope=None):
    if prope is None:
        from solarwm.backends.minimax_h3.torch_prope import prope_qkv_separate
        prope = prope_qkv_separate
    c = control
    rows = int(c.layout.rows_per_video_frame)
    condition = int(c.layout.condition_indices.numel())
    prefix = condition + int(c.layout.audio_indices.numel())
    ids = c.compute_ids
    batch = q.shape[0]
    views = torch.eye(4, device=q.device, dtype=torch.float32).repeat(batch, prefix, 1, 1)
    ks = torch.eye(3, device=q.device, dtype=torch.float32).repeat(batch, prefix, 1, 1)
    views[:, c.layout.condition_video_indices] = c.viewmats[:, :1]
    ks[:, c.layout.condition_video_indices] = c.intrinsics[:, :1]

    def slice_rope(rotary, start, end):
        return tuple(x[start:end] for x in rotary)

    def calculate(query, key, value, qr, kr, qcam, kcam):
        query, key = apply_rotary(query, *qr), apply_rotary(key, *kr)
        query, key, value, basis = prope(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            query_viewmats=qcam[0], query_K=qcam[1], key_viewmats=kcam[0], key_K=kcam[1])
        out = attend(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
        return basis(out.transpose(1, 2)).transpose(1, 2)

    out = [calculate(q[:, :condition], k[:, :condition], v[:, :condition],
                     slice_rope(c.prefix_rotary, 0, condition),
                     slice_rope(c.prefix_rotary, 0, condition),
                     (views[:, :condition], ks[:, :condition]),
                     (views[:, :condition], ks[:, :condition]))]
    if prefix > condition:
        out.append(calculate(q[:, condition:prefix], k[:, :prefix], v[:, :prefix],
                             slice_rope(c.prefix_rotary, condition, prefix), c.prefix_rotary,
                             (views[:, condition:prefix], ks[:, condition:prefix]), (views, ks)))
    history = c.cache.history(layer, c.chunk_index)
    history_k = torch.cat([x[1] for x in history], 1) if history else k[:, :0]
    history_v = torch.cat([x[2] for x in history], 1) if history else v[:, :0]
    current_k, current_v = c.workspace.assemble(
        layer, k[:, prefix:], v[:, prefix:], ids, c.refresh_ids, 5 * rows)
    first = max(0, c.chunk_index - 5)
    local_frames = (c.chunk_index - first + 1) * 5
    rotary = c.window_rotaries[local_frames]
    qr = tuple(x[(local_frames - 5) * rows:].index_select(0, ids) for x in rotary)
    kr = tuple(torch.cat((p, w), 0) for p, w in zip(c.prefix_rotary, rotary))
    window_views = c.viewmats[:, 1 + first * 5:1 + (c.chunk_index + 1) * 5].repeat_interleave(rows, 1)
    window_ks = c.intrinsics[:, 1 + first * 5:1 + (c.chunk_index + 1) * 5].repeat_interleave(rows, 1)
    out.append(calculate(q[:, prefix:], torch.cat((k[:, :prefix], history_k, current_k), 1),
                         torch.cat((v[:, :prefix], history_v, current_v), 1), qr, kr,
                         (window_views[:, -5 * rows:].index_select(1, ids),
                          window_ks[:, -5 * rows:].index_select(1, ids)),
                         (torch.cat((views, window_views), 1), torch.cat((ks, window_ks), 1))))
    if c.commit_cache:
        c.cache.commit(layer, c.chunk_index, current_k, current_v)
    return torch.cat(out, 1)


def install_processors(model):
    """Replace upstream hard-coded dispatch without renaming any model parameters."""
    from solarwm.backends.minimax_h3 import model as native

    class SparseDispatch(native.SolarMiniMaxH3Transformer3DModel):
        @staticmethod
        def _sgf_attention_forward(attn, query, key, value, control):
            if not isinstance(control, SparseControl):
                return native.SolarMiniMaxH3Transformer3DModel._sgf_attention_forward(
                    attn, query, key, value, control)
            sp = native.is_sequence_parallel_enabled()
            exchange = native.sequence_model_parallel_all_to_all_4D
            if sp:
                query, key, value = [exchange(x, scatter_dim=2, gather_dim=1)
                                     for x in (query, key, value)]
            def attend(q, k, v):
                from flash_attn import flash_attn_func
                return flash_attn_func(q.contiguous(), k.contiguous(), v.contiguous(),
                                       dropout_p=0.0, causal=False)
            result = sparse_attention(query, key, value, control=control,
                                      layer=attn.h3_layer_index,
                                      apply_rotary=native._apply_rotary_emb, attend=attend)
            if sp:
                result = exchange(result, scatter_dim=1, gather_dim=2)
            return attn.to_out[1](attn.to_out[0](result.flatten(2, 3).type_as(query)))

    class Processor(native.SolarMiniMaxH3AttnProcessor):
        def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
            if not isinstance(attention_mask, native.H3AttentionControl):
                return super().__call__(attn, hidden_states, rotary_emb, attention_mask)
            c = attention_mask
            if c.sequence_lengths:
                native.get_sequence_parallel_state().sequence_lengths = c.sequence_lengths
            return SparseDispatch._attention_forward(
                attn, hidden_states, rotary_emb, c.attention_mask,
                fused_prope=c.fused_prope, prope_token_indices=c.prope_token_indices,
                prope_frame_ids=c.prope_frame_ids, cam_viewmats=c.cam_viewmats,
                cam_K=c.cam_K, prope_kwargs=c.prope_kwargs, sgf_attention=c.sgf_attention)

    for i, block in enumerate(model.transformer_blocks):
        block.attn.set_processor(Processor())
        block.attn.h3_layer_index = i

