"""LongLive prompt-switch recache; no changes to shared attention classes."""
from contextlib import contextmanager
import math

import torch

from methods.Wan21.shared.backbone.wan.modules.causal_model import CausalWanSelfAttention, causal_rope_apply
from methods.Wan21.shared.backbone.wan.modules.attention import attention


class LongLiveSelfAttention(CausalWanSelfAttention):
    """Only the explicit prompt-recache path differs from standard Wan."""

    def forward(self, x, seq_lens, grid_sizes, freqs, block_mask, kv_cache=None,
                current_start=0, cache_start=None, write_cache=True, grid_shapes=None,
                sparse_forward=None, sparse_forward_kwargs=None):
        if not getattr(self, "longlive_recache", False):
            return super().forward(x, seq_lens, grid_sizes, freqs, block_mask, kv_cache,
                                   current_start, cache_start, write_cache, grid_shapes,
                                   sparse_forward, sparse_forward_kwargs)
        if kv_cache is None or sparse_forward is not None or not write_cache:
            raise ValueError("Recache requires a dense committed forward")
        b, s = x.shape[:2]
        q = self.norm_q(self.q(x)).view(b, s, self.num_heads, self.head_dim)
        k = self.norm_k(self.k(x)).view(b, s, self.num_heads, self.head_dim)
        v = self.v(x).view(b, s, self.num_heads, self.head_dim)
        frame_tokens = int(math.prod(grid_sizes[0][1:]))
        q = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start//frame_tokens).to(v.dtype)
        k = causal_rope_apply(k, grid_sizes, freqs, start_frame=current_start//frame_tokens).to(v.dtype)
        old_global, old_local = int(kv_cache["global_end_index"]), int(kv_cache["local_end_index"])
        current_end = current_start+s
        if current_end > old_global:
            raise ValueError("Prompt recache must only rewrite generated history")
        local_end = old_local+current_end-old_global
        local_start = local_end-s
        if local_start < 0:
            raise ValueError("Recache exceeds retained window")
        recompute = current_end <= old_global and current_start > 0
        sink_tokens = self.sink_size*frame_tokens
        write_start = max(local_start, sink_tokens) if recompute and self.longlive_global_sink else local_start
        offset = write_start-local_start
        kv_cache["k"][:, write_start:local_end] = k[:, offset:]
        kv_cache["v"][:, write_start:local_end] = v[:, offset:]
        if sink_tokens:
            budget = self.max_attention_size-sink_tokens
            pieces = [slice(0, sink_tokens)]
            if budget > 0:
                pieces.append(slice(max(sink_tokens, local_end-budget), local_end))
            keys = torch.cat([kv_cache["k"][:, part] for part in pieces], dim=1)
            values = torch.cat([kv_cache["v"][:, part] for part in pieces], dim=1)
        else:
            start = max(0, local_end-self.max_attention_size)
            keys, values = kv_cache["k"][:, start:local_end], kv_cache["v"][:, start:local_end]
        result = self.o(attention(q, keys, values).flatten(2))
        if not recompute:
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end)
        return result


def install_recache_attention(generator):
    # Instance-local promotion retains all loaded parameter names and values.
    for block in generator.model.blocks:
        if type(block.self_attn) is CausalWanSelfAttention:
            block.self_attn.__class__ = LongLiveSelfAttention
        if not isinstance(block.self_attn, LongLiveSelfAttention):
            raise TypeError("Unexpected attention implementation")


def clear_text_cache(caches):
    for layer in caches:
        layer["k"].zero_()
        layer["v"].zero_()
        layer["is_init"] = False


@contextmanager
def recache_mode(generator, global_sink):
    for block in generator.model.blocks:
        block.self_attn.longlive_recache = True
        block.self_attn.longlive_global_sink = bool(global_sink)
    try:
        yield
    finally:
        for block in generator.model.blocks:
            block.self_attn.longlive_recache = False


@torch.no_grad()
def recache(generator, history, current_frame, conditional, kv_cache, text_cache,
            frame_tokens, *, global_sink=True):
    clear_text_cache(text_cache)
    if current_frame == 0:
        return
    if not global_sink:
        for layer in kv_cache:
            layer["k"].zero_()
            layer["v"].zero_()
    count = min(12, int(current_frame))
    # history contains the most recent retained latent frames, in chronological order.
    history = history[:, -count:].to(next(generator.parameters()).device)
    for layer in kv_cache:
        layer.pop("_router_workspace", None)
    with recache_mode(generator, global_sink):
        generator(noisy_image_or_video=history, conditional_dict=conditional,
                  timestep=torch.zeros((1, count), device=history.device),
                  kv_cache=kv_cache, crossattn_cache=text_cache,
                  current_start=(current_frame-count)*frame_tokens, write_cache=True)
    for layer in kv_cache:
        layer["_host_position"] = (int(layer["global_end_index"]), int(layer["local_end_index"]))
    clear_text_cache(text_cache)
