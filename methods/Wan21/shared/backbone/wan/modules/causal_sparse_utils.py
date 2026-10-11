import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.utils.checkpoint

from methods.Wan21.shared.backbone.wan.distributed.ulysses import all_to_all, pad_to_multiple
from methods.Wan21.shared.backbone.wan.modules.attention import attention
from methods.Wan21.shared.backbone.wan.modules.causal_model import (
    _cache_history_local_end,
    _module_param_device,
    _module_param_dtype,
    _sink_and_recent_history_slices,
)
from methods.Wan21.shared.backbone.wan.modules.model import sinusoidal_embedding_1d


@dataclass
class ChunkKVWorkspace:
    """Reusable inference-only storage for one adaptive chunk's KV outputs."""

    layers: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    prefix: list[tuple[torch.Tensor, torch.Tensor]]
    chunk_tokens: int = 0
    local_start: int = 0
    committed_tokens: int = 0
    piece_indices: Optional[torch.Tensor] = None
    global_end: int = 0
    local_end: int = 0
    direct_write: bool = False

    @classmethod
    def allocate(cls, kv_cache, max_chunk_tokens: int):
        prefix = []
        for cache in kv_cache:
            shape = (cache["k"].shape[0], max_chunk_tokens, *cache["k"].shape[2:])
            prefix.append(
                (cache["k"].new_empty(shape), cache["v"].new_empty(shape))
            )
        return cls(
            layers=[
                (cache["k"], cache["v"], cache["global_end_index"], cache["local_end_index"])
                for cache in kv_cache
            ],
            prefix=prefix,
        )

    @property
    def capacity(self) -> int:
        return int(self.prefix[0][0].shape[1])

    def reset(
        self,
        *,
        chunk_tokens: int,
        local_start: int,
        allow_rollover: bool = False,
        sink_tokens: int = 0,
    ):
        local_end = int(local_start) + int(chunk_tokens)
        capacity = int(self.layers[0][0].shape[1])
        if local_end > capacity:
            if not allow_rollover:
                raise RuntimeError("adaptive sparse direct cache write exceeds allocated KV cache size")
            evicted_tokens = local_end - capacity
            rolled_tokens = int(local_start) - evicted_tokens - int(sink_tokens)
            if rolled_tokens < 0:
                raise RuntimeError("adaptive sparse rollover would evict sink tokens")
            source_start = int(sink_tokens) + evicted_tokens
            source = slice(source_start, source_start + rolled_tokens)
            destination = slice(int(sink_tokens), int(sink_tokens) + rolled_tokens)
            for cache_k, cache_v, _, _ in self.layers:
                for tensor in (cache_k, cache_v):
                    tensor[:, destination].copy_(tensor[:, source].clone())
            local_start = int(local_start) - evicted_tokens
        self.chunk_tokens = int(chunk_tokens)
        self.local_start = int(local_start)
        self.committed_tokens = 0
        self.piece_indices = None
        return int(local_start)

    def begin_piece(self, cache_indices: torch.Tensor):
        piece_tokens = int(cache_indices.numel())
        if self.committed_tokens + piece_tokens > self.chunk_tokens:
            raise RuntimeError("chunk KV workspace collected more tokens than the chunk contains")
        self.piece_indices = cache_indices

    def prefix_cache(self, block_index: int):
        if self.committed_tokens == 0:
            return None
        prefix_k, prefix_v = self.prefix[block_index]
        return {
            "k": prefix_k[:, :self.committed_tokens],
            "v": prefix_v[:, :self.committed_tokens],
        }

    def collect_block(
        self,
        block_index: int,
        roped_k: torch.Tensor,
        value: torch.Tensor,
        collect_positions: torch.Tensor,
    ):
        piece = slice(self.committed_tokens, self.committed_tokens + self.piece_indices.numel())
        chunk = slice(self.local_start, self.local_start + self.chunk_tokens)
        prefix_k, prefix_v = (buffer[:, piece] for buffer in self.prefix[block_index])
        cache_k, cache_v, _, _ = self.layers[block_index]
        for source, prefix, cache in ((roped_k, prefix_k, cache_k), (value, prefix_v, cache_v)):
            torch.index_select(source, 1, collect_positions, out=prefix)
            cache[:, chunk].index_copy_(1, self.piece_indices, prefix)

    def finish_piece(self):
        self.committed_tokens += int(self.piece_indices.numel())
        self.piece_indices = None

    def commit(self, *, global_end: int, local_end: int):
        if self.committed_tokens != self.chunk_tokens:
            raise RuntimeError(
                f"chunk KV workspace expected {self.chunk_tokens} tokens, collected {self.committed_tokens}"
            )
        for _, _, global_cursor, local_cursor in self.layers:
            global_cursor.fill_(int(global_end))
            local_cursor.fill_(int(local_end))


def rope_apply_positions(x: torch.Tensor, rope_positions: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply Wan 3D RoPE with explicit per-token patch-grid positions."""
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    rope_positions = rope_positions.to(device=x.device, dtype=torch.long)

    out = []
    for i in range(x.shape[0]):
        pos = rope_positions[i]
        x_i = torch.view_as_complex(x[i].to(torch.float64).reshape(x.shape[1], n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][pos[:, 0]],
                freqs[1][pos[:, 1]],
                freqs[2][pos[:, 2]],
            ],
            dim=-1,
        ).reshape(x.shape[1], 1, -1)
        out.append(torch.view_as_real(x_i * freqs_i).flatten(2))
    return torch.stack(out).type_as(x)


class CausalSparseTokenMixin:
    """Shared sparse-token mechanics for causal Wan variants."""

    def _video_tokens(self, x: torch.Tensor):
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        patch_dtype = _module_param_dtype(self.patch_embedding)
        patch_device = _module_param_device(self.patch_embedding)
        x_list = [self.patch_embedding(u.to(device=patch_device, dtype=patch_dtype).unsqueeze(0)) for u in x]
        grid_shapes = [tuple(int(value) for value in u.shape[2:]) for u in x_list]
        grid_sizes = torch.tensor(grid_shapes, dtype=torch.long, device=x.device)
        tokens = [u.flatten(2).transpose(1, 2) for u in x_list]
        tokens = torch.cat(tokens)
        return tokens, grid_sizes, grid_shapes

    def _context_tokens(self, context, clip_fea=None):
        context_lens = None
        text_dtype = _module_param_dtype(self.text_embedding)
        context = self.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]).to(device=_module_param_device(self.text_embedding), dtype=text_dtype)
        )
        if clip_fea is not None:
            img_dtype = _module_param_dtype(self.img_emb)
            context_clip = self.img_emb(clip_fea.to(device=context.device, dtype=img_dtype))
            context = torch.concat([context_clip, context], dim=1)
        return context, context_lens

    def _time_tokens(self, t: torch.Tensor, dtype: torch.dtype):
        time_dtype = _module_param_dtype(self.time_embedding)
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(
                device=_module_param_device(self.time_embedding),
                dtype=time_dtype,
            )
        )
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        eh = e.unflatten(dim=0, sizes=t.shape).unsqueeze(2)
        return e0, eh

    def _time_tokens_for_active_grid(
        self,
        t: torch.Tensor,
        dtype: torch.dtype,
        active_token_indices: torch.Tensor,
        grid_shapes,
        full_timestep: Optional[torch.Tensor] = None,
    ):
        patch_frames = int(grid_shapes[0][0])
        frame_tokens = int(grid_shapes[0][1] * grid_shapes[0][2])
        full_tokens = patch_frames * frame_tokens
        active_token_indices = active_token_indices.to(device=t.device, dtype=torch.long)

        if full_timestep is None:
            is_full_active = (
                t.ndim == 2
                and t.shape[0] == 1
                and t.shape[1] == full_tokens
                and active_token_indices.numel() == full_tokens
                and torch.equal(
                    active_token_indices,
                    torch.arange(full_tokens, device=active_token_indices.device, dtype=torch.long),
                )
            )
            if not is_full_active:
                return self._time_tokens(t, dtype)
            full_timestep = t

        full_timestep = full_timestep.to(device=t.device, dtype=t.dtype)
        if full_timestep.ndim == 4:
            full_timestep = full_timestep.reshape(full_timestep.shape[0], -1)

        frame_timestep = None
        full_token_timestep = None
        if full_timestep.ndim == 2 and full_timestep.shape == (1, patch_frames):
            frame_timestep = full_timestep
        elif full_timestep.ndim == 2 and full_timestep.shape == (1, full_tokens):
            full_token_timestep = full_timestep
            grid_timestep = full_timestep.reshape(1, patch_frames, frame_tokens)
            candidate = grid_timestep[:, :, :1]
            # This is per-frame uniformity, not whole-chunk uniformity.  Keep the
            # data-dependent check: selecting the wrong time-embedding batching
            # path changes BF16 cache values even when timestep values match.
            if bool(torch.equal(grid_timestep, candidate.expand_as(grid_timestep))):
                frame_timestep = candidate.squeeze(-1)
        else:
            return self._time_tokens(t, dtype)

        if frame_timestep is not None:
            e0_frame, eh_frame = self._time_tokens(frame_timestep, dtype)
            e0 = e0_frame.repeat_interleave(frame_tokens, dim=1).index_select(1, active_token_indices)
            eh = eh_frame.repeat_interleave(frame_tokens, dim=1).index_select(1, active_token_indices)
            return e0, eh

        expected_active_timestep = full_token_timestep.index_select(1, active_token_indices)
        if expected_active_timestep.shape != t.shape or not torch.equal(expected_active_timestep, t):
            return self._time_tokens(t, dtype)

        unique_timestep, inverse_indices = torch.unique(
            full_token_timestep.flatten(),
            sorted=True,
            return_inverse=True,
        )
        e0_unique, eh_unique = self._time_tokens(unique_timestep.reshape(1, -1), dtype)
        e0_full = e0_unique.reshape(-1, 6, self.dim).index_select(0, inverse_indices)
        eh_full = eh_unique.reshape(-1, 1, self.dim).index_select(0, inverse_indices)
        e0_full = e0_full.reshape(1, full_tokens, 6, self.dim)
        eh_full = eh_full.reshape(1, full_tokens, 1, self.dim)
        return (
            e0_full.index_select(1, active_token_indices),
            eh_full.index_select(1, active_token_indices),
        )

    def _history_kv(
        self,
        kv_cache: Optional[dict],
        cache_end: Optional[int] = None,
        *,
        attn,
        reserved_tokens: int,
    ):
        if kv_cache is None:
            return [], []
        host_position = kv_cache.get("_host_position")
        local_end = int(
            host_position[1]
            if host_position is not None
            else kv_cache["local_end_index"].item()
        )
        if local_end <= 0:
            return [], []

        frame_length = int(getattr(attn, "frame_length", 0) or 0)
        sink_size = int(getattr(attn, "sink_size", 0))
        if sink_size and frame_length <= 0:
            raise RuntimeError(
                "sparse attention with sink tokens requires configured frame_length"
            )
        sink_tokens = sink_size * frame_length
        history_end = _cache_history_local_end(
            kv_cache,
            cache_end=cache_end,
            sink_tokens=sink_tokens,
        )
        max_attention_size = getattr(attn, "max_attention_size", None)
        if max_attention_size is None:
            max_attention_size = int(kv_cache["k"].shape[1])
        history_slices = _sink_and_recent_history_slices(
            history_end=history_end,
            max_attention_size=int(max_attention_size),
            reserved_tokens=int(reserved_tokens),
            sink_tokens=sink_tokens,
        )
        return (
            [kv_cache["k"][:, piece] for piece in history_slices],
            [kv_cache["v"][:, piece] for piece in history_slices],
        )

    @staticmethod
    def _cat_existing(items):
        items = [x for x in items if x is not None and x.shape[1] > 0]
        if not items:
            return None
        return torch.cat(items, dim=1)

    def _self_attn_sparse(
        self,
        block,
        x_attn: torch.Tensor,
        rope_positions: torch.Tensor,
        kv_cache: Optional[dict],
        cache_end: Optional[int],
        prefix_cache: Optional[dict] = None,
        collect_indices: Optional[torch.Tensor] = None,
        kv_workspace: Optional[ChunkKVWorkspace] = None,
        block_index: Optional[int] = None,
    ):
        attn = block.self_attn
        return attn(
            x_attn,
            seq_lens=None,
            grid_sizes=None,
            freqs=None,
            block_mask=None,
            sparse_forward=self._self_attn_sparse_impl,
            sparse_forward_kwargs={
                "rope_positions": rope_positions,
                "kv_cache": kv_cache,
                "cache_end": cache_end,
                "prefix_cache": prefix_cache,
                "collect_indices": collect_indices,
                "kv_workspace": kv_workspace,
                "block_index": block_index,
            },
        )

    def _self_attn_sparse_impl(
        self,
        *,
        attn,
        x_attn: torch.Tensor,
        rope_positions: torch.Tensor,
        kv_cache: Optional[dict],
        cache_end: Optional[int],
        prefix_cache: Optional[dict] = None,
        collect_indices: Optional[torch.Tensor] = None,
        kv_workspace: Optional[ChunkKVWorkspace] = None,
        block_index: Optional[int] = None,
    ):
        b, s, n, d = x_attn.shape[0], x_attn.shape[1], attn.num_heads, attn.head_dim
        q = attn.norm_q(attn.q(x_attn)).view(b, s, n, d)
        k = attn.norm_k(attn.k(x_attn)).view(b, s, n, d)
        v = attn.v(x_attn).view(b, s, n, d)

        sp_size = int(getattr(attn, "sp_size", 1) or 1)
        if sp_size > 1:
            # Ulysses: x_attn carries this rank's shard of the active tokens.  Trade
            # it for a head shard so RoPE, the cache history and the collected KV all
            # cover the full active set, exactly as they do on a single GPU.
            s = int(rope_positions.shape[1])
            q = all_to_all(q, scatter_dim=2, gather_dim=1, group=attn.sp_group)[:, :s]
            k = all_to_all(k, scatter_dim=2, gather_dim=1, group=attn.sp_group)[:, :s]
            v = all_to_all(v, scatter_dim=2, gather_dim=1, group=attn.sp_group)[:, :s]

        roped_q = rope_apply_positions(q, rope_positions, self.freqs).type_as(v)
        roped_k = rope_apply_positions(k, rope_positions, self.freqs).type_as(v)

        prefix_k = prefix_cache["k"] if prefix_cache is not None else None
        prefix_v = prefix_cache["v"] if prefix_cache is not None else None
        prefix_tokens = 0 if prefix_k is None else int(prefix_k.shape[1])
        hist_k, hist_v = self._history_kv(
            kv_cache,
            cache_end=cache_end,
            attn=attn,
            reserved_tokens=s + prefix_tokens,
        )
        key = self._cat_existing([*hist_k, prefix_k, roped_k])
        value = self._cat_existing([*hist_v, prefix_v, v])
        y = attention(roped_q, key, value)

        selected_cache = None
        if collect_indices is not None:
            if kv_workspace is not None:
                kv_workspace.collect_block(block_index, roped_k, v, collect_indices)
            else:
                selected_cache = {
                    "k": roped_k.index_select(1, collect_indices),
                    "v": v.index_select(1, collect_indices),
                }
        if sp_size > 1:
            # Restore the active-token sharding expected by the rest of the block.
            y = all_to_all(
                pad_to_multiple(y, sp_size, dim=1),
                scatter_dim=1,
                gather_dim=2,
                group=attn.sp_group,
            )
        y = attn.o(y.flatten(2))
        return y, selected_cache

    def _run_sparse_block(
        self,
        block,
        x: torch.Tensor,
        e: torch.Tensor,
        rope_positions: torch.Tensor,
        context: torch.Tensor,
        context_lens,
        kv_cache: Optional[dict],
        crossattn_cache: Optional[dict],
        cache_end: Optional[int],
        prefix_cache: Optional[dict] = None,
        collect_indices: Optional[torch.Tensor] = None,
        kv_workspace: Optional[ChunkKVWorkspace] = None,
        block_index: Optional[int] = None,
    ):
        e = (block.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        e = tuple(u.squeeze(2) for u in e)
        y, selected_cache = self._self_attn_sparse(
            block=block,
            x_attn=block.norm1(x) * (1 + e[1]) + e[0],
            rope_positions=rope_positions,
            kv_cache=kv_cache,
            cache_end=cache_end,
            prefix_cache=prefix_cache,
            collect_indices=collect_indices,
            kv_workspace=kv_workspace,
            block_index=block_index,
        )
        x = x + y * e[2]
        x = x + block.cross_attn(block.norm3(x), context, context_lens, crossattn_cache=crossattn_cache)
        y = block.ffn(block.norm2(x) * (1 + e[4]) + e[3])
        x = x + y * e[5]
        return x, selected_cache

    def _invoke_sparse_block(
        self,
        *,
        block,
        x: torch.Tensor,
        e: torch.Tensor,
        rope_positions: torch.Tensor,
        context: torch.Tensor,
        context_lens,
        kv_cache: Optional[dict],
        crossattn_cache: Optional[dict],
        cache_end: Optional[int],
        prefix_cache: Optional[dict] = None,
        collect_indices: Optional[torch.Tensor] = None,
        kv_workspace: Optional[ChunkKVWorkspace] = None,
        block_index: Optional[int] = None,
    ):
        """Enter a sparse block through its forward boundary for FSDP."""

        return block(
            x,
            e=e,
            seq_lens=None,
            grid_sizes=None,
            freqs=None,
            context=context,
            context_lens=context_lens,
            block_mask=None,
            sparse_forward=self._run_sparse_block,
            sparse_forward_kwargs={
                "rope_positions": rope_positions,
                "kv_cache": kv_cache,
                "crossattn_cache": crossattn_cache,
                "cache_end": cache_end,
                "prefix_cache": prefix_cache,
                "collect_indices": collect_indices,
                "kv_workspace": kv_workspace,
                "block_index": block_index,
            },
        )

    def _run_sparse_block_maybe_checkpoint(
        self,
        block,
        x: torch.Tensor,
        e: torch.Tensor,
        rope_positions: torch.Tensor,
        context: torch.Tensor,
        context_lens,
        kv_cache: Optional[dict],
        crossattn_cache: Optional[dict],
        cache_end: Optional[int],
        prefix_cache: Optional[dict] = None,
        collect_indices: Optional[torch.Tensor] = None,
        kv_workspace: Optional[ChunkKVWorkspace] = None,
        block_index: Optional[int] = None,
    ):
        if not (torch.is_grad_enabled() and self.gradient_checkpointing):
            return self._invoke_sparse_block(
                block=block,
                x=x,
                e=e,
                rope_positions=rope_positions,
                context=context,
                context_lens=context_lens,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                cache_end=cache_end,
                prefix_cache=prefix_cache,
                collect_indices=collect_indices,
                kv_workspace=kv_workspace,
                block_index=block_index,
            )

        if collect_indices is None or kv_workspace is not None:
            def custom_forward(tokens):
                return self._invoke_sparse_block(
                    block=block,
                    x=tokens,
                    e=e,
                    rope_positions=rope_positions,
                    context=context,
                    context_lens=context_lens,
                    kv_cache=kv_cache,
                    # Cross-attention cache mutates on first use, which breaks
                    # checkpoint recomputation consistency.
                    crossattn_cache=None,
                    cache_end=cache_end,
                    prefix_cache=prefix_cache,
                    collect_indices=collect_indices,
                    kv_workspace=kv_workspace,
                    block_index=block_index,
                )[0]

            return torch.utils.checkpoint.checkpoint(
                custom_forward,
                x,
                use_reentrant=False,
            ), None

        def custom_forward(tokens):
            output, selected_cache = self._invoke_sparse_block(
                block=block,
                x=tokens,
                e=e,
                rope_positions=rope_positions,
                context=context,
                context_lens=context_lens,
                kv_cache=kv_cache,
                # Cross-attention cache mutates on first use, which breaks
                # checkpoint recomputation consistency.
                crossattn_cache=None,
                cache_end=cache_end,
                prefix_cache=prefix_cache,
                collect_indices=collect_indices,
            )
            return output, selected_cache["k"], selected_cache["v"]

        output, cache_k, cache_v = torch.utils.checkpoint.checkpoint(
            custom_forward,
            x,
            use_reentrant=False,
        )
        return output, {"k": cache_k, "v": cache_v}

    def _unpatchify_sparse(
        self,
        token_output: torch.Tensor,
        token_indices: torch.Tensor,
        grid_shapes,
    ):
        c = self.out_dim
        out = []
        for u, v in zip(token_output, grid_shapes):
            dense = u.new_zeros((math.prod(v), math.prod(self.patch_size) * c))
            dense.index_copy_(0, token_indices, u)
            dense = dense.view(*v, *self.patch_size, c)
            dense = torch.einsum('fhwpqrc->cfphqwr', dense)
            dense = dense.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(dense)
        return torch.stack(out)

    def _write_indexed_chunk_cache(
        self,
        kv_cache,
        indexed_kv_cache,
        chunk_tokens: int,
        cache_start: int,
        error_prefix: str,
    ):
        for block_index, cache in enumerate(kv_cache):
            global_end = int(cache["global_end_index"].item())
            local_end_index = int(cache["local_end_index"].item())
            if global_end == int(cache_start + chunk_tokens):
                local_start = local_end_index - chunk_tokens
            elif global_end == int(cache_start):
                local_start = local_end_index
            else:
                raise RuntimeError(
                    f"Unexpected KV cache global_end_index={global_end}; "
                    f"expected {cache_start} or {cache_start + chunk_tokens}"
                )
            local_end = local_start + chunk_tokens
            if local_end > cache["k"].shape[1]:
                raise RuntimeError(f"{error_prefix} cache write exceeds allocated KV cache size")

            dense_k = cache["k"].new_zeros((1, chunk_tokens, cache["k"].shape[2], cache["k"].shape[3]))
            dense_v = cache["v"].new_zeros((1, chunk_tokens, cache["v"].shape[2], cache["v"].shape[3]))
            for piece in indexed_kv_cache[block_index]:
                dense_k.index_copy_(1, piece["indices"], piece["k"].to(device=dense_k.device, dtype=dense_k.dtype))
                dense_v.index_copy_(1, piece["indices"], piece["v"].to(device=dense_v.device, dtype=dense_v.dtype))

            cache["k"][:, local_start:local_end] = dense_k
            cache["v"][:, local_start:local_end] = dense_v
            cache["global_end_index"].fill_(int(cache_start + chunk_tokens))
            cache["local_end_index"].fill_(local_end)
