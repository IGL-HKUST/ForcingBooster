from typing import Optional

import torch

from methods.Wan21.shared.backbone.wan.modules.causal_model import CausalWanModel
from methods.Wan21.shared.backbone.wan.modules.causal_sparse_utils import CausalSparseTokenMixin, ChunkKVWorkspace


class AdaptiveCausalWanModel(CausalSparseTokenMixin, CausalWanModel):
    """Causal Wan extension for adaptive patch-wise sparse Q sequences."""

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
        clip_fea=None,
        y=None,
        collect_token_indices=None,
        collect_active_cache: bool = False,
        return_patch_exit_logits: bool = False,
        full_timestep=None,
        kv_workspace: Optional[ChunkKVWorkspace] = None,
        cache_only: bool = False,
    ):
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        tokens, grid_sizes, grid_shapes = self._video_tokens(x)
        if tokens.shape[0] != 1:
            raise ValueError("adaptive sparse inference currently supports batch_size == 1")
        active_token_indices = active_token_indices.to(device=tokens.device, dtype=torch.long)
        tokens = tokens.index_select(1, active_token_indices)
        active_rope_positions = rope_positions.index_select(1, active_token_indices)
        if tokens.shape[1] > seq_len:
            raise ValueError(f"sequence length {tokens.shape[1]} exceeds seq_len={seq_len}")
        context, context_lens = self._context_tokens(context, clip_fea=clip_fea)
        e0, eh = self._time_tokens_for_active_grid(
            t,
            tokens.dtype,
            active_token_indices=active_token_indices,
            grid_shapes=grid_shapes,
            full_timestep=full_timestep,
        )

        collect_positions = None
        cache_indices = None
        if collect_active_cache:
            collect_positions = torch.arange(tokens.shape[1], device=tokens.device)
            cache_indices = active_token_indices
        elif collect_token_indices is not None:
            collect_token_indices = collect_token_indices.to(device=tokens.device, dtype=torch.long)
            collect_positions = torch.searchsorted(active_token_indices, collect_token_indices)
            if kv_workspace is None and not torch.equal(
                active_token_indices.index_select(0, collect_positions), collect_token_indices
            ):
                raise ValueError("collect_token_indices must be a subset of active_token_indices")
            cache_indices = collect_token_indices

        if kv_workspace is not None and cache_indices is not None:
            kv_workspace.begin_piece(cache_indices)

        collected_cache = []
        for block_index, block in enumerate(self.blocks):
            prefix_cache = (
                kv_workspace.prefix_cache(block_index)
                if kv_workspace is not None
                else (clear_kv_cache[block_index] if clear_kv_cache is not None else None)
            )
            tokens, selected_cache = self._run_sparse_block_maybe_checkpoint(
                block=block,
                x=tokens,
                e=e0,
                rope_positions=active_rope_positions,
                context=context,
                context_lens=context_lens,
                kv_cache=kv_cache[block_index],
                crossattn_cache=crossattn_cache[block_index] if crossattn_cache is not None else None,
                cache_end=cache_end,
                prefix_cache=prefix_cache,
                collect_indices=collect_positions,
                kv_workspace=kv_workspace,
                block_index=block_index,
            )
            if cache_indices is not None and kv_workspace is None:
                collected_cache.append({**selected_cache, "indices": cache_indices})

        if kv_workspace is not None and cache_indices is not None:
            kv_workspace.finish_piece()

        patch_exit_logits = None
        if return_patch_exit_logits:
            if self.patch_exit_router_head is None:
                raise ValueError("return_patch_exit_logits=True requires patch_exit_router_head")
            patch_exit_logits = self.patch_exit_router_head(tokens, grid_sizes)

        out = None
        if not cache_only:
            out = self._unpatchify_sparse(self.head(tokens, eh), active_token_indices, grid_shapes)
        if cache_indices is not None:
            cache_result = None if kv_workspace is not None else collected_cache
            if patch_exit_logits is not None:
                return out, cache_result, patch_exit_logits
            return out, cache_result
        if patch_exit_logits is not None:
            return out, patch_exit_logits
        return out

    def write_sparse_chunk_cache(
        self,
        kv_cache,
        sparse_kv_cache,
        chunk_tokens: int,
        cache_start: int,
    ):
        indexed = [[sparse_kv_cache[block_index]] for block_index in range(len(kv_cache))]
        self._write_indexed_chunk_cache(
            kv_cache=kv_cache,
            indexed_kv_cache=indexed,
            chunk_tokens=chunk_tokens,
            cache_start=cache_start,
            error_prefix="adaptive sparse",
        )

    def forward(self, *args, adaptive_forward_mode: Optional[str] = None, **kwargs):
        if adaptive_forward_mode is None:
            return super().forward(*args, **kwargs)
        if adaptive_forward_mode == "sparse":
            return self.adaptive_sparse_forward(*args, **kwargs)
        raise ValueError(f"Invalid adaptive_forward_mode={adaptive_forward_mode}")
