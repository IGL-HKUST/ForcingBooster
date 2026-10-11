"""Detached raw (post QK norm, pre RoPE) caches and per-chunk refresh state."""
from dataclasses import dataclass, field
import torch


@dataclass
class RawKVCache:
    layers: dict = field(default_factory=dict)

    def history(self, layer, chunk_index):
        return self.layers.get(layer, [])

    def commit(self, layer, chunk_index, key, value):
        self.layers[layer] = [*self.history(layer, chunk_index),
                              (chunk_index, key.detach().clone(), value.detach().clone())][-5:]

    def clear(self):
        self.layers.clear()


@dataclass
class PatchWorkspace:
    """Only previously clean-refreshed patch K/V persists across denoising calls."""
    layers: dict = field(default_factory=dict)

    def assemble(self, layer, key, value, compute_ids, refresh_ids, total):
        shape = (key.shape[0], total, *key.shape[2:])
        if layer not in self.layers:
            self.layers[layer] = (key.new_zeros(shape), value.new_zeros(shape))
        saved_k, saved_v = self.layers[layer]
        if refresh_ids.numel():
            slots = torch.searchsorted(compute_ids, refresh_ids)
            saved_k.index_copy_(1, refresh_ids, key.index_select(1, slots).detach())
            saved_v.index_copy_(1, refresh_ids, value.index_select(1, slots).detach())
        return (saved_k.index_copy(1, compute_ids, key),
                saved_v.index_copy(1, compute_ids, value))

