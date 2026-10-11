"""Shared-noise replay and disposable cache transactions for LongLive."""
import copy

import torch

from methods.Wan21.shared.rollout.router_rollout import RouterRollout


def clone_cache(cache):
    if cache is None:
        return None
    return [{key: (value.detach().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value))
             for key, value in layer.items() if key != "_router_workspace"} for layer in cache]


class LongLiveRollout(RouterRollout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._noise_key = None
        self._noise_bank = None

    def reset_noise(self):
        self._noise_key = self._noise_bank = None

    def bind_noise(self, noisy_input, current_start_frame, bank=None):
        key = (noisy_input.data_ptr(), int(current_start_frame), tuple(noisy_input.shape))
        if key != self._noise_key or bank is not None:
            self._noise_key = key
            self._noise_bank = bank if bank is not None else [torch.randn_like(noisy_input) for _ in range(self.num_steps-1)]

    def add_patch_noise(self, pred_x0, next_timestep_tokens):
        if self._noise_bank is None:
            raise RuntimeError("Bind a chunk noise bank before sampling")
        next_value = float(next_timestep_tokens.reshape(-1)[0])
        values = self.denoising_step_list[1:].tolist()
        index = min(range(len(values)), key=lambda i: abs(values[i]-next_value))
        if abs(values[index]-next_value) > 1e-3:
            raise ValueError("Unexpected transition timestep")
        sigma = self.patch_values_to_latent(self._lookup_precomputed_sigma(next_timestep_tokens), pred_x0)
        return ((1-sigma)*pred_x0 + sigma*self._noise_bank[index]).to(pred_x0.dtype)

    def run_router(self, **kwargs):
        self.bind_noise(kwargs["noisy_input"], kwargs["current_start_frame"])
        if not kwargs.get("commit_cache", True):
            kwargs = dict(kwargs, kv_cache=clone_cache(kwargs["kv_cache"]),
                          crossattn_cache=clone_cache(kwargs["crossattn_cache"]), commit_cache=True)
        return super().run_router(**kwargs)

    def run_chunk_fixed_steps(self, **kwargs):
        self.bind_noise(kwargs["noisy_input"], kwargs["current_start_frame"])
        if not kwargs.get("write_cache", True):
            kwargs = dict(kwargs, kv_cache=clone_cache(kwargs["kv_cache"]),
                          crossattn_cache=clone_cache(kwargs["crossattn_cache"]), write_cache=True)
        return super().run_chunk_fixed_steps(**kwargs)

    def logits_at_step(self, **kwargs):
        self.bind_noise(kwargs["noisy_input"], kwargs["current_start_frame"])
        kwargs = dict(kwargs, kv_cache=clone_cache(kwargs["kv_cache"]),
                      crossattn_cache=clone_cache(kwargs["crossattn_cache"]))
        return super().logits_at_step(**kwargs)
