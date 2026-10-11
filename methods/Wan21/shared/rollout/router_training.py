"""Training-only trajectory records, indexed noise and isolated cache branches."""
from dataclasses import dataclass
import copy

import torch

from .router_rollout import RouterRollout


def clone_training_cache(cache, *, device=None):
    if cache is None:
        return None
    return [{key: value.detach().to(device=value.device if device is None else device, copy=True)
             if torch.is_tensor(value) else copy.deepcopy(value)
             for key, value in layer.items() if key != '_router_workspace'} for layer in cache]


def offload_training_history(kv_cache, crossattn_cache):
    """Retire prefix GPU storage before recording the target chunk boundary.

    Only one boundary is retained, not per-step snapshots. Replace tensors in
    the owned cache dictionaries so aliases cannot retain a second GPU history.
    Values, dtypes, capacity, and committed cursors are preserved; each branch
    restores an independent GPU copy and cannot mutate this CPU boundary.
    """
    for cache in (kv_cache, crossattn_cache):
        if cache is None:
            continue
        for layer in cache:
            layer.pop('_router_workspace', None)
            for key, value in layer.items():
                if torch.is_tensor(value):
                    layer[key] = value.detach().to(device='cpu')


def preserving_exit_map(steps_map, selected_step):
    return steps_map.clamp(max=selected_step)


@dataclass
class TrajectoryRecord:
    noisy_input: torch.Tensor
    current_start_frame: int
    kv_cache: list
    crossattn_cache: list
    transition_noise: tuple[torch.Tensor, ...]
    steps_map: torch.Tensor


class TrainingTrajectoryMixin:
    """Use with a method rollout; inference does not instantiate this mixin."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clear_training_noise()

    def clear_training_noise(self):
        self._transition_noise = None

    def sample_transition_noise(self, noisy_input):
        return torch.randn_like(noisy_input)

    def _write_final_cache(self, kv_cache, num_tokens, cache_start):
        super()._write_final_cache(kv_cache, num_tokens, cache_start)
        # Committed KV and cursors are the history. The per-chunk collection
        # buffers are scratch space; retaining them would keep another full
        # chunk of KV alive while an isolated branch allocates its workspace.
        kv_cache[0].pop('_router_workspace', None)

    @torch.no_grad()
    def record_traj(self, **kwargs):
        self.clear_training_noise()
        self._transition_noise = tuple(self.sample_transition_noise(kwargs['noisy_input'])
                                       for _ in range(self.num_steps - 1))
        try:
            result = self.run_router(**dict(kwargs, commit_cache=False))
            return TrajectoryRecord(
                kwargs['noisy_input'], kwargs['current_start_frame'],
                kwargs['kv_cache'], kwargs['crossattn_cache'],
                self._transition_noise, result.steps_map.detach())
        except BaseException:
            self.clear_training_noise()
            raise

    def bind_traj(self, record):
        self._transition_noise = record.transition_noise

    def add_transition_noise(self, pred_x0, next_timestep_tokens, *, transition_index):
        if self._transition_noise is None:
            return super().add_transition_noise(
                pred_x0, next_timestep_tokens, transition_index=transition_index)
        if not 0 <= transition_index < len(self._transition_noise):
            raise ValueError('Invalid trajectory transition index')
        sigma = self.patch_values_to_latent(self._lookup_precomputed_sigma(next_timestep_tokens), pred_x0)
        return ((1 - sigma) * pred_x0 + sigma * self._transition_noise[transition_index]).to(pred_x0.dtype)

    def run_router(self, **kwargs):
        if not kwargs.get('commit_cache', True):
            device = kwargs['noisy_input'].device
            kwargs = dict(kwargs, kv_cache=clone_training_cache(kwargs['kv_cache'], device=device),
                          crossattn_cache=clone_training_cache(kwargs['crossattn_cache'], device=device), commit_cache=True)
        return super().run_router(**kwargs)

    def run_chunk_fixed_steps(self, **kwargs):
        if not kwargs.get('write_cache', True):
            device = kwargs['noisy_input'].device
            kwargs = dict(kwargs, kv_cache=clone_training_cache(kwargs['kv_cache'], device=device),
                          crossattn_cache=clone_training_cache(kwargs['crossattn_cache'], device=device), write_cache=True)
        return super().run_chunk_fixed_steps(**kwargs)

    def logits_at_step(self, **kwargs):
        device = kwargs['noisy_input'].device
        return super().logits_at_step(**dict(
            kwargs, kv_cache=clone_training_cache(kwargs['kv_cache'], device=device),
            crossattn_cache=clone_training_cache(kwargs['crossattn_cache'], device=device)))


class OnPolicyTrainingRollout(TrainingTrajectoryMixin, RouterRollout):
    pass
