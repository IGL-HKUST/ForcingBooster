"""Native four-step generation and patch early exits."""
from dataclasses import dataclass
import torch
from .cache import PatchWorkspace


@dataclass
class ChunkResult:
    rows: torch.Tensor
    steps: torch.Tensor


class RouterRollout:
    """kv_commit "wan": a patch's K/V from its clean refresh forward become its cross-chunk cache;
    the chunk-end forward only computes patches still lacking clean K/V (those finishing the last
    step). "full" recomputes the whole chunk at t=1 for the cache, as the native sampler does."""

    def __init__(self, backbone, head, *, threshold=0.5, kv_commit="wan"):
        from ..backbone.sgf import h3_sgf_schedule
        if kv_commit not in ("wan", "full"):
            raise ValueError("kv_commit must be wan or full")
        self.backbone, self.head, self.threshold, self.kv_commit = backbone, head, threshold, kv_commit
        self.times = h3_sgf_schedule()[0].timesteps

    def transitions(self, noise, generator, inputs):
        from ..backbone.layout import patchify_video
        from ..backbone.sgf_rollout import _random_like
        return tuple(patchify_video(_random_like(noise, inputs=inputs, generator=generator))
                     for _ in range(3))

    @torch.no_grad()
    def chunk(self, inputs, noise_rows, *, index, cache, transition_noise,
              forced_steps=None, commit=True):
        total = noise_rows.shape[1]
        current = noise_rows.clone()
        final = torch.empty_like(current)
        exits = torch.full((total,), 4, device=current.device, dtype=torch.long)
        active = torch.ones(total, device=current.device, dtype=torch.bool)
        refresh = torch.empty(0, device=current.device, dtype=torch.long)
        workspace = PatchWorkspace()
        times = self.times.to(device=current.device)
        for step in range(1, 5):
            ids = torch.where(active)[0]
            if not ids.numel():
                break
            x0, hidden = self.backbone.forward(
                inputs=inputs, rows=current, timestep=times[step - 1], chunk_index=index,
                cache=cache, active_ids=ids, refresh_ids=refresh, workspace=workspace,
                features=(step < 4 and forced_steps is None))
            if step == 4:
                stop = torch.ones(ids.numel(), device=ids.device, dtype=torch.bool)
            elif forced_steps is not None:
                stop = forced_steps[ids] == step
            else:
                stop = self.head(hidden).reshape(-1).sigmoid() >= self.threshold
            if inputs.sp_enabled:
                from ..backbone.distributed import broadcast_sp_tensor
                broadcast_sp_tensor(stop)
            stopped = ids[stop]
            final[:, stopped] = x0[:, stop]
            current[:, stopped] = x0[:, stop]  # clean input on the next refresh
            exits[stopped] = step
            active[stopped] = False
            if step < 4:
                remain = ids[~stop]
                t = times[step]
                current[:, remain] = t * x0[:, ~stop] + (1 - t) * transition_noise[step - 1][:, remain]
            refresh = stopped
        if commit and self.kv_commit == "full":
            self.backbone.commit(inputs, final, index, cache)
        elif commit:
            # `refresh` holds the last exits, whose clean K/V the workspace does not have yet.
            self.backbone.forward(inputs=inputs, rows=final, timestep=final.new_tensor(1.0),
                                  chunk_index=index, cache=cache, active_ids=refresh,
                                  refresh_ids=refresh[:0], workspace=workspace, commit=True)
        return ChunkResult(final, exits)

    @torch.no_grad()
    def native_chunk(self, inputs, noise, *, index, cache, transition_noise):
        from ..backbone.sgf_rollout import h3_student_forward
        from ..backbone.layout import patchify_video, unpatchify_video
        current = noise
        times = self.times.to(noise.device)
        for step, t in enumerate(times):
            clean = h3_student_forward(self.backbone.model, inputs, current, t,
                                      chunk_index=index, cache=cache)
            if step < 3:
                n = unpatchify_video(transition_noise[step], 5, inputs.latent_height, inputs.latent_width)
                current = times[step + 1] * clean + (1 - times[step + 1]) * n
        h3_student_forward(self.backbone.model, inputs, clean, noise.new_tensor(1.0),
                           chunk_index=index, cache=cache, commit_cache=True)
        rows = patchify_video(clean)
        return ChunkResult(rows, torch.full((rows.shape[1],), 4, device=rows.device, dtype=torch.long))

