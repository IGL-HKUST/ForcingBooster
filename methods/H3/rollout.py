"""Native four-step generation and patch early exits."""
from dataclasses import dataclass
import torch
from .cache import PatchWorkspace


@dataclass
class ChunkResult:
    rows: torch.Tensor
    steps: torch.Tensor


class RouterRollout:
    def __init__(self, backbone, head, *, threshold=0.5):
        from solarwm.backends.minimax_h3.sgf import h3_sgf_schedule
        self.backbone, self.head, self.threshold = backbone, head, threshold
        self.times = h3_sgf_schedule()[0].timesteps

    def transitions(self, noise, generator, inputs):
        from solarwm.backends.minimax_h3.layout import patchify_video
        from solarwm.backends.minimax_h3.sgf_rollout import _random_like
        return tuple(patchify_video(_random_like(noise, inputs=inputs, generator=generator))
                     for _ in range(3))

    @torch.no_grad()
    def chunk(self, inputs, noise_rows, *, index, cache, transition_noise,
              forced_steps=None):
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
                from solarwm.backends.minimax_h3.distributed import broadcast_sp_tensor
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
        self.backbone.commit(inputs, final, index, cache)
        return ChunkResult(final, exits)

    @torch.no_grad()
    def native_chunk(self, inputs, noise, *, index, cache, transition_noise):
        from solarwm.backends.minimax_h3.sgf_rollout import h3_student_forward
        from solarwm.backends.minimax_h3.layout import patchify_video, unpatchify_video
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

