"""Exit and reference branches of one never-committed target chunk for router training."""
from dataclasses import dataclass
from typing import Optional
import torch
from .cache import PatchWorkspace


@dataclass
class TargetBranches:
    active_ids: torch.Tensor
    x0_exit: torch.Tensor
    x0_ref: torch.Tensor
    hidden: Optional[torch.Tensor]
    exits: Optional[torch.Tensor]


@torch.no_grad()
def target_branches(rollout, inputs, noise_rows, *, index, cache, transition_noise,
                    selected_step, record=False):
    """Run RouterRollout.chunk's head decisions up to the selected step s, then fork.

    The patches A still active at s give x0_exit (their step-s estimate) and the head features
    at s; x0_ref continues all of A to step 4. record also continues chunk()'s own decisions
    from s, giving its hard exit steps. Neither branch is committed to the cache.
    """
    s = int(selected_step)
    if s not in (1, 2, 3):
        raise ValueError("selected_step must be 1, 2 or 3")
    device = noise_rows.device
    total = noise_rows.shape[1]
    current = noise_rows.clone()
    exits = torch.full((total,), 4, device=device, dtype=torch.long)
    active = torch.ones(total, device=device, dtype=torch.bool)
    refresh = torch.empty(0, device=device, dtype=torch.long)
    workspace = PatchWorkspace()
    times = rollout.times.to(device=device)

    def forward(rows, step, ids, refresh, features):
        return rollout.backbone.forward(
            inputs=inputs, rows=rows, timestep=times[step - 1], chunk_index=index, cache=cache,
            active_ids=ids, refresh_ids=refresh, workspace=workspace, features=features)

    def advance(step, ids, x0, hidden):
        # RouterRollout.chunk's update of `current` after one denoising step.
        if step == 4:
            stop = torch.ones(ids.numel(), device=device, dtype=torch.bool)
        else:
            stop = rollout.head(hidden).reshape(-1).sigmoid() >= rollout.threshold
        if inputs.sp_enabled:
            from ..backbone.distributed import broadcast_sp_tensor
            broadcast_sp_tensor(stop)
        stopped = ids[stop]
        current[:, stopped] = x0[:, stop]
        exits[stopped] = step
        active[stopped] = False
        if step < 4:
            remain = ids[~stop]
            t = times[step]
            current[:, remain] = t * x0[:, ~stop] + (1 - t) * transition_noise[step - 1][:, remain]
        return stopped

    for step in range(1, s + 1):
        ids = torch.where(active)[0]
        if not ids.numel():
            empty = current[:, ids]
            return TargetBranches(ids, empty, empty, None, exits if record else None)
        x0, hidden = forward(current, step, ids, refresh, True)
        if step < s:
            refresh = advance(step, ids, x0, hidden)
    # One PatchWorkspace serves both branches: it is written only for refreshed patches. The
    # reference never refreshes (A runs to step 4), and the record branch, run last, refreshes
    # only patches of A, whose saved K/V the reference never reads (it recomputes all of A).
    active_ids, t = ids, times[s]
    reference = current.clone()
    reference[:, active_ids] = t * x0 + (1 - t) * transition_noise[s - 1][:, active_ids]
    x0_ref = x0
    for step in range(s + 1, 5):
        x0_ref, _ = forward(reference, step, active_ids, active_ids[:0], False)
        if step < 4:
            t = times[step]
            reference[:, active_ids] = t * x0_ref + (1 - t) * transition_noise[step - 1][:, active_ids]
    del reference
    if record:
        refresh = advance(s, active_ids, x0, hidden)
        for step in range(s + 1, 5):
            ids = torch.where(active)[0]
            if not ids.numel():
                break
            x, h = forward(current, step, ids, refresh, step < 4)
            refresh = advance(step, ids, x, h)
    return TargetBranches(active_ids, x0, x0_ref, hidden, exits if record else None)
