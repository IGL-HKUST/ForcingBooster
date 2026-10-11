"""Common full-step self-supervision objective and training contract."""
import math

import torch


def validate_router_training_config(config):
    """Reject obsolete training controls instead of silently ignoring them."""
    values = config if hasattr(config, 'keys') else vars(config)
    removed = []
    for key in values:
        name = key.removeprefix('router_')
        if name in {'training_mode', 'mse_weight'} or name.startswith(('lpips', 'nfe_cost_')):
            removed.append(key)
    if removed:
        raise ValueError(
            f"Removed router training options: {', '.join(sorted(removed))}. "
            'Remove these fields; training always preserves exit history and uses '
            'latent MSE plus uniform NFE. Use nfe_weight for the trade-off.'
        )


def router_training_contract(nfe_weight):
    weight = float(nfe_weight)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('nfe_weight must be finite and non-negative')
    return dict(version=1, objective='latent_mse_plus_uniform_nfe',
                normalization='active_patches_per_rank', exit_history='preserved',
                transition_noise='explicit_index', nfe_weight=weight)


def validate_router_resume(saved, nfe_weight):
    if saved != router_training_contract(nfe_weight):
        raise ValueError('Router training contract mismatch or old checkpoint: '
                         'load weights as initialization instead of resuming optimizer state')


def router_objective(*, logits, exit_mse, continue_steps, selected_step,
                     total_patches, nfe_weight=0.03):
    """Average over local active patches; an empty rank contributes zero."""
    router_training_contract(nfe_weight)
    probs = torch.sigmoid(logits.float()).reshape(-1)
    error = exit_mse.detach().to(probs).reshape(-1)
    cost = continue_steps.detach().to(probs).reshape(-1)
    if (probs.shape != error.shape or probs.shape != cost.shape
            or total_patches <= 0 or probs.numel() > total_patches):
        raise ValueError('Invalid router objective patch shapes or total patch count')
    # Callers provide only active logits. Keep the empty sum differentiable
    # while avoiding division by zero on ranks with no active patches.
    denominator = max(probs.numel(), 1)
    mse = (probs * error).sum() / denominator
    expected_steps = probs * selected_step + (1 - probs) * cost
    nfe = expected_steps.sum() / denominator
    loss = mse + float(nfe_weight) * nfe
    zero = loss.detach().new_zeros(())
    metrics = dict(router_loss=loss.detach(), router_mse=mse.detach(), router_nfe=nfe.detach(),
                   router_nfe_active_mean=expected_steps.detach().mean() if probs.numel() else zero,
                   router_exit_mse_mean=error.mean() if probs.numel() else zero,
                   router_exit_mse_std=error.std(unbiased=False) if probs.numel() else zero,
                   router_exit_prob=probs.detach().mean() if probs.numel() else zero,
                   router_skip_iter=zero + int(probs.numel() == 0))
    return loss, metrics
