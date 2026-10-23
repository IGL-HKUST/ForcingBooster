"""Latent MSE plus uniform NFE over active patches, and the contract a resume must match."""
import math
import torch

CONTRACT_VERSION = 2


def h3_router_contract(nfe_weight, kv_commit="full"):
    weight = float(nfe_weight)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("nfe_weight must be finite and non-negative")
    contract = dict(version=CONTRACT_VERSION, model="solarwm-h3-stage2", normalization="active_patches",
                    objective="latent_mse_plus_uniform_nfe", exit_history="preserved",
                    reference="active_at_decision_to_step4", transition_noise="explicit_index",
                    nfe_weight=weight)
    if kv_commit != "full":
        # History chunks are rolled out with this cache rule; older bundles imply "full".
        contract["kv_commit"] = kv_commit
    return contract


def continue_cost(exit_steps, selected_step, rule, num_steps=4):
    """Steps an active patch is charged if it does not exit at ``selected_step``."""
    if rule == "four":
        return torch.full_like(exit_steps, num_steps, dtype=torch.float32)
    if rule == "wan":
        target = torch.full_like(exit_steps, min(selected_step + 1, num_steps))
        return torch.maximum(exit_steps, target).float()
    raise ValueError(f"Unknown continue cost rule {rule!r}; expected 'four' or 'wan'")


def router_objective(*, logits, exit_mse, continue_steps, selected_step, nfe_weight):
    """Same math as the Wan21 objective; no active patch gives a differentiable zero."""
    weight = h3_router_contract(nfe_weight)["nfe_weight"]
    probs = torch.sigmoid(logits.float()).reshape(-1)
    error = exit_mse.detach().to(probs).reshape(-1)
    cost = continue_steps.detach().to(probs).reshape(-1)
    if probs.shape != error.shape or probs.shape != cost.shape:
        raise ValueError(f"Router objective shapes differ: logits {tuple(probs.shape)}, "
                         f"exit_mse {tuple(error.shape)}, continue_steps {tuple(cost.shape)}")
    n = max(probs.numel(), 1)
    mse = (probs * error).sum() / n
    nfe = (probs * selected_step + (1 - probs) * cost).sum() / n
    loss = mse + weight * nfe
    metrics = dict(loss=float(loss), mse=float(mse), nfe=float(nfe), weighted_nfe=weight * float(nfe),
                   exit_prob=float(probs.mean()) if probs.numel() else 0.0,
                   active_count=int(probs.numel()))
    return loss, metrics
