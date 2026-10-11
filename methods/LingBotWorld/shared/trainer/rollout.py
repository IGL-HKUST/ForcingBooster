"""Router rollout with LingBot's upstream random-number layout."""

from __future__ import annotations

import torch

from methods.Wan21.shared.rollout.router_rollout import RouterChunkResult, RouterRollout
from .noise import randn_cfhw_as_ntchw


class LingBotRouterRollout(RouterRollout):
    def __init__(self, *args, noise_generator: torch.Generator | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.noise_generator = noise_generator
        self.distributed_lockstep = True

    def set_noise_generator(self, generator: torch.Generator) -> None:
        self.noise_generator = generator

    def add_patch_noise(self, pred_x0, next_timestep_tokens):
        if self.noise_generator is None:
            raise RuntimeError("LingBot rollout requires an explicit per-sample noise generator")
        noise = randn_cfhw_as_ntchw(pred_x0, generator=self.noise_generator)
        sigma = self._lookup_precomputed_sigma(next_timestep_tokens)
        sigma = self.patch_values_to_latent(sigma, pred_x0)
        return ((1 - sigma) * pred_x0 + sigma * noise).type_as(noise)

    def _forward_dense_committing(
        self,
        *,
        latents,
        conditional_dict,
        full_timestep,
        kv_cache,
        crossattn_cache,
        current_start,
    ):
        """Run the Fast model's cache-writing dense forward."""

        if full_timestep.ndim != 2 or full_timestep.shape[0] != 1:
            raise ValueError(
                "LingBot dense fixed-step timestep must have shape [1, tokens]"
            )
        _, pred_x0 = self.generator(
            noisy_image_or_video=latents,
            conditional_dict=conditional_dict,
            # Upstream passes one scalar timestep per sample and expands it to
            # the token grid inside the model.
            timestep=full_timestep[:, 0],
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            write_cache=True,
            precomputed_sigma=self._lookup_precomputed_sigma(full_timestep),
        )
        return pred_x0

    def run_chunk_dense_fixed_steps(
        self,
        *,
        noisy_input,
        conditional_dict,
        current_start_frame: int,
        kv_cache,
        crossattn_cache,
    ) -> RouterChunkResult:
        """Run every configured denoising step without entering sparse mode."""

        patch_shape, _, num_tokens, _, timesteps, context_timestep, current_start = (
            self._common_state(noisy_input, current_start_frame)
        )
        device = noisy_input.device

        def full_timestep(fill_value):
            return torch.full(
                (1, num_tokens),
                float(fill_value),
                device=device,
                dtype=torch.float32,
            )

        latents = noisy_input
        seq_len_log = []
        for step_index, timestep in enumerate(timesteps):
            pred_x0 = self._forward_dense_committing(
                latents=latents,
                conditional_dict=conditional_dict,
                full_timestep=full_timestep(timestep),
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
            )
            seq_len_log.append(
                {
                    "stage": int(step_index),
                    "mode": "dense",
                    "seq_len": int(num_tokens),
                }
            )
            if step_index + 1 < len(timesteps):
                latents = self.add_patch_noise(
                    pred_x0.detach(),
                    full_timestep(timesteps[step_index + 1]),
                ).detach()
            else:
                latents = pred_x0

        self._forward_dense_committing(
            latents=latents,
            conditional_dict=conditional_dict,
            full_timestep=full_timestep(context_timestep),
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
        )
        seq_len_log.append(
            {
                "stage": int(self.num_steps),
                "mode": "dense@clear",
                "seq_len": int(num_tokens),
            }
        )

        steps_map = torch.full(
            patch_shape, self.num_steps, device=device, dtype=torch.long
        )
        dense_reference_nfe = self.num_steps * num_tokens
        chunk_log = {
            "max_steps": int(self.num_steps),
            "patch_counts": {
                step: int(num_tokens if step == self.num_steps else 0)
                for step in range(1, self.num_steps + 1)
            },
            "patch_avg_step": float(self.num_steps),
            "router_nfe": int(dense_reference_nfe),
            "dense_reference_nfe": int(dense_reference_nfe),
            "nfe_ratio": 1.0,
            "speedup": 1.0,
            "cache_clear_forwards": 1,
            "cache_clear_tokens": int(num_tokens),
        }
        if self.num_steps == 4:
            chunk_log["full4_nfe"] = int(dense_reference_nfe)
        return RouterChunkResult(
            latents=latents,
            steps_map=steps_map,
            seq_len_log=seq_len_log,
            chunk_log=chunk_log,
        )

    def run_chunk_dense_fixed4(self, **kwargs) -> RouterChunkResult:
        """Compatibility alias for the released four-step dense baseline."""

        if self.num_steps != 4:
            raise ValueError("run_chunk_dense_fixed4 requires exactly four timesteps")
        return self.run_chunk_dense_fixed_steps(**kwargs)


__all__ = ["LingBotRouterRollout"]


from methods.Wan21.shared.rollout.router_training import TrainingTrajectoryMixin


class LingBotTrainingRollout(TrainingTrajectoryMixin, LingBotRouterRollout):
    """Isolated training branches with the original CFHW noise layout."""

    def sample_transition_noise(self, noisy_input):
        if self.noise_generator is None:
            raise RuntimeError("Training requires a per-sample noise generator")
        return randn_cfhw_as_ntchw(noisy_input, generator=self.noise_generator)
