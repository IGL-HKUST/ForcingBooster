from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from methods.Wan21.shared.backbone.wan.modules.causal_sparse_utils import ChunkKVWorkspace


@dataclass
class RouterChunkResult:
    latents: torch.Tensor
    steps_map: torch.Tensor
    seq_len_log: list[dict]
    chunk_log: dict


class RouterRollout:
    """Patch-wise causal rollout driven by an online binary exit router."""

    def __init__(
        self,
        *,
        generator,
        scheduler,
        denoising_step_list,
        frame_seq_length: int = 0,
        context_noise: int = 0,
        exit_threshold: float = 0.5,
    ) -> None:
        self.generator = generator
        self.scheduler = scheduler
        self.denoising_step_list = torch.as_tensor(denoising_step_list, dtype=torch.float32)
        if self.denoising_step_list.ndim != 1 or len(self.denoising_step_list) < 2:
            raise ValueError("router rollout requires at least 2 denoising steps")
        if not bool(torch.isfinite(self.denoising_step_list).all()):
            raise ValueError("router denoising steps must contain only finite values")
        self.num_steps = int(len(self.denoising_step_list))
        self.frame_seq_length = int(frame_seq_length)
        self.context_noise = int(context_noise)
        self.exit_threshold = float(exit_threshold)
        sigma_timesteps = self.denoising_step_list.detach().cpu().to(torch.float32)
        sigma_timesteps = torch.cat(
            [sigma_timesteps, torch.tensor([float(self.context_noise)], dtype=torch.float32)]
        )
        scheduler_timesteps = self.scheduler.timesteps.detach().cpu().double()
        scheduler_sigmas = self.scheduler.sigmas.detach().cpu()
        sigma_ids = torch.argmin(
            (scheduler_timesteps.unsqueeze(0) - sigma_timesteps.double().unsqueeze(1)).abs(),
            dim=1,
        )
        resolved_sigmas = scheduler_sigmas[sigma_ids]
        # Wan's training grid omits the terminal zero when extra_one_step=True.
        # The appended context timestep 0 belongs to the cache-only clean-KV
        # forward and must use sigma 0 rather than the nearest non-zero entry.
        resolved_sigmas = torch.where(
            sigma_timesteps == 0,
            torch.zeros_like(resolved_sigmas),
            resolved_sigmas,
        )
        order = sigma_timesteps.argsort()
        self._sigma_lookup_cpu = sigma_timesteps[order], resolved_sigmas[order]
        self._sigma_device_cache = {}

    def _lookup_precomputed_sigma(self, timestep: torch.Tensor):
        cached = self._sigma_device_cache.get(timestep.device)
        if cached is None:
            cached = tuple(value.to(timestep.device) for value in self._sigma_lookup_cpu)
            self._sigma_device_cache[timestep.device] = cached
        known_timesteps, known_sigmas = cached
        return known_sigmas[torch.searchsorted(known_timesteps, timestep)]

    def patch_shape(self, batch_size, current_num_frames, height, width):
        patch_t, patch_h, patch_w = self.generator.model.patch_size
        return (batch_size, current_num_frames // patch_t, height // patch_h, width // patch_w)

    def make_patch_positions(self, batch_size, current_num_frames, height, width, device, start_frame=0):
        patch_t, patch_h, patch_w = self.generator.model.patch_size
        patch_frame = current_num_frames // patch_t
        patch_height = height // patch_h
        patch_width = width // patch_w
        f = torch.arange(start_frame, start_frame + patch_frame, device=device)
        h = torch.arange(patch_height, device=device)
        w = torch.arange(patch_width, device=device)
        grid = torch.stack(torch.meshgrid(f, h, w, indexing="ij"), dim=-1).reshape(1, -1, 3)
        return grid.expand(batch_size, -1, -1).contiguous()

    def patch_values_to_latent(self, values, latents):
        batch_size, num_frame, _, height, width = latents.shape
        patch_t, patch_h, patch_w = self.generator.model.patch_size
        patch_frame = num_frame // patch_t
        patch_height = height // patch_h
        patch_width = width // patch_w
        values = values.view(batch_size, patch_frame, patch_height, patch_width)
        return (
            values.repeat_interleave(patch_t, dim=1)
            .repeat_interleave(patch_h, dim=2)
            .repeat_interleave(patch_w, dim=3)
            .unsqueeze(2)
        )

    def add_transition_noise(self, pred_x0, next_timestep_tokens, *, transition_index):
        # Training uses the explicit index; inference keeps its original RNG path.
        return self.add_patch_noise(pred_x0, next_timestep_tokens)

    def add_patch_noise(self, pred_x0, next_timestep_tokens):
        noise = torch.randn_like(pred_x0)
        sigma = self._lookup_precomputed_sigma(next_timestep_tokens)
        sigma = self.patch_values_to_latent(sigma, pred_x0)
        return ((1 - sigma) * pred_x0 + sigma * noise).type_as(noise)

    def merge_active_patches(self, current_latents, candidate_latents, active_mask):
        active = self.patch_values_to_latent(active_mask.to(dtype=current_latents.dtype), current_latents).bool()
        return torch.where(active, candidate_latents, current_latents)

    @staticmethod
    def _detach_sparse_cache(sparse_cache):
        if sparse_cache is None:
            return None
        detached = []
        for block in sparse_cache:
            detached.append(
                {
                    "k": block["k"].detach(),
                    "v": block["v"].detach(),
                    "indices": block["indices"].detach(),
                }
            )
        return detached

    @staticmethod
    def _merge_sparse_caches(cache_pieces):
        cache_pieces = [piece for piece in cache_pieces if piece is not None]
        if not cache_pieces:
            return None
        merged = []
        for block_index in range(len(cache_pieces[0])):
            merged.append(
                {
                    "k": torch.cat([piece[block_index]["k"] for piece in cache_pieces], dim=1),
                    "v": torch.cat([piece[block_index]["v"] for piece in cache_pieces], dim=1),
                    "indices": torch.cat([piece[block_index]["indices"] for piece in cache_pieces], dim=0),
                }
            )
        return merged

    def reset_cache_state(self, kv_cache):
        workspace = kv_cache[0].get("_router_workspace")
        if workspace is not None:
            workspace.global_end = workspace.local_end = 0
            workspace.direct_write = False
        for cache in kv_cache:
            cache["_host_position"] = (0, 0)

    @staticmethod
    def _kv_workspace_for_forward(kv_cache):
        workspace = kv_cache[0].get("_router_workspace")
        return workspace if workspace is not None and workspace.direct_write else None

    def _prefix_cache(self, cache_pieces, kv_cache):
        if self._kv_workspace_for_forward(kv_cache) is not None:
            return None
        return self._merge_sparse_caches(cache_pieces)

    def _forward_dense(
        self,
        *,
        latents,
        conditional_dict,
        full_timestep,
        kv_cache,
        crossattn_cache,
        current_start,
        cache_start,
        return_logits=False,
        requires_grad=False,
    ):
        model_timestep = self._dense_model_timestep(full_timestep)
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        with context:
            output = self.generator(
                noisy_image_or_video=latents,
                conditional_dict=conditional_dict,
                timestep=model_timestep,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                write_cache=False,
                return_patch_exit_logits=return_logits,
                precomputed_sigma=self._lookup_precomputed_sigma(full_timestep),
            )
        if return_logits:
            _, pred_x0, logits = output
            return pred_x0, logits
        _, pred_x0 = output
        return pred_x0, None

    def _dense_model_timestep(self, full_timestep):
        if full_timestep.ndim != 2 or full_timestep.shape[0] != 1:
            return full_timestep
        num_tokens = full_timestep.shape[1]
        patch_t, _, _ = self.generator.model.patch_size
        frame_tokens = self.frame_seq_length
        if frame_tokens <= 0 or num_tokens % frame_tokens != 0:
            return full_timestep
        patch_frames = num_tokens // frame_tokens
        grid_timestep = full_timestep.reshape(1, patch_frames, frame_tokens)
        frame_timestep = grid_timestep[:, :, :1]
        return frame_timestep.squeeze(-1).repeat_interleave(patch_t, dim=1)

    def _forward_sparse(
        self,
        *,
        latents,
        conditional_dict,
        full_timestep,
        active_indices,
        collect_indices,
        clear_kv_cache,
        token_positions,
        kv_cache,
        crossattn_cache,
        cache_start,
        return_logits=False,
        requires_grad=False,
        cache_only=False,
    ):
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        with context:
            active_timestep = full_timestep.index_select(1, active_indices)
            output = self.generator(
                noisy_image_or_video=latents,
                conditional_dict=conditional_dict,
                timestep=active_timestep,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                cache_start=cache_start,
                adaptive_forward_mode="sparse",
                rope_positions=token_positions,
                active_token_indices=active_indices,
                clear_kv_cache=clear_kv_cache,
                collect_token_indices=collect_indices,
                full_timestep=full_timestep,
                return_patch_exit_logits=return_logits,
                kv_workspace=self._kv_workspace_for_forward(kv_cache),
                cache_only=cache_only,
            )
            sparse_cache = None
            logits = None
            if collect_indices is not None and collect_indices.numel() > 0:
                if return_logits:
                    flow_pred, sparse_cache, logits = output
                else:
                    flow_pred, sparse_cache = output
            else:
                if return_logits:
                    flow_pred, logits = output
                else:
                    flow_pred = output
            pred_x0 = None
            if not cache_only:
                pred_x0 = self.generator._convert_flow_pred_to_x0_video(
                    flow_pred=flow_pred,
                    xt=latents,
                    timestep=full_timestep,
                    precomputed_sigma=self._lookup_precomputed_sigma(full_timestep),
                )
        return pred_x0, self._detach_sparse_cache(sparse_cache), logits

    def _validate_inputs(self, noisy_input):
        batch_size, current_num_frames, _, height, width = noisy_input.shape
        if batch_size != 1:
            raise ValueError("router rollout currently supports batch_size == 1")
        patch_t, patch_h, patch_w = self.generator.model.patch_size
        if current_num_frames % patch_t != 0 or height % patch_h != 0 or width % patch_w != 0:
            raise ValueError("latent shape must be divisible by Wan patch_size")
        return batch_size, current_num_frames, height, width

    def _common_state(self, noisy_input, current_start_frame):
        batch_size, current_num_frames, height, width = self._validate_inputs(noisy_input)
        device = noisy_input.device
        patch_shape = self.patch_shape(batch_size, current_num_frames, height, width)
        token_positions = self.make_patch_positions(
            batch_size, current_num_frames, height, width, device, start_frame=current_start_frame
        )
        num_tokens = token_positions.shape[1]
        all_indices = torch.arange(num_tokens, device=device, dtype=torch.long)
        timesteps = self.denoising_step_list.to(device=device, dtype=torch.float32)
        context_timestep = torch.tensor(float(self.context_noise), device=device, dtype=torch.float32)
        current_start = current_start_frame * self.frame_seq_length
        return patch_shape, token_positions, num_tokens, all_indices, timesteps, context_timestep, current_start

    @staticmethod
    def _indices(mask):
        return mask[0].nonzero(as_tuple=False).flatten().to(dtype=torch.long)

    def _lockstep_active_indices(self, active_indices, fallback_indices):
        """Keep optional sparse stages collective-safe across FSDP ranks.

        Different data-parallel samples may have no local token for a stage even
        though another rank must enter the FSDP-wrapped backbone.  Such ranks run
        one ignored dummy token so every rank performs the same module forwards.
        """

        local_has_active = bool(active_indices.numel() > 0)
        if not bool(getattr(self, "distributed_lockstep", False)) or not dist.is_initialized():
            return active_indices, local_has_active, local_has_active
        global_has_active = torch.tensor(
            int(local_has_active), device=fallback_indices.device, dtype=torch.int32
        )
        dist.all_reduce(global_has_active, op=dist.ReduceOp.MAX)
        should_run = bool(global_has_active.item())
        if not should_run or local_has_active:
            return active_indices, local_has_active, should_run
        if fallback_indices.numel() == 0:
            raise RuntimeError("distributed sparse lockstep requires a dummy token")
        return fallback_indices[:1], False, True

    @staticmethod
    def _snapshot_chunk_cache_state(kv_cache):
        workspace = kv_cache[0].get("_router_workspace")
        workspace_state = None
        if workspace is not None:
            workspace_state = {
                "chunk_tokens": workspace.chunk_tokens,
                "local_start": workspace.local_start,
                "committed_tokens": workspace.committed_tokens,
                "piece_indices": workspace.piece_indices,
                "global_end": workspace.global_end,
                "local_end": workspace.local_end,
                "direct_write": workspace.direct_write,
            }
        return {
            "cursors": [
                (int(cache["global_end_index"].item()), int(cache["local_end_index"].item()))
                for cache in kv_cache
            ],
            "host_positions": [
                ("_host_position" in cache, cache.get("_host_position"))
                for cache in kv_cache
            ],
            "workspace": workspace,
            "workspace_state": workspace_state,
        }

    @staticmethod
    def _restore_chunk_cache_state(kv_cache, snapshot):
        for cache, (global_end, local_end), (had_host_position, host_position) in zip(
            kv_cache,
            snapshot["cursors"],
            snapshot["host_positions"],
        ):
            cache["global_end_index"].fill_(global_end)
            cache["local_end_index"].fill_(local_end)
            if had_host_position:
                cache["_host_position"] = host_position
            else:
                cache.pop("_host_position", None)

        workspace = snapshot["workspace"]
        if workspace is None:
            kv_cache[0].pop("_router_workspace", None)
            return

        kv_cache[0]["_router_workspace"] = workspace
        workspace_state = snapshot["workspace_state"]
        workspace.chunk_tokens = workspace_state["chunk_tokens"]
        workspace.local_start = workspace_state["local_start"]
        workspace.committed_tokens = workspace_state["committed_tokens"]
        workspace.piece_indices = workspace_state["piece_indices"]
        workspace.global_end = workspace_state["global_end"]
        workspace.local_end = workspace_state["local_end"]
        workspace.direct_write = workspace_state["direct_write"]

    @staticmethod
    def _validate_speculative_cache_capacity(kv_cache, local_end, num_tokens):
        cache_capacity = int(kv_cache[0]["k"].shape[1])
        if int(local_end) + int(num_tokens) > cache_capacity:
            raise RuntimeError(
                "non-committing router cache transaction cannot rollover; "
                f"local_end={local_end}, chunk_tokens={num_tokens}, capacity={cache_capacity}"
            )

    def _prepare_chunk_cache(self, kv_cache, num_tokens, cache_start, *, allow_rollover=None):
        position = kv_cache[0].get("_host_position", (int(cache_start), int(cache_start)))
        global_end, local_end = position
        if global_end != int(cache_start):
            raise RuntimeError(
                f"host KV cursor global_end={global_end}, expected cache_start={cache_start}"
            )
        speculative_transaction = allow_rollover is False
        if speculative_transaction:
            self._validate_speculative_cache_capacity(kv_cache, local_end, num_tokens)
        workspace = kv_cache[0].get("_router_workspace")
        if (
            workspace is None
            or workspace.capacity < num_tokens
            or workspace.layers[0][0] is not kv_cache[0]["k"]
        ):
            workspace = ChunkKVWorkspace.allocate(kv_cache, max_chunk_tokens=num_tokens)
            kv_cache[0]["_router_workspace"] = workspace
        workspace.global_end, workspace.local_end = position
        local_attn_size = int(getattr(self.generator.model, "local_attn_size", -1))
        sink_frames = int(getattr(self.generator.model.blocks[0].self_attn, "sink_size", 0))
        configured_rollover = local_attn_size != -1
        if allow_rollover is None:
            allow_rollover = configured_rollover
        else:
            allow_rollover = bool(allow_rollover) and configured_rollover
        workspace.local_end = workspace.reset(
            chunk_tokens=num_tokens,
            local_start=local_end,
            allow_rollover=allow_rollover,
            sink_tokens=sink_frames * self.frame_seq_length,
        )
        for cache in kv_cache:
            cache["_host_position"] = (global_end, workspace.local_end)
            if workspace.local_end != local_end:
                cache["local_end_index"].fill_(workspace.local_end)
        workspace.direct_write = True

    def _write_final_cache(self, kv_cache, num_tokens, cache_start):
        workspace = self._kv_workspace_for_forward(kv_cache)
        if workspace is None:
            raise RuntimeError("router direct-write cache state is not initialized")
        local_end = workspace.local_end + num_tokens
        global_end = cache_start + num_tokens
        workspace.commit(global_end=global_end, local_end=local_end)
        workspace.global_end, workspace.local_end = global_end, local_end
        workspace.direct_write = False
        for cache in kv_cache:
            cache["_host_position"] = (global_end, local_end)

    def run_chunk_fixed_steps(
        self,
        *,
        noisy_input,
        conditional_dict,
        current_start_frame: int,
        kv_cache,
        crossattn_cache,
        steps_map,
        write_cache: bool = True,
    ) -> RouterChunkResult:
        """Run a fixed routing map, transactionally when cache writes are disabled."""

        if write_cache:
            return self._run_chunk_fixed_steps_impl(
                noisy_input=noisy_input,
                conditional_dict=conditional_dict,
                current_start_frame=current_start_frame,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                steps_map=steps_map,
                write_cache=True,
            )

        snapshot = self._snapshot_chunk_cache_state(kv_cache)
        try:
            return self._run_chunk_fixed_steps_impl(
                noisy_input=noisy_input,
                conditional_dict=conditional_dict,
                current_start_frame=current_start_frame,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                steps_map=steps_map,
                write_cache=False,
            )
        finally:
            self._restore_chunk_cache_state(kv_cache, snapshot)

    def _run_chunk_fixed_steps_impl(
        self,
        *,
        noisy_input,
        conditional_dict,
        current_start_frame: int,
        kv_cache,
        crossattn_cache,
        steps_map,
        write_cache: bool = True,
    ) -> RouterChunkResult:
        patch_shape, token_positions, num_tokens, all_indices, timesteps, context_timestep, current_start = (
            self._common_state(noisy_input, current_start_frame)
        )
        device = noisy_input.device
        steps_map = torch.as_tensor(steps_map, device=device, dtype=torch.long).reshape(patch_shape)
        steps_flat = steps_map.reshape(1, -1)
        if bool(((steps_flat < 1) | (steps_flat > self.num_steps)).any()):
            raise ValueError(
                f"steps_map values must lie in [1, {self.num_steps}], "
                f"got min={int(steps_flat.min())} max={int(steps_flat.max())}"
            )
        cache_start = current_start
        if write_cache:
            self._prepare_chunk_cache(kv_cache, num_tokens, cache_start)
        latents = noisy_input
        cache_pieces = []
        seq_len_log = []

        def full_timestep(fill_value):
            return torch.full((1, num_tokens), float(fill_value), device=device, dtype=torch.float32)

        def mixed_timestep(default_value, clear_steps):
            values = full_timestep(default_value)
            for exit_step in clear_steps:
                clear = self._indices(steps_flat == exit_step)
                if clear.numel() > 0:
                    values[:, clear] = context_timestep
            return values

        steps = range(1, self.num_steps + 1)
        masks = {step: (steps_flat == step) for step in steps}
        indices = {step: self._indices(masks[step]) for step in steps}

        pred_x0, _ = self._forward_dense(
            latents=latents,
            conditional_dict=conditional_dict,
            full_timestep=full_timestep(timesteps[0]),
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            cache_start=cache_start,
        )
        latents = self.merge_active_patches(latents, pred_x0.detach(), masks[1])
        latents = self.merge_active_patches(
            latents,
            self.add_transition_noise(pred_x0.detach(), full_timestep(timesteps[1]), transition_index=0).detach(),
            steps_flat >= 2,
        )
        seq_len_log.append(
            {
                "stage": 0,
                "mode": "dense",
                "timestep": float(timesteps[0]),
                "seq_len": int(num_tokens),
            }
        )

        for stage_index in range(1, self.num_steps):
            previous_exit_step = stage_index
            current_exit_step = stage_index + 1
            active = self._indices(steps_flat >= previous_exit_step)
            run_active, local_has_active, should_run = self._lockstep_active_indices(
                active, all_indices
            )
            if not should_run:
                continue
            collect = (
                indices[previous_exit_step]
                if local_has_active and indices[previous_exit_step].numel() > 0
                else None
            )
            pred_x0, cache, _ = self._forward_sparse(
                latents=latents,
                conditional_dict=conditional_dict,
                full_timestep=mixed_timestep(
                    timesteps[stage_index], range(1, previous_exit_step + 1)
                ),
                active_indices=run_active,
                collect_indices=collect,
                clear_kv_cache=(
                    None
                    if stage_index == 1
                    else self._prefix_cache(cache_pieces, kv_cache)
                ),
                token_positions=token_positions,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                cache_start=cache_start,
            )
            if local_has_active and cache is not None:
                cache_pieces.append(cache)
            if local_has_active:
                latents = self.merge_active_patches(
                    latents, pred_x0.detach(), masks[current_exit_step]
                )
                if current_exit_step < self.num_steps:
                    latents = self.merge_active_patches(
                        latents,
                        self.add_transition_noise(
                            pred_x0.detach(), full_timestep(timesteps[stage_index + 1]), transition_index=stage_index
                        ).detach(),
                        steps_flat >= current_exit_step + 1,
                    )
            seq_len_log.append(
                {
                    "stage": int(stage_index),
                    "mode": "mixed_sparse",
                    "timestep": float(timesteps[stage_index]),
                    "seq_len": int(active.numel()),
                }
            )

        active = indices[self.num_steps]
        run_active, local_has_active, should_run = self._lockstep_active_indices(
            active, all_indices
        )
        cache_clear_forwards = 0
        cache_clear_tokens = 0
        if should_run:
            _, cache, _ = self._forward_sparse(
                latents=latents,
                conditional_dict=conditional_dict,
                full_timestep=mixed_timestep(
                    context_timestep, range(1, self.num_steps + 1)
                ),
                active_indices=run_active,
                collect_indices=active if local_has_active else None,
                clear_kv_cache=self._prefix_cache(cache_pieces, kv_cache),
                token_positions=token_positions,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                cache_start=cache_start,
                cache_only=True,
            )
            if local_has_active and cache is not None:
                cache_pieces.append(cache)
            cache_clear_forwards = 1
            cache_clear_tokens = int(active.numel())
            seq_len_log.append(
                {
                    "stage": int(self.num_steps),
                    "mode": "sparse@clear",
                    "timestep": float(context_timestep),
                    "seq_len": int(active.numel()),
                }
            )

        if write_cache:
            self._write_final_cache(kv_cache, num_tokens, cache_start)

        counts = {step: int(indices[step].numel()) for step in steps}
        router_nfe = sum(step * count for step, count in counts.items())
        dense_reference_nfe = self.num_steps * num_tokens
        avg_step = router_nfe / float(num_tokens)
        chunk_log = {
            "max_steps": int(self.num_steps),
            "patch_counts": counts,
            "patch_avg_step": float(avg_step),
            "router_nfe": int(router_nfe),
            "dense_reference_nfe": int(dense_reference_nfe),
            "nfe_ratio": float(router_nfe / float(dense_reference_nfe)),
            "speedup": (
                float(dense_reference_nfe / float(router_nfe)) if router_nfe else 1.0
            ),
            "cache_clear_forwards": int(cache_clear_forwards),
            "cache_clear_tokens": int(cache_clear_tokens),
        }
        if self.num_steps == 4:
            chunk_log["full4_nfe"] = int(dense_reference_nfe)
        return RouterChunkResult(
            latents=latents,
            steps_map=steps_map.detach(),
            seq_len_log=seq_len_log,
            chunk_log=chunk_log,
        )

    def run_router(
        self,
        *,
        noisy_input,
        conditional_dict,
        current_start_frame: int,
        kv_cache,
        crossattn_cache,
        chunk_index: int,
        force_max_steps: bool = False,
        commit_cache: bool = True,
    ) -> RouterChunkResult:
        """Run one routed chunk, optionally rolling back all effective KV-cache state."""
        del chunk_index
        common_state = self._common_state(noisy_input, current_start_frame)
        patch_shape, _, num_tokens, _, _, _, current_start = common_state
        if force_max_steps:
            steps_map = torch.full(
                patch_shape,
                self.num_steps,
                device=noisy_input.device,
                dtype=torch.long,
            )
            fixed_step_kwargs = {
                "noisy_input": noisy_input,
                "conditional_dict": conditional_dict,
                "current_start_frame": current_start_frame,
                "kv_cache": kv_cache,
                "crossattn_cache": crossattn_cache,
                "steps_map": steps_map,
                "write_cache": True,
            }
            if commit_cache:
                result = self.run_chunk_fixed_steps(**fixed_step_kwargs)
            else:
                snapshot = self._snapshot_chunk_cache_state(kv_cache)
                position = kv_cache[0].get("_host_position", (int(current_start), int(current_start)))
                self._validate_speculative_cache_capacity(kv_cache, position[1], num_tokens)
                try:
                    result = self.run_chunk_fixed_steps(**fixed_step_kwargs)
                finally:
                    self._restore_chunk_cache_state(kv_cache, snapshot)
            result.chunk_log.update(
                {"forced": True, "selected_step": int(self.num_steps)}
            )
            return result

        cache_start = current_start
        if commit_cache:
            self._prepare_chunk_cache(kv_cache, num_tokens, cache_start)
            result = self._run_router_prepared(
                noisy_input=noisy_input,
                conditional_dict=conditional_dict,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                common_state=common_state,
            )
            self._write_final_cache(kv_cache, num_tokens, cache_start)
            return result

        snapshot = self._snapshot_chunk_cache_state(kv_cache)
        try:
            self._prepare_chunk_cache(
                kv_cache,
                num_tokens,
                cache_start,
                allow_rollover=False,
            )
            return self._run_router_prepared(
                noisy_input=noisy_input,
                conditional_dict=conditional_dict,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                common_state=common_state,
            )
        finally:
            self._restore_chunk_cache_state(kv_cache, snapshot)

    def _run_router_prepared(
        self,
        *,
        noisy_input,
        conditional_dict,
        kv_cache,
        crossattn_cache,
        common_state,
    ) -> RouterChunkResult:
        patch_shape, token_positions, num_tokens, all_indices, timesteps, context_timestep, current_start = common_state
        device = noisy_input.device
        assigned = torch.zeros((1, num_tokens), device=device, dtype=torch.long)
        latents = noisy_input
        seq_len_log = []
        exit_counts = {}
        cache_start = current_start

        def full_timestep(fill_value):
            return torch.full((1, num_tokens), float(fill_value), device=device, dtype=torch.float32)

        def mixed_timestep(default_value, clear_steps):
            values = full_timestep(default_value)
            for step in clear_steps:
                clear = self._indices(assigned == step)
                if clear.numel() > 0:
                    values[:, clear] = context_timestep
            return values

        def decide(logits, active_indices, step):
            probs = torch.sigmoid(logits.float())
            active_unassigned = assigned[:, active_indices] == 0
            selected = (probs > self.exit_threshold) & active_unassigned
            if selected.ndim == 2:
                selected = selected[0]
            exit_indices = active_indices[selected]
            exit_counts[step] = int(exit_indices.numel())
            if exit_indices.numel() > 0:
                assigned[:, exit_indices] = step
            return probs, exit_indices

        pred_x0, logits = self._forward_dense(
            latents=latents,
            conditional_dict=conditional_dict,
            full_timestep=full_timestep(timesteps[0]),
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            cache_start=cache_start,
            return_logits=True,
        )
        _, exit_indices = decide(logits, all_indices, 1)
        latents = self.merge_active_patches(latents, pred_x0.detach(), assigned == 1)
        latents = self.merge_active_patches(
            latents,
            self.add_transition_noise(pred_x0.detach(), full_timestep(timesteps[1]), transition_index=0).detach(),
            assigned == 0,
        )
        seq_len_log.append(
            {
                "stage": 0,
                "mode": "dense",
                "timestep": float(timesteps[0]),
                "seq_len": int(num_tokens),
            }
        )

        previous_exit_indices = exit_indices
        for stage_index in range(1, self.num_steps - 1):
            previous_exit_step = stage_index
            current_exit_step = stage_index + 1
            active = self._indices(
                (assigned == 0) | (assigned == previous_exit_step)
            )
            run_active, local_has_active, should_run = self._lockstep_active_indices(
                active, all_indices
            )
            if not should_run:
                previous_exit_indices = active
                continue
            collect = (
                previous_exit_indices
                if local_has_active and previous_exit_indices.numel() > 0
                else None
            )
            pred_x0, _, logits = self._forward_sparse(
                latents=latents,
                conditional_dict=conditional_dict,
                full_timestep=mixed_timestep(
                    timesteps[stage_index], range(1, previous_exit_step + 1)
                ),
                active_indices=run_active,
                collect_indices=collect,
                clear_kv_cache=None,
                token_positions=token_positions,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                cache_start=cache_start,
                return_logits=True,
            )
            if local_has_active:
                _, exit_indices = decide(logits, active, current_exit_step)
                latents = self.merge_active_patches(
                    latents, pred_x0.detach(), assigned == current_exit_step
                )
                latents = self.merge_active_patches(
                    latents,
                    self.add_transition_noise(
                        pred_x0.detach(), full_timestep(timesteps[stage_index + 1]), transition_index=stage_index
                    ).detach(),
                    assigned == 0,
                )
                previous_exit_indices = exit_indices
            else:
                previous_exit_indices = active
            seq_len_log.append(
                {
                    "stage": int(stage_index),
                    "mode": "mixed_sparse",
                    "timestep": float(timesteps[stage_index]),
                    "seq_len": int(active.numel()),
                }
            )

        assigned[assigned == 0] = self.num_steps
        exit_counts[self.num_steps] = num_tokens - sum(
            exit_counts.get(step, 0) for step in range(1, self.num_steps)
        )
        active = self._indices(
            (assigned == self.num_steps - 1) | (assigned == self.num_steps)
        )
        run_active, local_has_active, should_run = self._lockstep_active_indices(
            active, all_indices
        )
        if should_run:
            collect = (
                self._indices(assigned == self.num_steps - 1)
                if local_has_active
                else active
            )
            pred_x0, _, _ = self._forward_sparse(
                latents=latents,
                conditional_dict=conditional_dict,
                full_timestep=mixed_timestep(
                    timesteps[self.num_steps - 1], range(1, self.num_steps)
                ),
                active_indices=run_active,
                collect_indices=collect if collect.numel() > 0 else None,
                clear_kv_cache=None,
                token_positions=token_positions,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                cache_start=cache_start,
            )
            if local_has_active:
                latents = self.merge_active_patches(
                    latents, pred_x0.detach(), assigned == self.num_steps
                )
            seq_len_log.append(
                {
                    "stage": int(self.num_steps - 1),
                    "mode": "mixed_sparse",
                    "timestep": float(timesteps[self.num_steps - 1]),
                    "seq_len": int(active.numel()),
                }
            )

        active = self._indices(assigned == self.num_steps)
        run_active, local_has_active, should_run = self._lockstep_active_indices(
            active, all_indices
        )
        cache_clear_forwards = 0
        cache_clear_tokens = 0
        if should_run:
            self._forward_sparse(
                latents=latents,
                conditional_dict=conditional_dict,
                full_timestep=mixed_timestep(
                    context_timestep, range(1, self.num_steps + 1)
                ),
                active_indices=run_active,
                collect_indices=active if local_has_active else None,
                clear_kv_cache=None,
                token_positions=token_positions,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                cache_start=cache_start,
                cache_only=True,
            )
            cache_clear_forwards = 1
            cache_clear_tokens = int(active.numel())
            seq_len_log.append(
                {
                    "stage": int(self.num_steps),
                    "mode": "sparse@clear",
                    "timestep": float(context_timestep),
                    "seq_len": int(active.numel()),
                }
            )

        steps_map = assigned.reshape(patch_shape)
        counts = {
            step: int(exit_counts.get(step, 0))
            for step in range(1, self.num_steps + 1)
        }
        router_nfe = sum(step * count for step, count in counts.items())
        dense_reference_nfe = self.num_steps * num_tokens
        avg_step = router_nfe / float(num_tokens)
        selected_step = int((router_nfe + num_tokens - 1) // num_tokens)
        chunk_log = {
            "max_steps": int(self.num_steps),
            "patch_counts": counts,
            "patch_avg_step": float(avg_step),
            "router_nfe": int(router_nfe),
            "dense_reference_nfe": int(dense_reference_nfe),
            "nfe_ratio": float(router_nfe / float(dense_reference_nfe)),
            "speedup": (
                float(dense_reference_nfe / float(router_nfe)) if router_nfe else 1.0
            ),
            "cache_clear_forwards": int(cache_clear_forwards),
            "cache_clear_tokens": int(cache_clear_tokens),
            "forced": False,
            "selected_step": selected_step,
        }
        if self.num_steps == 4:
            chunk_log["full4_nfe"] = int(dense_reference_nfe)
        return RouterChunkResult(
            latents=latents,
            steps_map=steps_map.detach(),
            seq_len_log=seq_len_log,
            chunk_log=chunk_log,
        )

    def logits_at_step(
        self,
        *,
        noisy_input,
        conditional_dict,
        current_start_frame: int,
        kv_cache,
        crossattn_cache,
        steps_map,
        selected_step: int,
    ):
        patch_shape, token_positions, num_tokens, all_indices, timesteps, context_timestep, current_start = (
            self._common_state(noisy_input, current_start_frame)
        )
        device = noisy_input.device
        steps_map = torch.as_tensor(steps_map, device=device, dtype=torch.long).reshape(patch_shape)
        steps_flat = steps_map.reshape(1, -1)
        if bool(((steps_flat < 1) | (steps_flat > self.num_steps)).any()):
            raise ValueError(f"steps_map values must lie in [1, {self.num_steps}]")
        if not 1 <= int(selected_step) < self.num_steps:
            raise ValueError(
                f"selected_step must lie in [1, {self.num_steps - 1}], got {selected_step}"
            )
        latents = noisy_input
        cache_pieces = []
        cache_start = current_start

        def full_timestep(fill_value):
            return torch.full((1, num_tokens), float(fill_value), device=device, dtype=torch.float32)

        def mixed_timestep(default_value, clear_steps):
            values = full_timestep(default_value)
            for exit_step in clear_steps:
                clear = self._indices(steps_flat == exit_step)
                if clear.numel() > 0:
                    values[:, clear] = context_timestep
            return values

        if selected_step == 1:
            _, logits = self._forward_dense(
                latents=latents,
                conditional_dict=conditional_dict,
                full_timestep=full_timestep(timesteps[0]),
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                return_logits=True,
                requires_grad=True,
            )
            active = all_indices
            return logits.index_select(1, active), active

        pred_x0, _ = self._forward_dense(
            latents=latents,
            conditional_dict=conditional_dict,
            full_timestep=full_timestep(timesteps[0]),
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            cache_start=cache_start,
        )
        latents = self.merge_active_patches(latents, pred_x0.detach(), steps_flat == 1)
        latents = self.merge_active_patches(
            latents,
            self.add_transition_noise(pred_x0.detach(), full_timestep(timesteps[1]), transition_index=0).detach(),
            steps_flat >= 2,
        )

        for stage_index in range(1, int(selected_step)):
            previous_exit_step = stage_index
            current_exit_step = stage_index + 1
            active_previous = self._indices(steps_flat >= previous_exit_step)
            run_active, local_has_active, should_run = self._lockstep_active_indices(
                active_previous, all_indices
            )
            active_target = self._indices(steps_flat >= current_exit_step)
            if not should_run:
                empty = latents.new_empty((latents.shape[0], 0, 1))
                return empty, active_target
            target_stage = current_exit_step == int(selected_step)
            collect = self._indices(steps_flat == previous_exit_step)
            pred_x0, cache, logits = self._forward_sparse(
                latents=latents,
                conditional_dict=conditional_dict,
                full_timestep=mixed_timestep(
                    timesteps[stage_index], range(1, previous_exit_step + 1)
                ),
                active_indices=run_active,
                collect_indices=(
                    collect if local_has_active and collect.numel() > 0 else None
                ),
                clear_kv_cache=(
                    None
                    if stage_index == 1
                    else self._prefix_cache(cache_pieces, kv_cache)
                ),
                token_positions=token_positions,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                cache_start=cache_start,
                return_logits=target_stage,
                requires_grad=target_stage,
            )
            if target_stage:
                if not local_has_active:
                    return logits[:, :0], active_target
                selected_positions = torch.searchsorted(active_previous, active_target)
                return logits.index_select(1, selected_positions), active_target

            if local_has_active and cache is not None:
                cache_pieces.append(cache)
            if local_has_active:
                latents = self.merge_active_patches(
                    latents, pred_x0.detach(), steps_flat == current_exit_step
                )
                latents = self.merge_active_patches(
                    latents,
                    self.add_transition_noise(
                        pred_x0.detach(), full_timestep(timesteps[stage_index + 1]), transition_index=stage_index
                    ).detach(),
                    steps_flat >= current_exit_step + 1,
                )

        raise RuntimeError("router logits replay did not reach the selected step")
