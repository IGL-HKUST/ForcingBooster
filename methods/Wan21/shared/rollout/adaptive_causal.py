from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


@dataclass
class AdaptiveChunkResult:
    latents: torch.Tensor
    steps_map: torch.Tensor
    gradient_mask: Optional[torch.Tensor]
    denoised_timestep_from: Optional[int]
    denoised_timestep_to: Optional[int]
    seq_len_log: list[dict]


class AdaptiveCausalRollout:
    def __init__(
        self,
        *,
        generator,
        scheduler,
        denoising_step_list,
        frame_seq_length: int = 0,
        context_noise: int = 0,
        exit_step_weights: Optional[list[float]] = None,
        steps_map_path: Optional[str] = None,
        first_chunk_exit_step: Optional[int] = None,
        ensure_all_exit_steps: bool = False,
        verbose: bool = False,
    ) -> None:
        self.generator = generator
        self.scheduler = scheduler
        self.denoising_step_list = torch.as_tensor(denoising_step_list, dtype=torch.float32)
        if len(self.denoising_step_list) != 4:
            raise ValueError(
                "adaptive causal rollout only supports a 4-step schedule; "
                f"got {self.denoising_step_list.tolist()}"
            )
        self.frame_seq_length = int(frame_seq_length)
        self.context_noise = int(context_noise)
        self.exit_step_weights = exit_step_weights or [0.1, 0.2, 0.3, 0.4]
        if len(self.exit_step_weights) != 4:
            raise ValueError("adaptive_exit_step_weights must contain 4 values for exits 1/2/3/4")
        if any(float(weight) <= 0 for weight in self.exit_step_weights):
            raise ValueError("adaptive_exit_step_weights must be positive")
        self.steps_map_path = steps_map_path
        if first_chunk_exit_step is not None and int(first_chunk_exit_step) not in {1, 2, 3, 4}:
            raise ValueError("adaptive_first_chunk_exit_step must be one of {1,2,3,4}")
        self.first_chunk_exit_step = None if first_chunk_exit_step is None else int(first_chunk_exit_step)
        self._loaded_steps_map = None
        self.ensure_all_exit_steps = bool(ensure_all_exit_steps)
        self.verbose = bool(verbose)

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

    def add_patch_noise(self, pred_x0, next_timestep_tokens):
        expanded_timestep = self.patch_values_to_latent(next_timestep_tokens, pred_x0)
        return self.scheduler.add_patch_noise(pred_x0, torch.randn_like(pred_x0), expanded_timestep)

    def merge_active_patches(self, current_latents, candidate_latents, active_mask):
        active = self.patch_values_to_latent(active_mask.to(dtype=current_latents.dtype), current_latents).bool()
        return torch.where(active, candidate_latents, current_latents)

    def _load_steps_map(self):
        if self.steps_map_path is None:
            return None
        if self._loaded_steps_map is not None:
            return self._loaded_steps_map
        path = self.steps_map_path
        if path.endswith(".pt") or path.endswith(".pth"):
            data = torch.load(path, map_location="cpu")
        elif path.endswith(".json"):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported adaptive_steps_map_path suffix: {path}")
        if isinstance(data, dict):
            if "steps_map" in data:
                data = data["steps_map"]
            elif "patchwise_steps_map" in data:
                data = data["patchwise_steps_map"]
        self._loaded_steps_map = torch.as_tensor(data, dtype=torch.long)
        return self._loaded_steps_map

    def _sample_steps_map(self, shape, device):
        weights = torch.tensor(self.exit_step_weights, device=device, dtype=torch.float32)
        probs = weights / weights.sum()
        flat = torch.multinomial(probs, int(torch.tensor(shape).prod().item()), replacement=True) + 1
        if self.ensure_all_exit_steps and flat.numel() >= 4:
            flat[:4] = torch.arange(1, 5, device=device, dtype=flat.dtype)
            flat = flat[torch.randperm(flat.numel(), device=device)]
        elif not bool((flat == 4).any()):
            flat[0] = 4
        return flat.reshape(shape).to(dtype=torch.long)

    def _select_loaded_steps_map(self, shape, device, chunk_index):
        loaded = self._load_steps_map()
        if loaded is None:
            return None
        tensor = loaded
        if tensor.ndim == len(shape) + 1:
            tensor = tensor[int(chunk_index) % tensor.shape[0]]
        if tensor.ndim == 2 and tensor.shape == torch.Size([shape[0], int(torch.tensor(shape[1:]).prod().item())]):
            tensor = tensor.reshape(shape)
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"Loaded adaptive steps map shape {tuple(tensor.shape)} does not match {tuple(shape)}")
        return tensor.to(device=device, dtype=torch.long)

    def make_steps_map(self, shape, device, chunk_index, steps_map=None):
        if steps_map is None:
            steps_map = self._select_loaded_steps_map(shape, device, chunk_index)
        if steps_map is None and self.first_chunk_exit_step is not None and int(chunk_index) == 0:
            steps_map = torch.full(shape, self.first_chunk_exit_step, device=device, dtype=torch.long)
        if steps_map is None:
            steps_map = self._sample_steps_map(shape, device)
        steps_map = torch.as_tensor(steps_map, device=device, dtype=torch.long)
        if tuple(steps_map.shape) != tuple(shape):
            if steps_map.ndim == 2 and steps_map.shape == torch.Size([shape[0], int(torch.tensor(shape[1:]).prod().item())]):
                steps_map = steps_map.reshape(shape)
            else:
                raise ValueError(f"adaptive steps map shape {tuple(steps_map.shape)} does not match {tuple(shape)}")
        valid_values = torch.isin(steps_map, torch.tensor([1, 2, 3, 4], device=device))
        if not bool(valid_values.all()):
            bad = torch.unique(steps_map[~valid_values]).detach().cpu().tolist()
            raise ValueError(f"adaptive steps map contains values outside {{1,2,3,4}}: {bad}")
        return steps_map

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
        num_blocks = len(cache_pieces[0])
        for block_index in range(num_blocks):
            merged.append(
                {
                    "k": torch.cat([piece[block_index]["k"] for piece in cache_pieces], dim=1),
                    "v": torch.cat([piece[block_index]["v"] for piece in cache_pieces], dim=1),
                    "indices": torch.cat([piece[block_index]["indices"] for piece in cache_pieces], dim=0),
                }
            )
        return merged

    def _metadata_timestep(self, timestep_value, device):
        timestep_float = float(torch.as_tensor(timestep_value).detach().float().cpu().item())
        if timestep_float == 0:
            return 0
        timesteps = self.scheduler.timesteps.to(device=device)
        value = torch.as_tensor(timestep_float, device=device, dtype=timesteps.dtype)
        return int(1000 - torch.argmin((timesteps - value).abs(), dim=0).item())

    def _selected_timestep_range(self, exit_step: Optional[int], device):
        if exit_step is None:
            return None, None
        from_value = self.denoising_step_list[exit_step - 1].to(device=device)
        if exit_step < len(self.denoising_step_list):
            to_value = self.denoising_step_list[exit_step].to(device=device)
        else:
            to_value = 0
        return self._metadata_timestep(from_value, device), self._metadata_timestep(to_value, device)

    @staticmethod
    def _is_rank0() -> bool:
        return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

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
        requires_grad,
    ):
        model_timestep = self._dense_model_timestep(full_timestep)
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        with context:
            _, pred_x0 = self.generator(
                noisy_image_or_video=latents,
                conditional_dict=conditional_dict,
                timestep=model_timestep,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                write_cache=False,
            )
        return pred_x0

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
        if bool(torch.equal(grid_timestep, frame_timestep.expand_as(grid_timestep))):
            return frame_timestep.squeeze(-1).repeat_interleave(patch_t, dim=1)
        return full_timestep

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
        requires_grad,
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
            )
            if collect_indices is not None and collect_indices.numel() > 0:
                flow_pred, sparse_cache = output
            else:
                flow_pred = output
                sparse_cache = None
            pred_x0 = self.generator._convert_flow_pred_to_x0_video(
                flow_pred=flow_pred,
                xt=latents,
                timestep=full_timestep,
            )
        return pred_x0, self._detach_sparse_cache(sparse_cache)

    def _update_group(self, latents, pred_x0, group_mask, keep_grad):
        if not bool(group_mask.any()):
            return latents
        candidate = pred_x0 if keep_grad else pred_x0.detach()
        return self.merge_active_patches(latents, candidate, group_mask)

    def _update_group_with_noise(self, latents, pred_x0, group_mask, next_timestep_tokens, keep_grad):
        if not bool(group_mask.any()):
            return latents
        source = pred_x0 if keep_grad else pred_x0.detach()
        candidate = self.add_patch_noise(source, next_timestep_tokens)
        if not keep_grad:
            candidate = candidate.detach()
        return self.merge_active_patches(latents, candidate, group_mask)

    def _choose_exit_step(self, requested_exit_step):
        if requested_exit_step is None:
            return None
        requested_exit_step = int(requested_exit_step)
        if requested_exit_step not in {1, 2, 3, 4}:
            raise ValueError(f"adaptive selected exit step must be one of {{1,2,3,4}}, got {requested_exit_step}")
        return requested_exit_step

    def run_chunk(
        self,
        *,
        noisy_input,
        conditional_dict,
        current_start_frame: int,
        current_num_frames: int,
        kv_cache,
        crossattn_cache,
        chunk_index: int = 0,
        steps_map=None,
        grad_exit_step: Optional[int] = None,
        enable_truncated_grad: bool = False,
        force_uniform_exit_step: bool = False,
        full_gradient_mask: bool = False,
    ) -> AdaptiveChunkResult:
        batch_size, _, _, height, width = noisy_input.shape
        if batch_size != 1:
            raise ValueError("adaptive causal rollout currently supports batch_size == 1")

        device = noisy_input.device
        dtype = torch.float32
        patch_t, patch_h, patch_w = self.generator.model.patch_size
        if current_num_frames % patch_t != 0 or height % patch_h != 0 or width % patch_w != 0:
            raise ValueError("latent shape must be divisible by Wan patch_size for adaptive rollout")
        patch_shape = (batch_size, current_num_frames // patch_t, height // patch_h, width // patch_w)
        selected_exit_step = self._choose_exit_step(grad_exit_step)
        if force_uniform_exit_step:
            if selected_exit_step is None:
                raise ValueError("force_uniform_exit_step requires grad_exit_step")
            steps_map = torch.full(patch_shape, selected_exit_step, device=device, dtype=torch.long)
        else:
            steps_map = self.make_steps_map(
                patch_shape,
                device,
                chunk_index,
                steps_map=steps_map,
            )
        if selected_exit_step is not None and not force_uniform_exit_step:
            threshold = torch.full_like(steps_map, selected_exit_step)
            steps_map = torch.minimum(steps_map, threshold)
        steps_flat = steps_map.reshape(batch_size, -1)

        token_positions = self.make_patch_positions(
            batch_size, current_num_frames, height, width, device, start_frame=current_start_frame
        )
        num_tokens = token_positions.shape[1]
        all_indices = torch.arange(num_tokens, device=device, dtype=torch.long)
        masks = {exit_step: (steps_flat == exit_step) for exit_step in (1, 2, 3, 4)}
        indices = {
            exit_step: masks[exit_step][0].nonzero(as_tuple=False).flatten().to(dtype=torch.long)
            for exit_step in (1, 2, 3, 4)
        }
        timesteps = self.denoising_step_list.to(device=device, dtype=dtype)
        context_timestep = torch.tensor(float(self.context_noise), device=device, dtype=dtype)
        current_start = current_start_frame * self.frame_seq_length
        cache_start = current_start
        latents = noisy_input
        cache_pieces = []
        seq_len_log = []

        def full_timestep(fill_value):
            return torch.full((batch_size, num_tokens), float(fill_value), device=device, dtype=dtype)

        def mixed_timestep(default_value, clear_steps):
            values = full_timestep(default_value)
            for exit_step in clear_steps:
                if indices[exit_step].numel() > 0:
                    values[:, indices[exit_step]] = context_timestep
            return values

        def active_from(min_exit_step):
            active_mask = steps_flat >= min_exit_step
            return active_mask[0].nonzero(as_tuple=False).flatten().to(dtype=torch.long)

        selected_mask_flat = torch.zeros_like(steps_flat, dtype=torch.bool)
        if enable_truncated_grad and selected_exit_step is not None:
            if full_gradient_mask:
                selected_mask_flat = torch.ones_like(steps_flat, dtype=torch.bool)
            else:
                selected_mask_flat = masks[selected_exit_step]

        # Stage 0: full chunk at 1000; tokens assigned to exit 1 become clear.
        pred_x0 = self._forward_dense(
            latents=latents,
            conditional_dict=conditional_dict,
            full_timestep=full_timestep(timesteps[0]),
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            cache_start=cache_start,
            requires_grad=enable_truncated_grad and selected_exit_step == 1,
        )
        latents = self._update_group(
            latents,
            pred_x0,
            masks[1],
            keep_grad=enable_truncated_grad and selected_exit_step == 1,
        )
        continue_mask = steps_flat >= 2
        latents = self._update_group_with_noise(
            latents,
            pred_x0,
            continue_mask,
            full_timestep(timesteps[1]),
            keep_grad=False,
        )
        seq_len_log.append({"stage": 0, "mode": "full@1000", "seq_len": int(num_tokens)})

        # Stage 1: full-length mixed forward, collecting exit-1 clear KV.
        should_run = bool((steps_flat >= 2).any() or masks[1].any())
        if should_run:
            collect = indices[1] if indices[1].numel() > 0 else None
            pred_x0, cache = self._forward_sparse(
                latents=latents,
                conditional_dict=conditional_dict,
                full_timestep=mixed_timestep(timesteps[1], clear_steps=[1]),
                active_indices=all_indices,
                collect_indices=collect,
                clear_kv_cache=None,
                token_positions=token_positions,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                cache_start=cache_start,
                requires_grad=enable_truncated_grad and selected_exit_step == 2,
            )
            if cache is not None:
                cache_pieces.append(cache)
            latents = self._update_group(
                latents,
                pred_x0,
                masks[2],
                keep_grad=enable_truncated_grad and selected_exit_step == 2,
            )
            latents = self._update_group_with_noise(
                latents,
                pred_x0,
                steps_flat >= 3,
                full_timestep(timesteps[2]),
                keep_grad=False,
            )
            seq_len_log.append({"stage": 1, "mode": "full@mixed", "seq_len": int(num_tokens)})

        # Stage 2: sparse active exits 2/3/4 at 500, collecting exit-2 clear KV.
        should_run = bool((steps_flat >= 3).any() or masks[2].any())
        if should_run:
            active = active_from(2)
            collect = indices[2] if indices[2].numel() > 0 else None
            pred_x0, cache = self._forward_sparse(
                latents=latents,
                conditional_dict=conditional_dict,
                full_timestep=mixed_timestep(timesteps[2], clear_steps=[1, 2]),
                active_indices=active,
                collect_indices=collect,
                clear_kv_cache=self._merge_sparse_caches(cache_pieces),
                token_positions=token_positions,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                cache_start=cache_start,
                requires_grad=enable_truncated_grad and selected_exit_step == 3,
            )
            if cache is not None:
                cache_pieces.append(cache)
            latents = self._update_group(
                latents,
                pred_x0,
                masks[3],
                keep_grad=enable_truncated_grad and selected_exit_step == 3,
            )
            latents = self._update_group_with_noise(
                latents,
                pred_x0,
                steps_flat >= 4,
                full_timestep(timesteps[3]),
                keep_grad=False,
            )
            seq_len_log.append({"stage": 2, "mode": "sparse@500", "seq_len": int(active.numel())})

        # Stage 3: sparse active exits 3/4 at 250, collecting exit-3 clear KV.
        should_run = bool((steps_flat >= 4).any() or masks[3].any())
        if should_run:
            active = active_from(3)
            collect = indices[3] if indices[3].numel() > 0 else None
            pred_x0, cache = self._forward_sparse(
                latents=latents,
                conditional_dict=conditional_dict,
                full_timestep=mixed_timestep(timesteps[3], clear_steps=[1, 2, 3]),
                active_indices=active,
                collect_indices=collect,
                clear_kv_cache=self._merge_sparse_caches(cache_pieces),
                token_positions=token_positions,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                cache_start=cache_start,
                requires_grad=enable_truncated_grad and selected_exit_step == 4,
            )
            if cache is not None:
                cache_pieces.append(cache)
            latents = self._update_group(
                latents,
                pred_x0,
                masks[4],
                keep_grad=enable_truncated_grad and selected_exit_step == 4,
            )
            seq_len_log.append({"stage": 3, "mode": "sparse@250", "seq_len": int(active.numel())})

        # Stage 4: final clear handoff for exit-4 tokens.
        if bool(masks[4].any()):
            active = indices[4]
            _, cache = self._forward_sparse(
                latents=latents,
                conditional_dict=conditional_dict,
                full_timestep=mixed_timestep(context_timestep, clear_steps=[1, 2, 3, 4]),
                active_indices=active,
                collect_indices=active,
                clear_kv_cache=self._merge_sparse_caches(cache_pieces),
                token_positions=token_positions,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                cache_start=cache_start,
                requires_grad=False,
            )
            cache_pieces.append(cache)
            seq_len_log.append({"stage": 4, "mode": "sparse@clear", "seq_len": int(active.numel())})

        final_cache = self._merge_sparse_caches(cache_pieces)
        if final_cache is None:
            raise RuntimeError("adaptive rollout did not collect any KV cache for the current chunk")
        collected = torch.cat([piece[0]["indices"] for piece in cache_pieces], dim=0)
        if collected.numel() != num_tokens or torch.unique(collected).numel() != num_tokens:
            raise RuntimeError(
                f"adaptive cache compose expected {num_tokens} unique tokens, got "
                f"{collected.numel()} entries and {torch.unique(collected).numel()} unique entries"
            )
        self.generator.write_sparse_chunk_cache(
            kv_cache=kv_cache,
            sparse_kv_cache=final_cache,
            chunk_tokens=num_tokens,
            cache_start=cache_start,
        )

        gradient_mask = None
        if enable_truncated_grad and selected_exit_step is not None and bool(selected_mask_flat.any()):
            gradient_mask = self.patch_values_to_latent(selected_mask_flat.to(dtype=latents.dtype), latents).bool()
            gradient_mask = gradient_mask.expand_as(latents)
        timestep_from, timestep_to = self._selected_timestep_range(selected_exit_step, device)

        if self.verbose and self._is_rank0():
            counts = {exit_step: int(indices[exit_step].numel()) for exit_step in (1, 2, 3, 4)}
            print(
                "adaptive causal chunk: "
                f"tokens={num_tokens}, counts={counts}, selected_exit={selected_exit_step}, stages={seq_len_log}",
                flush=True,
            )

        return AdaptiveChunkResult(
            latents=latents,
            steps_map=steps_map.detach().cpu(),
            gradient_mask=gradient_mask,
            denoised_timestep_from=timestep_from,
            denoised_timestep_to=timestep_to,
            seq_len_log=seq_len_log,
        )
