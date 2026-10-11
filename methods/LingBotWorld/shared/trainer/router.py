"""Eight-way block-FSDP trainer for the LingBot v2 patch router."""

from __future__ import annotations

import gc
import json
import time
from typing import Any, Iterator

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from methods.LingBotWorld.shared.adapter.lingbot_v2 import (
    encode_lingbot_condition,
    initialize_lingbot_frozen_models,
)
from methods.LingBotWorld.shared.backbone import LingBotV2CausalAttentionBlock
from methods.LingBotWorld.shared.checkpoint import (
    load_router_weights,
    load_native_router_bundle,
    save_final_router_bundle,
)
from methods.LingBotWorld.shared.data import (
    LingBotV2DL3DVDataset,
    enumerate_dl3dv_scenes,
    lingbot_v2_collate,
    make_lingbot_v2_distributed_sampler,
)
from methods.LingBotWorld.shared.distributed import (
    SingleSampleCoordinator,
    full_state_dict,
    initialize_distributed_process_group,
    wrap_block_fsdp,
)
from .model import LingBotRouterTrainingModel, compose_router_loss
from .noise import initial_lingbot_noise
from methods.Wan21.shared.router_objective import router_training_contract, validate_router_resume


def _router_runtime(args: Any, timesteps: torch.Tensor) -> dict[str, Any]:
    return {
        "method": "LingBotWorld",
        "family": "lingbot_v2",
        "num_train_timesteps": int(args.num_train_timesteps),
        "timestep_indices": [int(item) for item in args.timestep_indices],
        "scheduler_shift": float(args.scheduler_shift),
        "resolved_timesteps": [int(item) for item in timesteps.tolist()],
        "max_steps": len(tuple(args.timestep_indices)),
        "num_frame_per_block": int(args.num_frame_per_block),
        "first_chunk_step": int(args.first_chunk_step),
        "router_head_contract": "shared_binary_v1",
    }


def _infinite_local_batches(
    args: Any,
    *,
    rank: int,
    world_size: int,
    entries: list[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    dataset = LingBotV2DL3DVDataset(
        data_root=args.data_path,
        entries=entries,
        split="train",
        profile=str(args.dataset_profile),
        target_height=int(args.image_size[0]),
        target_width=int(args.image_size[1]),
        random_sample=True,
        seed=int(args.seed),
        max_samples=getattr(args, "dataset_max_samples", None),
        chunk_size=int(args.num_frame_per_block),
    )
    epoch = 0
    while True:
        dataset.set_epoch(epoch)
        sampler = make_lingbot_v2_distributed_sampler(
            dataset,
            rank=rank,
            world_size=world_size,
            seed=int(args.seed),
            epoch=epoch,
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=0,
            persistent_workers=False,
            drop_last=True,
            collate_fn=lingbot_v2_collate,
        )
        yield from loader
        epoch += 1


def _single_batch_metadata(raw_batch: dict[str, Any]) -> dict[str, Any]:
    raw_metadata = raw_batch.get("sample_metadata") or []
    if isinstance(raw_metadata, dict):
        return dict(raw_metadata)
    metadata_items = list(raw_metadata)
    if len(metadata_items) != 1 or not isinstance(metadata_items[0], dict):
        raise ValueError(
            "each local LingBot batch must contain exactly one sample_metadata mapping"
        )
    return dict(metadata_items[0])


def _set_collective_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _max_across_ranks(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _materialize_and_broadcast_replicated_module(
    module: torch.nn.Module,
    *,
    device: torch.device,
    is_rank0: bool,
) -> None:
    if any(item.is_meta for item in (*module.parameters(), *module.buffers())):
        if is_rank0:
            raise RuntimeError("rank zero replicated router unexpectedly contains meta tensors")
        module.to_empty(device=device)
    else:
        module.to(device=device)
    for parameter in module.parameters():
        dist.broadcast(parameter.data, src=0)
    for buffer in module.buffers():
        dist.broadcast(buffer.data, src=0)


def _all_reduce_replicated_gradients(
    parameters: tuple[torch.nn.Parameter, ...],
) -> None:
    world_size = dist.get_world_size()
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)


class LingBotRouterTrainer:
    """Train one replicated router sample per rank over a sharded backbone."""

    def __init__(self, args: Any) -> None:
        self.args = args

    def _build_model(
        self, context: Any
    ) -> tuple[Any, torch.nn.Module, Any, Any, int, int, int]:
        from methods.LingBotWorld.shared.adapter.wrapper import LingBotDiffusionWrapper

        _set_collective_seed(int(self.args.seed))
        initialization_device: torch.device | str = (
            torch.device("cpu") if context.is_rank0 else "meta"
        )
        generator = LingBotDiffusionWrapper.from_pretrained(
            self.args.transformer_dir,
            transformer_index=self.args.transformer_index,
            architecture=self.args.architecture,
            torch_dtype=torch.bfloat16 if self.args.mixed_precision else torch.float32,
            device=initialization_device,
            local_attn_size=int(self.args.local_attn_size),
            sink_size=int(self.args.sink_size),
            load_weights=context.is_rank0,
            timestep_shift=float(self.args.scheduler_shift),
            num_train_timesteps=int(self.args.num_train_timesteps),
        )
        model = LingBotRouterTrainingModel(
            self.args,
            generator=generator,
            device=context.device,
        )
        trainable_before_wrap = {
            name: parameter.numel()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if not trainable_before_wrap or any(
            "patch_exit_router_head" not in name for name in trainable_before_wrap
        ):
            raise RuntimeError(
                "only patch_exit_router_head may be trainable; got "
                + ", ".join(trainable_before_wrap)
            )
        trainable_count = sum(trainable_before_wrap.values())
        self.router_runtime = _router_runtime(self.args, model.denoising_step_list)

        resume_bundle = None
        load_error = None
        source = self.args.resume_checkpoint or getattr(self.args, "initialization_checkpoint", None)
        if source and context.is_rank0:
            try:
                if self.args.resume_checkpoint:
                    bundle = load_native_router_bundle(source)
                    if bundle.manifest.model_id != self.args.model_id:
                        raise ValueError("checkpoint model identity mismatch")
                    validate_router_resume(bundle.training_state.get("router_training_contract"),
                                           self.args.router_nfe_weight)
                    state = bundle.training_state
                    if not 0 <= int(state["step"]) <= int(state["attempted_iterations"]):
                        raise ValueError("Invalid router checkpoint update/attempt counts")
                    if not state.get("router_optimizer"):
                        raise ValueError("Missing router optimizer state")
                    resume_bundle = bundle
                    model.load_state_dict(bundle.model_state["generator"], strict=True)
                    del bundle
                else:
                    model.load_state_dict(load_router_weights(source), strict=True)
            except Exception as exc:
                load_error = f"{type(exc).__name__}: {exc}"
        errors = [load_error]
        dist.broadcast_object_list(errors, src=0, device=context.device)
        if errors[0]:
            raise ValueError(errors[0])

        router_head = model.router_head
        _materialize_and_broadcast_replicated_module(
            router_head,
            device=context.device,
            is_rank0=context.is_rank0,
        )
        fsdp_config = dict(self.args.fsdp)
        fsdp_config["ulysses_enabled"] = not bool(self.args.no_ulysses)
        wrapped, topology = wrap_block_fsdp(
            model,
            transformer_block_cls=LingBotV2CausalAttentionBlock,
            device=context.device,
            fsdp_config=fsdp_config,
            ignored_modules=(router_head,),
        )
        trainable = list(router_head.parameters())
        if not trainable:
            raise RuntimeError("replicated router head has no trainable parameters")
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(self.args.learning_rate),
            betas=(float(self.args.beta1), float(self.args.beta2)),
            weight_decay=float(self.args.weight_decay),
        )
        start_step = 0
        start_attempt = 0
        if self.args.resume_checkpoint:
            optimizer_state = (
                resume_bundle.training_state["router_optimizer"]
                if context.is_rank0 and resume_bundle is not None
                else None
            )
            payload = [optimizer_state]
            dist.broadcast_object_list(payload, src=0, device=context.device)
            optimizer.load_state_dict(payload[0])
            start_value = (
                int(resume_bundle.training_state["step"])
                if context.is_rank0 and resume_bundle is not None
                else 0
            )
            value = torch.tensor(start_value, device=context.device, dtype=torch.long)
            dist.broadcast(value, src=0)
            start_step = int(value.item())
            attempt_value = (int(resume_bundle.training_state["attempted_iterations"])
                             if context.is_rank0 else 0)
            value.fill_(attempt_value)
            dist.broadcast(value, src=0)
            start_attempt = int(value.item())
            del resume_bundle
        return (
            wrapped,
            router_head,
            optimizer,
            topology,
            trainable_count,
            start_step,
            start_attempt,
        )

    def train(self) -> dict[str, Any]:
        if bool(self.args.same_sample_across_ranks):
            raise ValueError("LingBot router requires distinct per-rank samples")
        if str(self.args.conditioning_mode) != "per_rank_local":
            raise ValueError("LingBot router requires conditioning_mode=per_rank_local")
        if not bool(self.args.final_checkpoint_only) or bool(self.args.save_initial_checkpoint):
            raise ValueError("LingBot router only supports one final checkpoint")
        context = initialize_distributed_process_group(
            backend=str(self.args.backend),
            required_world_size=int(self.args.required_world_size),
        )
        coordinator = SingleSampleCoordinator(context)
        entries = None
        enumeration_error = None
        if context.is_rank0:
            try:
                enumeration = enumerate_dl3dv_scenes(
                    self.args.data_path,
                    split_interval=int(getattr(self.args, "dataset_split_interval", 20)),
                    required_source_frames=int(
                        getattr(self.args, "dataset_required_source_frames", 115)
                    ),
                    max_scenes=getattr(self.args, "dataset_max_scenes", None),
                )
                entries = enumeration["entries"]
                print(
                    "[lingbot-router] dataset "
                    + json.dumps(enumeration["statistics"], sort_keys=True),
                    flush=True,
                )
            except Exception as exc:
                enumeration_error = f"{type(exc).__name__}: {exc}"
        coordinator.raise_if_any_rank_error(
            enumeration_error, stage="DL3DV scene enumeration"
        )
        payload = [entries]
        dist.broadcast_object_list(payload, src=0, device=context.device)
        entries = payload[0]
        batches = _infinite_local_batches(
            self.args,
            rank=context.rank,
            world_size=context.world_size,
            entries=entries,
        )
        (
            model,
            router_head,
            optimizer,
            topology,
            trainable_count,
            step,
            attempt,
        ) = self._build_model(context)
        text_encoder = None
        vae = None
        component_error = None
        try:
            text_encoder, vae = initialize_lingbot_frozen_models(
                self.args,
                device=context.device,
            )
        except Exception as exc:
            component_error = f"{type(exc).__name__}: {exc}"
        coordinator.raise_if_any_rank_error(
            component_error,
            stage="resident frozen-model initialization",
        )
        components_ready = text_encoder is not None and vae is not None
        resident_model_ranks = coordinator.all_gather_objects(
            context.rank if components_ready else None
        )
        if resident_model_ranks != list(range(context.world_size)):
            raise RuntimeError(
                "LingBot resident frozen-model audit failed: "
                f"{resident_model_ranks}"
            )
        if context.is_rank0:
            print(
                "[lingbot-router] "
                f"world_size={context.world_size} effective_batch={self.args.total_batch_size} "
                f"conditioning=per_rank_local residency=gpu "
                f"resident_model_ranks={resident_model_ranks} "
                f"router=replicated_manual_allreduce "
                f"trainable={trainable_count} "
                f"start_step={step} final_step={self.args.max_train_steps} "
                f"schedule={self.router_runtime['resolved_timesteps']} no_ulysses=true",
                flush=True,
            )

        optimizer_update_verified = step >= int(self.args.max_train_steps)
        # Preserve the former seed sequence when every attempted batch updates
        # the optimizer, but advance the draw after an all-skip attempt.
        while step < int(self.args.max_train_steps):
            torch.cuda.synchronize(context.device)
            torch.cuda.reset_peak_memory_stats(context.device)
            iteration_started = time.perf_counter()
            step_seed = int(self.args.seed) + attempt * 1009
            attempt += 1
            conditioning_started = time.perf_counter()
            conditional_dict = None
            local_metadata = None
            preparation_error = None
            try:
                assert text_encoder is not None and vae is not None
                raw_batch = next(batches)
                local_metadata = _single_batch_metadata(raw_batch)
                conditional_dict = encode_lingbot_condition(
                    raw_batch,
                    args=self.args,
                    device=context.device,
                    text_encoder=text_encoder,
                    vae=vae,
                )
            except Exception as exc:
                preparation_error = f"{type(exc).__name__}: {exc}"
            coordinator.raise_if_any_rank_error(
                preparation_error,
                stage="local batch preparation",
            )
            metadata_values = coordinator.all_gather_objects(local_metadata)
            if not all(isinstance(item, dict) for item in metadata_values):
                raise RuntimeError(
                    f"LingBot metadata gather returned invalid values: {metadata_values}"
                )
            sample_ids = [str(item.get("sequence_id")) for item in metadata_values]
            if len(set(sample_ids)) != context.world_size:
                raise RuntimeError(
                    f"per-rank batch must contain {context.world_size} distinct samples: "
                    f"{sample_ids}"
                )
            assert conditional_dict is not None
            conditioning_elapsed = _max_across_ranks(
                time.perf_counter() - conditioning_started,
                context.device,
            )
            latent_height, latent_width = map(int, self.args.latent_size)
            noise, noise_generator = initial_lingbot_noise(
                (1, int(self.args.num_training_frames), 16, latent_height, latent_width),
                device=context.device,
                compute_dtype=torch.bfloat16 if self.args.mixed_precision else torch.float32,
                seed=step_seed + context.rank * 104729,
            )
            num_chunks = int(self.args.num_training_frames) // int(self.args.num_frame_per_block)
            chunk_index, selected_step = coordinator.route_choice(
                num_chunks=num_chunks,
                chunk_min=int(self.args.router_train_chunk_min),
                chunk_max=int(self.args.router_train_chunk_max),
                step_choices=tuple(self.args.router_train_step_choices),
                seed=step_seed + 1,
            )

            torch.cuda.synchronize(context.device)
            started = time.perf_counter()
            _set_collective_seed(step_seed + 2)
            with torch.no_grad():
                targets = model(
                    "targets",
                    noise=noise,
                    conditional_dict=conditional_dict,
                    chunk_index=chunk_index,
                    selected_step=selected_step,
                    noise_generator=noise_generator,
                )

            active_flag = torch.tensor(
                [int(not targets.skip)], device=context.device, dtype=torch.int32
            )
            active_ranks = active_flag.clone()
            dist.all_reduce(active_ranks, op=dist.ReduceOp.SUM)
            if int(active_ranks.item()) == 0:
                # AdamW applies decoupled weight decay to an explicit zero
                # gradient.  A globally empty target therefore skips the whole
                # update and does not consume one of max_train_steps.
                if context.is_rank0:
                    sample_ids = [str(item.get("sequence_id")) for item in metadata_values]
                    print(
                        "LingBot router all-skip attempt "
                        f"attempt={attempt} optimizer_step={step} "
                        f"chunk_index={chunk_index} selected_step={selected_step} "
                        f"sample_ids={json.dumps(sample_ids)}",
                        flush=True,
                    )
                del targets, conditional_dict, noise
                coordinator.barrier()
                continue

            optimizer.zero_grad(set_to_none=True)
            trainable_parameters = tuple(router_head.parameters())
            verify_optimizer_update = (
                not optimizer_update_verified and int(active_ranks.item()) > 0
            )
            parameter_snapshot = (
                tuple(parameter.detach().clone() for parameter in trainable_parameters)
                if verify_optimizer_update
                else ()
            )
            _set_collective_seed(step_seed + 3)
            # This is intentionally the last generator forward before backward.
            logits, _ = model(
                "logits",
                targets=targets,
                conditional_dict=conditional_dict,
            )
            loss, metrics = compose_router_loss(
                logits=logits,
                targets=targets,
                nfe_weight=float(self.args.router_nfe_weight),
            )
            coordinator.assert_finite([loss, *metrics.values()], stage="loss")
            loss.backward()
            _all_reduce_replicated_gradients(trainable_parameters)
            coordinator.assert_finite(
                [parameter.grad for parameter in trainable_parameters],
                stage="backward",
            )
            if verify_optimizer_update:
                local_grad_present = any(
                    parameter.grad is not None and parameter.grad.numel() > 0
                    for parameter in trainable_parameters
                )
                grad_present = torch.tensor(
                    int(local_grad_present), device=context.device, dtype=torch.int32
                )
                dist.all_reduce(grad_present, op=dist.ReduceOp.MAX)
                if not bool(grad_present.item()):
                    raise RuntimeError(
                        "LingBot Router backward produced no replicated-head gradients"
                    )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, float(self.args.max_grad_norm)
            )
            coordinator.assert_finite([grad_norm], stage="gradient-clipping")
            if verify_optimizer_update and (
                not bool(torch.isfinite(grad_norm).all())
                or float(grad_norm.detach().float().item()) <= 0.0
            ):
                raise RuntimeError(
                    "LingBot Router first active backward has a non-positive gradient norm"
                )
            optimizer.step()
            if verify_optimizer_update:
                local_parameter_changed = any(
                    before.numel() > 0 and not torch.equal(before, parameter.detach())
                    for before, parameter in zip(parameter_snapshot, trainable_parameters)
                )
                parameter_changed = torch.tensor(
                    int(local_parameter_changed), device=context.device, dtype=torch.int32
                )
                dist.all_reduce(parameter_changed, op=dist.ReduceOp.MIN)
                if not bool(parameter_changed.item()):
                    raise RuntimeError(
                        "LingBot Router optimizer step did not change the head on every rank"
                    )
                optimizer_update_verified = True
            torch.cuda.synchronize(context.device)
            elapsed = _max_across_ranks(time.perf_counter() - started, context.device)
            iteration_elapsed = _max_across_ranks(
                time.perf_counter() - iteration_started,
                context.device,
            )
            allocated = _max_across_ranks(
                torch.cuda.max_memory_allocated(context.device) / (1024**2), context.device
            )
            reserved = _max_across_ranks(
                torch.cuda.max_memory_reserved(context.device) / (1024**2), context.device
            )
            metrics.update(
                router_grad_norm=grad_norm.detach(),
                router_conditioning_time_sec=loss.new_tensor(conditioning_elapsed),
                router_iteration_time_sec=loss.new_tensor(iteration_elapsed),
                router_step_time_sec=loss.new_tensor(elapsed),
                router_peak_allocated_mb=loss.new_tensor(allocated),
                router_peak_reserved_mb=loss.new_tensor(reserved),
                router_optimizer_update_verified=loss.new_tensor(
                    float(optimizer_update_verified)
                ),
            )
            for key, value in tuple(metrics.items()):
                if not torch.is_tensor(value):
                    continue
                reduced = value.detach().float().mean().to(context.device)
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                metrics[key] = reduced / context.world_size
            step += 1
            if context.is_rank0 and step % int(self.args.progress_log_iters) == 0:
                values = " ".join(
                    f"{key}={float(value.float().mean().item()):.6g}"
                    for key, value in sorted(metrics.items())
                    if torch.is_tensor(value)
                )
                sample_ids = [str(item.get("sequence_id")) for item in metadata_values]
                print(
                    f"LingBot router train step={step} "
                    f"sample_ids={json.dumps(sample_ids)} "
                    f"rank0_sample={json.dumps(metadata_values[0], sort_keys=True)} {values}",
                    flush=True,
                )
            del targets, logits, loss, metrics, conditional_dict, noise
            if step % int(self.args.gc_interval) == 0:
                gc.collect()
                torch.cuda.empty_cache()
            coordinator.barrier()

        if not optimizer_update_verified:
            raise RuntimeError("LingBot Router training never completed an active optimizer update")
        checkpoint_path = None
        if not bool(self.args.no_save):
            gathered_model = full_state_dict(model)
            gathered_optimizer = optimizer.state_dict() if context.is_rank0 else None
            if context.is_rank0:
                checkpoint_path = save_final_router_bundle(
                    output_dir=self.args.output_dir,
                    model_id=self.args.model_id,
                    step=step,
                    final_step=int(self.args.max_train_steps),
                    model_state=gathered_model,
                    optimizer_state=gathered_optimizer,
                    resolved_config=self.args.resolved_config,
                    fsdp_topology=topology.as_dict(),
                    seed=int(self.args.seed),
                    router_runtime=self.router_runtime,
                    training_contract=router_training_contract(self.args.router_nfe_weight),
                    attempted_iterations=attempt,
                    resume_checkpoint=self.args.resume_checkpoint,
                )
            coordinator.barrier()
        result = {
            "status": "complete",
            "step": step,
            "attempted_iterations": attempt,
            "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
            "trainable_parameters": trainable_count,
        }
        return result if context.is_rank0 else {"status": "worker_complete", "step": step}


__all__ = ["LingBotRouterTrainer"]
