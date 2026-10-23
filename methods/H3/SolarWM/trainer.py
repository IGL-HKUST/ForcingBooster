"""Router training: frozen SP8 Stage2 backbone, replicated FP32 head, resumable bundles."""
import hashlib
import json
import os
import time
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from ..shared.config import path
from ..shared.dist import rank_zero_call
from .bundle import (SCHEMA, epoch_order, load_bundle, restore_rng, rng_state, save_checkpoint,
                     validate_resume)
from .router.cache import RawKVCache
from .router.model import RouterHead
from .router.objective import continue_cost, h3_router_contract, router_objective
from .router.rollout import RouterRollout
from .router.training import target_branches
from .runtime import build_inputs, initialize, load_backbone


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def train_samples(cfg):
    """The training split in manifest order; the epoch order indexes this list."""
    samples = json.loads(path(cfg.train.manifest).read_text())["samples"]
    samples = [s for s in samples if s.get("split") == cfg.train.split]
    if not samples:
        raise ValueError(f"No H3 training samples in split {cfg.train.split!r}")
    return samples


def weights_identity(cfg):
    stage2 = path(cfg.stage2_path)
    complete = hashlib.sha256((stage2 / "COMPLETE.json").read_bytes()).hexdigest()
    step = int(json.loads((stage2 / "checkpoint-manifest.json").read_text())["step"])
    # load_initial_weights names its Stage2 EMA source the same way.
    return dict(weights=digest(dict(base=os.path.realpath(cfg.base_path), stage2=complete)),
                source=f"stage2:ema:step={step}")


def frozen_fingerprint(model):
    """sha256 over this rank's local frozen tensors: its FSDP base shard and replicated LoRA."""
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        h.update(f"{name}:{parameter.dtype}:{tuple(parameter.shape)};".encode())
        h.update(parameter.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy())
    return h.hexdigest()


def assert_replicated(tensors):
    flat = torch.cat([x.reshape(-1) for x in tensors])
    high, low = flat.clone(), flat.clone()
    dist.all_reduce(high, op=dist.ReduceOp.MAX)
    dist.all_reduce(low, op=dist.ReduceOp.MIN)
    same = torch.tensor(int(torch.equal(high, flat) and torch.equal(low, flat)), device=flat.device)
    dist.all_reduce(same, op=dist.ReduceOp.MIN)
    if not same.item():
        raise RuntimeError("Router head gradients are non-finite or differ across SP ranks")


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_cpu(v) for v in value)
    return value


def attempt_branches(cfg, rollout, sample, device, *, attempts, target_chunk, selected_step,
                     record, event=None):
    """History chunks 0..j-1 committed as in router inference (chunk 0 at four steps, then the
    current head), then the uncommitted target-chunk branches. Noise follows the attempt index."""
    from .backbone.layout import patchify_video
    inputs, noise, rng = build_inputs(cfg, sample, device, int(cfg.seed) + 7919 * attempts)
    cache = RawKVCache()
    for index in range(target_chunk):
        n = noise[:, :, 5 * index:5 * index + 5]
        transitions = rollout.transitions(n, rng, inputs)
        rows = patchify_video(n)
        forced = torch.full((rows.shape[1],), 4, device=device, dtype=torch.long) if index == 0 else None
        rollout.chunk(inputs, rows, index=index, cache=cache, transition_noise=transitions,
                      forced_steps=forced)
        if event:
            event(dict(event="history_chunk", chunk=index))
    n = noise[:, :, 5 * target_chunk:5 * target_chunk + 5]
    transitions = rollout.transitions(n, rng, inputs)
    rows = patchify_video(n)
    branch = target_branches(rollout, inputs, rows, index=target_chunk, cache=cache,
                             transition_noise=transitions, selected_step=selected_step, record=record)
    cache.clear()
    return branch, rows.shape[1]


def train(cfg):
    started = time.perf_counter()
    t = cfg.train
    output = path(cfg.output_dir)
    if (output / "train.jsonl").exists() and not t.resume:
        raise FileExistsError(f"Existing training run requires train.resume or a new output_dir: {output}")
    rank, device = initialize(cfg, require_world=8)
    world = dist.get_world_size()
    samples = train_samples(cfg)
    seed, count = int(cfg.seed), len(samples)
    contract = h3_router_contract(t.nfe_weight, cfg.kv_commit)
    identity, data_identity = weights_identity(cfg), digest(samples)
    experiment = OmegaConf.to_container(t.experiment, resolve=True)

    def event(value):
        if rank == 0:
            print(json.dumps(value), flush=True)

    def setup():
        (output / "checks").mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output / "resolved_config.yaml")
    rank_zero_call(setup)
    cfg.resident_weights = False  # FSDP: the trainer also needs room for its branches
    backbone = load_backbone(cfg)
    frozen = frozen_fingerprint(backbone.model) if t.audit_frozen else None
    head = RouterHead(hidden_dim=int(cfg.head_dim), initial_exit_bias=float(t.initial_exit_bias))
    updates = attempts = cursor = epoch = 0
    order = epoch_order(seed, epoch, count)
    bundle = None
    if cfg.resume_path:
        bundle = load_bundle(cfg.resume_path)
        validate_resume(bundle, contract=contract, head_dim=cfg.head_dim, learning_rate=t.learning_rate,
                        exit_threshold=cfg.exit_threshold, seed=seed, num_train=count)
        for key, value in dict(identity=identity, data_identity=data_identity).items():
            if bundle[key] != value:
                raise ValueError(f"Cannot resume H3 router bundle: {key} {bundle[key]} differs from {value}")
        if len(bundle["rank_rng"]) != world:
            raise ValueError(f"Cannot resume H3 router bundle of {len(bundle['rank_rng'])} ranks on {world}")
    elif cfg.init_head_path:
        bundle = load_bundle(cfg.init_head_path)
    if bundle is not None:
        head.load_state_dict(bundle["head"], strict=True)
    head.to(device)
    for parameter in head.parameters():
        dist.broadcast(parameter.data, src=0)
    params = list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=float(t.learning_rate), betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=0)
    if cfg.resume_path:
        optimizer.load_state_dict(bundle["optimizer"])
        updates, attempts, cursor, epoch = (int(bundle[k]) for k in ("updates", "attempts", "cursor", "epoch"))
        order = list(bundle["order"])
        restore_rng(bundle["rank_rng"][rank])
    del bundle
    rollout = RouterRollout(backbone, head, threshold=float(cfg.exit_threshold), kv_commit=cfg.kv_commit)
    record = t.continue_cost == "wan"
    targets, steps = [int(j) for j in t.target_chunks], [int(s) for s in t.selected_steps]
    save_steps, max_updates = {int(u) for u in t.save_steps}, int(t.max_updates)
    first, saved, checkpoint = updates, None, None
    log = output / "train.jsonl"
    load_seconds = time.perf_counter() - started
    event(dict(event="model_loaded", identity=identity["weights"], source=identity["source"]))

    def due(u):
        return u % int(t.save_every) == 0 or u in save_steps

    def save():
        states = [None] * world
        dist.all_gather_object(states, rng_state())

        def write():
            payload = dict(
                schema=SCHEMA, head={k: v.detach().to("cpu", copy=True) for k, v in head.state_dict().items()},
                optimizer=to_cpu(optimizer.state_dict()), identity=identity, data_identity=data_identity,
                contract=contract, head_dim=int(cfg.head_dim), updates=updates, attempts=attempts,
                cursor=cursor, order=list(order), epoch=epoch, rank_rng=states,
                learning_rate=float(t.learning_rate), exit_threshold=float(cfg.exit_threshold),
                experiment=experiment)
            target = str(save_checkpoint(output, payload, step=updates,
                                         resolved_config=OmegaConf.to_yaml(cfg, resolve=True)))
            event(dict(event="checkpoint_saved", updates=updates, attempts=attempts, checkpoint=target))
            return target
        return rank_zero_call(write)

    def append(row):
        with log.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    while updates < max_updates:
        if attempts >= int(t.max_attempts):
            raise RuntimeError(f"Exceeded train.max_attempts={t.max_attempts} before {max_updates} updates")
        sample = samples[order[cursor]]
        cursor += 1
        if cursor == count:
            epoch, cursor = epoch + 1, 0
            order = epoch_order(seed, epoch, count)
        attempts += 1
        g = torch.Generator().manual_seed(seed * 1_000_003 + attempts)
        j = targets[int(torch.randint(len(targets), (1,), generator=g))]
        s = steps[int(torch.randint(len(steps), (1,), generator=g))]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        attempt_started = time.perf_counter()
        event(dict(event="attempt_start", attempts=attempts, updates=updates, sample=sample["sample_id"],
                   target_chunk=j, selected_step=s))
        branch, total = attempt_branches(cfg, rollout, sample, device, attempts=attempts, target_chunk=j,
                                         selected_step=s, record=record, event=event)
        active = branch.active_ids.numel()
        row = dict(loss=None, mse=None, nfe=None, weighted_nfe=None, active_count=active,
                   active_ratio=active / total, skip=not active, gradient_norm=None)
        if active:
            logits = head(branch.hidden).reshape(-1)
            exit_mse = (branch.x0_exit.float() - branch.x0_ref.float()).square().mean(-1).reshape(-1)
            exits = branch.exits[branch.active_ids] if record else torch.full_like(branch.active_ids, 4)
            loss, metrics = router_objective(logits=logits, exit_mse=exit_mse,
                                             continue_steps=continue_cost(exits, s, t.continue_cost),
                                             selected_step=s, nfe_weight=float(t.nfe_weight))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite router loss at attempt {attempts}")
            loss.backward()
            grads = [p.grad for p in params]
            # Every rank sees the same all-gathered features; rank 0's gradient is applied everywhere.
            if updates == first or due(updates + 1) or updates + 1 == max_updates:
                assert_replicated(grads)
            for grad in grads:
                dist.broadcast(grad, src=0)
            if t.max_grad_norm is None:
                norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
            else:
                norm = torch.nn.utils.clip_grad_norm_(params, float(t.max_grad_norm))
            if not torch.isfinite(norm):
                raise FloatingPointError(f"Non-finite router gradient norm at attempt {attempts}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            row.update({k: metrics[k] for k in ("loss", "mse", "nfe", "weighted_nfe", "active_count")},
                       gradient_norm=float(norm))
            del logits, exit_mse, exits, loss, grads
        del branch
        torch.cuda.synchronize()
        row.update(updates=updates, attempts=attempts, sample=sample["sample_id"], target_chunk=j,
                   selected_step=s, seconds=time.perf_counter() - attempt_started,
                   normalization="active_patches",
                   max_allocated_gib=torch.cuda.max_memory_allocated(device) / 2 ** 30,
                   max_reserved_gib=torch.cuda.max_memory_reserved(device) / 2 ** 30,
                   load_seconds=load_seconds, experiment=experiment)
        rank_zero_call(lambda: append(row))
        if active and due(updates):
            checkpoint, saved = save(), updates
    if updates != first and saved != updates:
        checkpoint, saved = save(), updates
    if t.audit_frozen:
        ranks = [None] * world
        dist.all_gather_object(ranks, (frozen, frozen_fingerprint(backbone.model)))
        report = dict(passed=all(a == b for a, b in ranks), updates=updates,
                      ranks=[dict(rank=r, before=a, after=b, unchanged=a == b) for r, (a, b) in enumerate(ranks)])
        rank_zero_call(lambda: (output / "checks" / f"frozen_update_{updates:06d}.json").write_text(
            json.dumps(report, indent=2)))
        if not report["passed"]:
            raise RuntimeError("Frozen H3 backbone parameters changed during router training")
    summary = dict(updates=updates, attempts=attempts, skips=attempts - updates, checkpoint=checkpoint)
    rank_zero_call(lambda: (output / "training_summary.json").write_text(json.dumps(summary, indent=2)))
    backbone.hook.remove()
    dist.barrier()
    dist.destroy_process_group()
    return dict(updates=updates, attempts=attempts, checkpoint=checkpoint) if rank == 0 else None
