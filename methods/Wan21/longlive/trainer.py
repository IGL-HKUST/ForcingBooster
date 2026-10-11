"""Single-device router training with full reproducible resume state."""
import json
import random
import time
import uuid

import numpy as np
import torch
from omegaconf import OmegaConf

from methods.Wan21.shared.router_objective import (
    router_training_contract, validate_router_resume, validate_router_training_config)

from .config import resolve_path
from .model import LongLiveRouterModel
from .weights import atomic_save


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["cuda"])


def save_training(model, optimizer, cfg, updates, attempts, cursor, path, prompts):
    metadata = dict(model.longlive_metadata, router_trained=True, loss="latent_mse_plus_uniform_nfe",
                    nfe_weight=float(cfg.nfe_weight), head_dim=int(cfg.head_dim), training_frames=21)
    bundle = {"generator": {k: v.detach().cpu() for k, v in model.generator.state_dict().items()},
              "metadata": metadata, "training": {"optimizer": optimizer.state_dict(), "updates": updates,
              "attempts": attempts, "prompt_cursor": cursor, "prompts": prompts, "rng": rng_state(),
              "config": OmegaConf.to_container(cfg, resolve=True),
              "router_training_contract": router_training_contract(cfg.nfe_weight)}}
    state_steps = [int(value["step"]) for value in optimizer.state.values()]
    if not state_steps or any(step != updates for step in state_steps):
        raise AssertionError("Optimizer update count does not match checkpoint step")
    if any(not torch.isfinite(p).all() for p in model.router_head.parameters()):
        raise FloatingPointError("Nonfinite router checkpoint")
    atomic_save(bundle, path)


def train(cfg):
    validate_router_training_config(cfg)
    if cfg.weight_mode != "merged":
        raise ValueError("Router training uses the accepted merged generator")
    device = torch.device(cfg.device)
    torch.cuda.set_device(device)
    torch.manual_seed(int(cfg.seed))
    torch.cuda.manual_seed_all(int(cfg.seed))
    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    directory = resolve_path(cfg.output_dir)
    if (directory / "metrics.jsonl").exists() and not cfg.resume:
        raise FileExistsError(f"Existing training run requires resume or a new output_dir: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    model = LongLiveRouterModel(cfg, device)
    # Backbone loaded in BF16; keep the freshly loaded router in FP32 on resume.
    model.generator.to(device=device).eval()
    model.router_head.float()
    model.text_encoder.to(device=device, dtype=torch.bfloat16).eval()
    model.scheduler.timesteps = model.scheduler.timesteps.to(device)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("patch_exit_router_head" not in name for name, _ in trainable):
        raise AssertionError("Only router parameters may be trainable")
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=float(cfg.learning_rate), betas=(0.9, 0.999), weight_decay=0)
    prompts = [str(p) for p in cfg.get("train_prompts", cfg.prompts)]
    if cfg.prompt_file:
        prompts = [p.strip() for p in resolve_path(cfg.prompt_file).read_text().splitlines() if p.strip()]
    if not prompts:
        raise ValueError("Empty training prompt set")
    encoding = str(cfg.get("text_encoding", "precompute"))
    if encoding not in ("online", "precompute"):
        raise ValueError("text_encoding must be online or precompute")
    conditions = None
    if encoding == "precompute":
        if len(prompts) > 64:
            raise ValueError("Large prompt sets require text_encoding=online")
        with torch.no_grad():
            conditions = [model.text_encoder(text_prompts=[p]) for p in prompts]
        model.text_encoder.cpu()
    torch.cuda.empty_cache()
    updates = attempts = cursor = 0
    if cfg.resume:
        bundle = torch.load(resolve_path(cfg.resume), map_location="cpu", weights_only=False)
        state = bundle.get("training", {})
        validate_router_resume(state.get("router_training_contract"), cfg.nfe_weight)
        previous = state["config"]
        for key in ("height", "width", "head_dim", "nfe_weight", "learning_rate", "train_chunk_choices", "exit_threshold"):
            if previous[key] != OmegaConf.to_container(cfg, resolve=True)[key]:
                raise ValueError(f"Resume contract changed: {key}")
        if state["prompts"] != prompts:
            raise ValueError("Resume prompt order changed")
        if not 0 <= int(state["updates"]) <= int(state["attempts"]):
            raise ValueError("Invalid router checkpoint update/attempt counts")
        optimizer.load_state_dict(state["optimizer"])
        updates, attempts, cursor = state["updates"], state["attempts"], state["prompt_cursor"]
        # A checkpointed FP32 head must survive reconstruction bit-for-bit.
        for name, value in model.generator.state_dict().items():
            if "patch_exit_router_head" in name and not torch.equal(value.cpu(), bundle["generator"][name]):
                raise AssertionError(f"Router precision changed during resume: {name}")
        restore_rng(state["rng"])
        del bundle, state
    segment = uuid.uuid4().hex
    with (directory / "segments.jsonl").open("a") as stream:
        stream.write(json.dumps({"segment": segment, "resume_updates": updates, "resume_attempts": attempts,
                                 "prompt_cursor": cursor, "encoding": encoding})+"\n")
    OmegaConf.save(cfg, directory / "config.yaml")
    torch.cuda.reset_peak_memory_stats()
    while updates < int(cfg.max_updates):
        if attempts >= int(cfg.max_attempts):
            raise RuntimeError("Exceeded attempt budget before reaching requested optimizer updates")
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            conditional = (model.text_encoder(text_prompts=[prompts[cursor % len(prompts)]])
                           if conditions is None else conditions[cursor % len(conditions)])
        torch.cuda.synchronize()
        encoding_seconds = time.perf_counter()-start
        cursor += 1
        attempts += 1
        noise = torch.randn((1, 21, 16, int(cfg.height)//8, int(cfg.width)//8), device=device, dtype=torch.bfloat16)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = model.router_loss(noise=noise, conditional_dict=conditional)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite router loss")
        skipped = bool(metrics.get("router_skip_iter", torch.tensor(0)))
        if not skipped:
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.router_head.parameters(), 1.0)
            if not torch.isfinite(grad) or float(grad) == 0:
                raise FloatingPointError(f"Invalid router gradient: {grad}")
            optimizer.step()
            updates += 1
        torch.cuda.synchronize()
        record = {k: (v.item() if isinstance(v, torch.Tensor) and v.numel()==1 else v) for k, v in metrics.items()}
        record.update(updates=updates, attempts=attempts, seconds=time.perf_counter()-start,
                      encoding_seconds=encoding_seconds, segment=segment, prompt_index=(cursor-1)%len(prompts))
        with (directory / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record)+"\n")
        print(f"update={updates} attempt={attempts} loss={float(loss):.6f} seconds={record['seconds']:.2f}", flush=True)
        if not skipped and (updates % int(cfg.save_every) == 0 or updates == int(cfg.max_updates)):
            path = directory / f"checkpoint_{updates:06d}.pt"
            save_training(model, optimizer, cfg, updates, attempts, cursor, path, prompts)
    report = {"updates": updates, "attempts": attempts, "only_router_trainable": True,
              "loss": "latent_mse_plus_uniform_nfe", "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
              "resumed_from": str(cfg.resume) if cfg.resume else None,
              "checkpoint": str(directory / f"checkpoint_{updates:06d}.pt")}
    (directory / "training_report.json").write_text(json.dumps(report, indent=2))
    return report
