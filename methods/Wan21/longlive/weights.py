"""Official base + LoRA loading, isolated from other Wan deployments."""
import json
import os
from pathlib import Path

import torch

from . import UPSTREAM_COMMIT
from .config import resolve_path


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def build_generator(cfg, adaptive=True, dtype=torch.bfloat16):
    from methods.Wan21.shared.adapter.wan_wrapper import WanDiffusionWrapper
    generator = WanDiffusionWrapper(
        model_path=cfg.assets_path, model_name="Wan2.1-T2V-1.3B",
        timestep_shift=5.0, is_causal=True, local_attn_size=12, sink_size=3,
        causal_model_variant="adaptive" if adaptive else "dense", load_dtype=dtype,
    )
    generator.model.num_frame_per_block = 3
    return generator


def apply_official_weights(generator, cfg):
    import peft
    from methods.Wan21.shared.checkpoint.legacy import load_generator_checkpoint
    base_path, lora_path = resolve_path(cfg.base_path), resolve_path(cfg.lora_path)
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    load_generator_checkpoint(generator, base)
    del base
    targets = [f"blocks.{i}.{name}" for i, block in enumerate(generator.model.blocks)
               for name, module in block.named_modules() if isinstance(module, torch.nn.Linear)]
    if not targets or any("router" in key for key in targets):
        raise RuntimeError("Invalid LoRA target set")
    generator.model = peft.get_peft_model(generator.model, peft.LoraConfig(
        r=256, lora_alpha=256, lora_dropout=0.0, target_modules=targets))
    raw = torch.load(lora_path, map_location="cpu", weights_only=False)
    state = raw.get("generator_lora", raw)
    expected = peft.get_peft_model_state_dict(generator.model)
    if set(state) != set(expected):
        raise ValueError(f"LoRA keys differ: missing={sorted(set(expected)-set(state))[:8]}, "
                         f"unexpected={sorted(set(state)-set(expected))[:8]}")
    for key in expected:
        if state[key].shape != expected[key].shape:
            raise ValueError(f"LoRA shape mismatch for {key}")
    peft.set_peft_model_state_dict(generator.model, state)
    generator.requires_grad_(False).eval()
    return targets


def load_merged(generator, path, *, allow_router=False):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("metadata", {}).get("lora_merged") is not True:
        raise ValueError("Expected a prepared LongLive checkpoint with lora_merged=true")
    state = bundle["generator"]
    missing, unexpected = generator.load_state_dict(state, strict=False)
    permitted = ("model.patch_exit_router_head.",) if allow_router else ()
    if unexpected or any(not key.startswith(permitted) for key in missing):
        raise ValueError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return bundle.get("metadata", {})


@torch.no_grad()
def prepare(cfg):
    """Merge the official base and LoRA in FP32 for training and inference."""
    destination = resolve_path(cfg.merged_checkpoint)
    if destination.exists():
        raise FileExistsError(f"Prepared file already exists; inspect/reuse it: {destination}")
    generator = build_generator(cfg, adaptive=True, dtype=torch.float32).to(cfg.device)
    targets = apply_official_weights(generator, cfg)
    generator.to(device=cfg.device, dtype=torch.float32).eval()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        generator.model = generator.model.merge_and_unload(safe_merge=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    metadata = {
        "model_id": "longlive-1.3b", "upstream_commit": UPSTREAM_COMMIT,
        "lora_merged": True, "lora_rank": 256, "lora_alpha": 256,
        "lora_targets": targets,
        "sampling": {"steps": [1000, 750, 500, 250], "shift": 5, "window": 12, "sink": 3},
    }
    atomic_save({"generator": {k: v.detach().cpu() for k, v in generator.state_dict().items()},
                 "metadata": metadata}, destination)
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return {"checkpoint": str(destination), "metadata": metadata}
