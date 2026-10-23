"""Local frozen Stage2 loading, single-node SP2/4/8 execution and offline conditioning."""
import gc
import json
import os
import random
import numpy as np
import torch
import torch.distributed as dist
from ..shared.config import path
from ..shared.examples import condition_dir


def model_config(cfg):
    return dict(checkpoint_path=str(path(cfg.base_path)), torch_dtype="bfloat16",
                transformer_device_map=None, low_cpu_mem_usage=True, local_files_only=True,
                attention_backend="flash")


def initialize(cfg, require_world=None):
    """Sequence parallel spans every launched GPU of one node (56 heads: SP 2, 4 or 8)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(2)
    world = int(os.environ.get("WORLD_SIZE", 1))
    if require_world and world != require_world:
        raise RuntimeError(f"Launch with torchrun --nproc_per_node={require_world}")
    if world not in (2, 4, 8) or world != int(os.environ.get("LOCAL_WORLD_SIZE", 1)):
        raise RuntimeError("Launch with torchrun --nproc_per_node in {2,4,8} on one node")
    from .backbone.fsdp import initialize_distributed
    rank, world = initialize_distributed(sp_size=world, local_rank=int(os.environ["LOCAL_RANK"]))
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    return rank, torch.device("cuda", int(os.environ["LOCAL_RANK"]))


@torch.no_grad()
def merge_lora(model, device):
    """Fold every PEFT LoRA delta into its BF16 base weight (W + s·B·A in FP32, one rounding)
    and drop the adapter layers, removing two GEMMs per adapted linear from each forward."""
    from peft.tuners.lora import Linear as LoraLinear
    for module in model.modules():
        if not isinstance(module, LoraLinear):
            continue
        name = module.active_adapters[0]
        base = module.get_base_layer().weight
        delta = (module.lora_B[name].weight.to(device, torch.float32)
                 @ module.lora_A[name].weight.to(device, torch.float32)) * module.scaling[name]
        base.data.copy_((base.data.to(device, torch.float32) + delta).to(base.dtype).cpu())
    return model.unload()


def wait_for_gpu_memory(device, gib, seconds=180):
    """Hold a real allocation so the cooperative occupier releases its memory, then wait for room."""
    import time
    hold = torch.empty(2 * 2 ** 30, dtype=torch.uint8, device=device)
    deadline = time.time() + seconds
    while torch.cuda.mem_get_info(device)[0] < gib * 2 ** 30 and time.time() < deadline:
        time.sleep(5)
    del hold
    torch.cuda.empty_cache()


def resident(cfg):
    """Keep the whole merged 33B model on every GPU (no per-forward FSDP gathers): SP8 inference."""
    mode = cfg.get("resident_weights", "auto")
    if mode == "auto":
        return bool(cfg.get("merge_lora", True)) and dist.get_world_size() == 8
    return bool(mode)


def load_backbone(cfg):
    from .backbone.optional import load_transformer
    from .backbone.lora import inject_h3_lora
    from .backbone.weights import load_initial_weights
    from .backbone.fsdp import wrap_h3_fsdp
    from .router.model import H3Backbone
    # Upstream low_cpu_mem_usage loads through meta, materializing shards on CPU.
    # FSDP then moves/wraps individual modules, never the full 33B model on a GPU.
    modules = load_transformer(model_config(cfg), device=torch.device("cpu"))
    model, lora = inject_h3_lora(modules.transformer, dict(rank=384, alpha=384),
                               base_identity={})
    weights = load_initial_weights(dict(path=str(path(cfg.stage2_path)),
                                       stage="stage2", weight_source="ema"), lora)
    local = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local)
    ignored = lora.parameters
    if cfg.get("merge_lora", True):
        model, ignored = merge_lora(model, device), ()
        fp32_units = [m for m in model.modules() if any(m is u for u in modules.fp32_fsdp_units)]
        if len(fp32_units) != len(modules.fp32_fsdp_units):
            raise RuntimeError("LoRA merge lost an FP32 FSDP unit")
    model.eval().requires_grad_(False)
    if hasattr(model, "disable_gradient_checkpointing"):
        model.disable_gradient_checkpointing()
    backbone = H3Backbone(model)
    if resident(cfg):
        wait_for_gpu_memory(device, 74)
        backbone.wrap(model.to(device))
    else:
        for parameter in ignored:
            parameter.data = parameter.data.to(device)
        backbone.wrap(wrap_h3_fsdp(
            model, local_rank=local, transformer_block_cls=modules.transformer_block_cls,
            fp32_units=modules.fp32_fsdp_units, ignored_parameters=ignored,
            activation_checkpointing=False, frozen_base_shard_size=dist.get_world_size()))
    del modules, lora, model
    gc.collect()
    if dist.get_rank() == 0:
        print(json.dumps(dict(event="model_loaded", source=weights)), flush=True)
    return backbone


def load_condition(cfg, sample):
    from safetensors.torch import load_file
    return load_file(str(condition_dir(cfg, sample) / "condition.safetensors"))


def build_inputs(cfg, sample, device, seed):
    from safetensors.torch import load_file
    from .backbone.layout import patchify_video
    from .backbone.sgf_rollout import H3SGFInputs
    from .backbone.torch_flow import sample_shifted_timestep, scale_noise
    from .backbone.distributed import broadcast_sp_tensor
    values = load_condition(cfg, sample)
    rng = torch.Generator(device=device).manual_seed(seed)

    def random_like(value):
        n = torch.randn(value.shape, device=device, dtype=torch.float32, generator=rng)
        broadcast_sp_tensor(n)
        return n

    anchor = values["anchor_latents"].to(device).float()
    anchor = scale_noise(anchor, random_like(anchor), 0.999)
    silence = load_file(str(path(cfg.silence_file)))["silence_158f"]
    silence = silence.to(device).float().unsqueeze(0).permute(0, 1, 3, 2).reshape(1, -1, 32)
    audio_t = sample_shifted_timestep(1, 3.0, device, torch.float32, generator=rng).reshape(())
    broadcast_sp_tensor(audio_t)
    audio = scale_noise(silence, random_like(silence), audio_t)
    views, intrinsics = values["viewmats"].to(device), values["K"].to(device)
    inputs = H3SGFInputs(values["prompt_embeds"].to(device).bfloat16(),
                        values["text_token_tags"].to(device), patchify_video(anchor), audio, audio_t,
                        torch.cat((views[:1], views)).unsqueeze(0),
                        torch.cat((intrinsics[:1], intrinsics)).unsqueeze(0),
                        48, 84, 0.999, True)
    noise = random_like(torch.empty(1, 24, sample["rollout_latents"], 48, 84, device=device))
    return inputs, noise, rng
