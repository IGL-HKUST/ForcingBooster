"""Local frozen Stage2 loading, SP8 execution and offline conditioning."""
import gc
import json
import os
import random
import numpy as np
import torch
import torch.distributed as dist
from .config import path
from .upstream import activate


def model_config(cfg):
    return dict(checkpoint_path=str(path(cfg.base_path)), torch_dtype="bfloat16",
                transformer_device_map=None, low_cpu_mem_usage=True, local_files_only=True,
                attention_backend="flash")


def initialize(cfg):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(2)
    activate(path(cfg.upstream_root))
    if int(os.environ.get("WORLD_SIZE", 1)) != 8:
        raise RuntimeError("Launch infer with torchrun --nproc_per_node=8")
    from solarwm.backends.minimax_h3.fsdp import initialize_distributed
    rank, world = initialize_distributed(sp_size=8, local_rank=int(os.environ["LOCAL_RANK"]))
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    return rank, torch.device("cuda", int(os.environ["LOCAL_RANK"]))


def load_backbone(cfg):
    from solarwm.backends.minimax_h3.optional import load_transformer
    from solarwm.backends.minimax_h3.lora import inject_h3_lora
    from solarwm.backends.minimax_h3.weights import load_initial_weights
    from solarwm.backends.minimax_h3.fsdp import wrap_h3_fsdp
    from .backbone import H3Backbone
    # Upstream low_cpu_mem_usage loads through meta, materializing shards on CPU.
    # FSDP then moves/wraps individual modules, never the full 33B model on a GPU.
    modules = load_transformer(model_config(cfg), device=torch.device("cpu"))
    model, lora = inject_h3_lora(modules.transformer, dict(rank=384, alpha=384),
                               base_identity={})
    weights = load_initial_weights(dict(path=str(path(cfg.stage2_path)),
                                       stage="stage2", weight_source="ema"), lora)
    model.eval().requires_grad_(False)
    if hasattr(model, "disable_gradient_checkpointing"):
        model.disable_gradient_checkpointing()
    backbone = H3Backbone(model)
    local = int(os.environ["LOCAL_RANK"])
    for parameter in lora.parameters:
        parameter.data = parameter.data.to(torch.device("cuda", local))
    backbone.wrap(wrap_h3_fsdp(
        model, local_rank=local, transformer_block_cls=modules.transformer_block_cls,
        fp32_units=modules.fp32_fsdp_units, ignored_parameters=lora.parameters,
        activation_checkpointing=False, frozen_base_shard_size=8))
    del modules, lora, model
    gc.collect()
    if dist.get_rank() == 0:
        print(json.dumps(dict(event="model_loaded", source=weights)), flush=True)
    return backbone


def rank_zero_call(function):
    """Propagate rank-zero filesystem/decoder failures to all waiting SP peers."""
    result = [None]
    if dist.get_rank() == 0:
        try:
            result[0] = (function(), None)
        except Exception as exc:
            result[0] = (None, f"{type(exc).__name__}: {exc}")
    dist.broadcast_object_list(result, src=0)
    value, error = result[0]
    if error:
        raise RuntimeError(error)
    return value


def build_inputs(cfg, sample, device, seed):
    from safetensors.torch import load_file
    from solarwm.backends.minimax_h3.layout import patchify_video
    from solarwm.backends.minimax_h3.sgf_rollout import H3SGFInputs
    from solarwm.backends.minimax_h3.torch_flow import sample_shifted_timestep, scale_noise
    from solarwm.backends.minimax_h3.distributed import broadcast_sp_tensor
    from .data import load_condition
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
