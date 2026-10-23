"""SolarWM 158-frame conditions (LingBot examples, DL3DV windows): camera geometry and phased,
offline encoding."""
import gc
import hashlib
import json
import os
import numpy as np
import torch
from PIL import Image
from ..shared.config import ROOT, path
from ..shared.examples import (condition_dir, condition_key, lingbot_folders, read_lingbot,
                               read_manifest, sample_name, select_samples)
from .runtime import model_config


def make_samples(cfg):
    from .backbone.full_length import full_length_geometry
    geometry = full_length_geometry(158)
    return [dict(input_kind="lingbot_example", sample_id=folder.name,
                 num_frames=158, decode_latents=geometry.decode_latents,
                 rollout_latents=geometry.rollout_latents,
                 source_files={k: os.path.relpath(v, ROOT) for k, v in files.items()})
            for folder, files in lingbot_folders(cfg.examples_root)]


def read_example(sample):
    from .backbone.full_length import full_length_geometry
    image, caption, poses, intrinsics = read_lingbot(sample)
    poses = poses[:158].astype(np.float64)
    if poses.shape != (158, 4, 4):
        raise ValueError("LingBot poses must contain at least 158 camera frames")
    views = np.linalg.inv(poses) @ poses[:1]
    views[0] = np.eye(4)
    indices = list(full_length_geometry(158).camera_indices)
    fx, fy, cx, cy = intrinsics[0] / np.array([832, 480, 832, 480])
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
    image = image.resize((1344, 768), Image.Resampling.LANCZOS)
    return image, caption, dict(viewmats=torch.from_numpy(views[indices].astype(np.float32)),
                                K=K.repeat(len(indices), 1, 1))


def read_sample(cfg, sample):
    """First image, caption and camera: LingBot examples by input_kind, DL3DV windows without one."""
    if sample.get("input_kind") == "lingbot_example":
        return read_example(sample)
    if "input_kind" not in sample and "transform_path" in sample:
        from .dl3dv import read_dl3dv
        return read_dl3dv(sample, path(cfg.dl3dv_root))
    raise ValueError(f"Cannot prepare H3 sample {sample_name(sample)} (input_kind={sample.get('input_kind')!r})")


def shard_spec(value):
    """prepare_shard "i/n" (or null = 0/1) for independent single-GPU prepare processes."""
    if value is None or str(value) == "":
        return 0, 1
    index, count = (int(x) for x in str(value).split("/"))
    if not 0 <= index < count:
        raise ValueError(f"prepare_shard must be i/n with 0 <= i < n, got {value!r}")
    return index, count


def write_ready(root, sample):
    """READY.json: the sample key, its preprocessing and the condition file's sha256."""
    digest = hashlib.sha256((root / "condition.safetensors").read_bytes()).hexdigest()
    ready = dict(identity=dict(sample=condition_key(sample), name=sample_name(sample),
                               preprocessing=sample.get("preprocessing")), sha256=digest)
    (root / "READY.tmp").write_text(json.dumps(ready, indent=2) + "\n")
    (root / "READY.tmp").replace(root / "READY.json")


def encode(cfg, samples, make_silence=True):
    from safetensors.torch import save_file, load_file
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
    from .backbone.optional import load_conditioners
    from .backbone.official_codec import encode_joint_prompt_condition
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    pending = [s for s in samples if not (condition_dir(cfg, s) / "condition.safetensors").is_file()]
    if pending:
        modules = load_conditioners(model_config(cfg), device=device, qwen=True,
                                    video_vae=False, audio_vae=False, schedulers=False)
        for sample in pending:
            image, caption, _ = read_sample(cfg, sample)
            with torch.no_grad():
                prompt, tags = encode_joint_prompt_condition(
                    image, caption, processor=modules.processor, tokenizer=modules.tokenizer,
                    text_encoder=modules.text_encoder, device=device)
            if prompt.ndim == 2:
                prompt = prompt.unsqueeze(0)
            root = condition_dir(cfg, sample)
            root.mkdir(parents=True, exist_ok=True)
            save_file(dict(prompt_embeds=prompt.cpu().contiguous(), text_token_tags=tags.cpu().contiguous()),
                      str(root / "qwen.safetensors"))
        del modules, prompt, tags
        gc.collect()
        torch.cuda.empty_cache()
    silence_path = path(cfg.silence_file)
    if make_silence and not silence_path.is_file():
        modules = load_conditioners(model_config(cfg), device=device, qwen=False,
                                    video_vae=False, audio_vae=True, schedulers=False)
        with torch.no_grad():
            silence = modules.audio_vae.encode(torch.zeros(2, 1, 263 * 800, device=device), return_dict=False)[0].mode().float().cpu()
            mean = torch.tensor(modules.audio_vae.config.latents_mean).reshape(1, -1, 1)
            std = torch.tensor(modules.audio_vae.config.latents_std).reshape(1, -1, 1)
            silence = ((silence - mean) / std).bfloat16().contiguous()
        silence_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = silence_path.with_name(f"{silence_path.name}.{os.getpid()}.tmp")
        save_file(dict(silence_158f=silence), str(temporary))
        temporary.replace(silence_path)
        del modules
        gc.collect()
        torch.cuda.empty_cache()
    if pending:
        modules = load_conditioners(model_config(cfg), device=device, qwen=False,
                                    video_vae=True, audio_vae=False, schedulers=False)
        for sample in pending:
            image, _, camera = read_sample(cfg, sample)
            pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None, :, None].to(device)
            with torch.no_grad():
                anchor = encode_vae_condition(modules.video_vae, pixels, (0.485, 0.456, 0.406),
                                              (0.229, 0.224, 0.225), encode_seed=42)
            root = condition_dir(cfg, sample)
            values = load_file(str(root / "qwen.safetensors"))
            values.update(camera, anchor_latents=anchor.detach().cpu().contiguous())
            save_file(values, str(root / "condition.tmp"))
            (root / "condition.tmp").replace(root / "condition.safetensors")
            (root / "qwen.safetensors").unlink()
            write_ready(root, sample)
            print(json.dumps(dict(encoded=sample_name(sample))), flush=True)
    return dict(encoded=len(pending), samples=len(samples))


def prepare(cfg):
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise RuntimeError("Run prepare on one GPU without torchrun")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(2)
    # A given manifest is only read; without one the LingBot examples become conditions_dir/manifest.json.
    samples = select_samples(read_manifest(cfg) if cfg.manifest else make_samples(cfg), cfg)
    index, count = shard_spec(cfg.get("prepare_shard"))
    # Shard the name-sorted selection, not the pending list, so concurrent shards never overlap.
    result = encode(cfg, samples[index::count], make_silence=index == 0)
    if count > 1:
        result.update(shard=f"{index}/{count}", selected=len(samples))
    if not cfg.manifest and index == 0:
        root = path(cfg.conditions_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "manifest.json").write_text(json.dumps(dict(samples=samples), indent=2))
    return result
