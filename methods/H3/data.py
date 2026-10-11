"""LingBot asset examples and phased, offline H3 condition encoding."""
import gc
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .config import ROOT, path
from .runtime import model_config
from .upstream import activate


def condition_key(sample):
    # Preserve existing manifest-to-directory keys without retaining receipts.
    return hashlib.sha256(json.dumps(sample, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sample_name(sample):
    return Path(sample["source_files"]["image"]).parent.name


def select_samples(samples, cfg):
    samples = [s for s in samples if s.get("input_kind") == "lingbot_example"]
    if cfg.sample_ids:
        wanted = {str(x) for x in cfg.sample_ids}
        samples = [s for s in samples if sample_name(s) in wanted or s["sample_id"] in wanted]
        found = {name for s in samples for name in (sample_name(s), s["sample_id"])}
        if wanted - found:
            raise ValueError(f"Unknown LingBot examples: {sorted(wanted - found)}")
    if not samples:
        raise ValueError("No LingBot examples selected")
    return sorted(samples, key=sample_name)


def make_samples(cfg):
    from solarwm.backends.minimax_h3.full_length import full_length_geometry
    geometry = full_length_geometry(158)
    samples = []
    for folder in sorted(path(cfg.examples_root).iterdir()):
        if not folder.is_dir():
            continue
        files = dict(image=folder / "image.jpg", prompt=folder / "prompt.txt",
                     poses=folder / "poses.npy", intrinsics=folder / "intrinsics.npy")
        for filename in files.values():
            if not filename.is_file():
                raise FileNotFoundError(filename)
        samples.append(dict(input_kind="lingbot_example", sample_id=folder.name,
                            num_frames=158, decode_latents=geometry.decode_latents,
                            rollout_latents=geometry.rollout_latents,
                            source_files={k: os.path.relpath(v, ROOT) for k, v in files.items()}))
    return select_samples(samples, cfg)


def read_manifest(cfg):
    filename = path(cfg.manifest) if cfg.manifest else path(cfg.conditions_dir) / "manifest.json"
    return select_samples(json.loads(filename.read_text())["samples"], cfg)


def read_example(sample):
    from solarwm.backends.minimax_h3.full_length import full_length_geometry
    files = {k: path(v) for k, v in sample["source_files"].items()}
    poses = np.load(files["poses"], allow_pickle=False)[:158].astype(np.float64)
    intrinsics = np.load(files["intrinsics"], allow_pickle=False)
    if poses.shape != (158, 4, 4):
        raise ValueError("LingBot poses must contain at least 158 camera frames")
    views = np.linalg.inv(poses) @ poses[:1]
    views[0] = np.eye(4)
    indices = list(full_length_geometry(158).camera_indices)
    fx, fy, cx, cy = intrinsics[0] / np.array([832, 480, 832, 480])
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
    image = Image.open(files["image"]).convert("RGB").resize((1344, 768), Image.Resampling.LANCZOS)
    caption = (files["image"].parent / "prompt.txt").read_text().strip()
    return image, caption, dict(viewmats=torch.from_numpy(views[indices].astype(np.float32)),
                                K=K.repeat(len(indices), 1, 1))


def load_condition(cfg, sample):
    from safetensors.torch import load_file
    filename = path(cfg.conditions_dir) / condition_key(sample) / "condition.safetensors"
    return load_file(str(filename))


def encode(cfg, samples):
    from safetensors.torch import save_file, load_file
    from solarwm.backends.minimax_h3.optional import load_conditioners
    from solarwm.backends.minimax_h3.official_codec import encode_joint_prompt_condition
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    pending = [s for s in samples if not (path(cfg.conditions_dir) / condition_key(s) / "condition.safetensors").is_file()]
    if pending:
        modules = load_conditioners(model_config(cfg), device=device, qwen=True,
                                    video_vae=False, audio_vae=False, schedulers=False)
        for sample in pending:
            image, caption, _ = read_example(sample)
            with torch.no_grad():
                prompt, tags = encode_joint_prompt_condition(
                    image, caption, processor=modules.processor, tokenizer=modules.tokenizer,
                    text_encoder=modules.text_encoder, device=device)
            if prompt.ndim == 2:
                prompt = prompt.unsqueeze(0)
            root = path(cfg.conditions_dir) / condition_key(sample)
            root.mkdir(parents=True, exist_ok=True)
            save_file(dict(prompt_embeds=prompt.cpu().contiguous(), text_token_tags=tags.cpu().contiguous()),
                      str(root / "qwen.safetensors"))
        del modules, prompt, tags
        gc.collect()
        torch.cuda.empty_cache()
    silence_path = path(cfg.silence_file)
    if not silence_path.is_file():
        modules = load_conditioners(model_config(cfg), device=device, qwen=False,
                                    video_vae=False, audio_vae=True, schedulers=False)
        with torch.no_grad():
            silence = modules.audio_vae.encode(torch.zeros(2, 1, 263 * 800, device=device), return_dict=False)[0].mode().float().cpu()
            mean = torch.tensor(modules.audio_vae.config.latents_mean).reshape(1, -1, 1)
            std = torch.tensor(modules.audio_vae.config.latents_std).reshape(1, -1, 1)
            silence = ((silence - mean) / std).bfloat16().contiguous()
        silence_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = silence_path.with_suffix(".tmp")
        save_file(dict(silence_158f=silence), str(temporary))
        temporary.replace(silence_path)
        del modules
        gc.collect()
        torch.cuda.empty_cache()
    if pending:
        modules = load_conditioners(model_config(cfg), device=device, qwen=False,
                                    video_vae=True, audio_vae=False, schedulers=False)
        for sample in pending:
            image, _, camera = read_example(sample)
            pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None, :, None].to(device)
            with torch.no_grad():
                anchor = encode_vae_condition(modules.video_vae, pixels, (0.485, 0.456, 0.406),
                                              (0.229, 0.224, 0.225), encode_seed=42)
            root = path(cfg.conditions_dir) / condition_key(sample)
            values = load_file(str(root / "qwen.safetensors"))
            values.update(camera, anchor_latents=anchor.detach().cpu().contiguous())
            save_file(values, str(root / "condition.tmp"))
            (root / "condition.tmp").replace(root / "condition.safetensors")
            (root / "qwen.safetensors").unlink()
            print(json.dumps(dict(encoded=sample_name(sample))), flush=True)
    return dict(encoded=len(pending), samples=len(samples))


def prepare(cfg):
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise RuntimeError("Run prepare on one GPU without torchrun")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(2)
    activate(path(cfg.upstream_root))
    samples = read_manifest(cfg) if cfg.manifest else make_samples(cfg)
    result = encode(cfg, samples)
    if not cfg.manifest:
        root = path(cfg.conditions_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "manifest.json").write_text(json.dumps(dict(samples=samples), indent=2))
    return result
