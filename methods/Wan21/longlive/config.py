"""Small, standalone configuration contract (no artificial parent stage)."""
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]


def resolve_path(value):
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_config(path, runtime=None, overrides=()):
    defaults = OmegaConf.load(Path(__file__).parent / "config" / "defaults.yaml")
    configs = [defaults, OmegaConf.load(path)]
    if runtime:
        configs.append(OmegaConf.load(runtime))
    configs.append(OmegaConf.from_dotlist(list(overrides)))
    config = OmegaConf.merge(*configs)
    validate(config)
    return config


def validate(config):
    if int(config.num_frame_per_block) != 3:
        raise ValueError("LongLive v1.0 uses 3 latent frames per chunk")
    if int(config.local_attn_size) != 12 or int(config.sink_size) != 3:
        raise ValueError("This release preserves the official 12-frame window and 3-frame sink")
    if config.get("use_infinite_attention", False):
        raise ValueError("Infinity requires a different cache representation and is not supported")
    if float(config.context_noise) != 0:
        raise ValueError("This release uses the official clean-cache timestep context_noise=0")
    if int(config.save_every) < 1 or int(config.max_updates) < 1 or int(config.max_attempts) < 1:
        raise ValueError("Training update/save/attempt counts must be positive")
    if not 0 <= float(config.exit_threshold) <= 1 or float(config.nfe_weight) < 0:
        raise ValueError("Invalid exit threshold or NFE weight")
    frames = int(config.num_output_frames)
    if not 3 <= frames <= 1023 or frames % 3:
        raise ValueError("num_output_frames counts latent frames and must be a multiple of 3 in [3,1023]")
    if int(config.height) % 16 or int(config.width) % 16:
        raise ValueError("height and width must be divisible by 16")
    switches = [int(x) for x in config.switch_frame_indices]
    if switches != sorted(set(switches)) or any(x <= 0 or x >= frames or x % 3 for x in switches):
        raise ValueError("switch_frame_indices must be increasing chunk-aligned latent frame indices")
    if len(config.prompts) != len(switches) + 1 or any(not str(p).strip() for p in config.prompts):
        raise ValueError("provide one nonempty prompt for each segment")
    if config.mode not in ("native", "router"):
        raise ValueError("mode must be native or router")
    if config.weight_mode not in ("merged", "official_lora"):
        raise ValueError("weight_mode must be merged or official_lora")
    if config.weight_mode == "official_lora" and (config.mode != "native" or config.weights.checkpoint):
        raise ValueError("official_lora is an unmerged native reference, not a router checkpoint mode")
    if int(config.num_training_frames) != 21:
        raise ValueError("The initial router training contract is 21 latent frames")
    if float(config.timestep_shift) != 5 or list(config.denoising_steps) != [1000, 750, 500, 250]:
        raise ValueError("Preserve the official four-step schedule and shift=5")
    if config.get("forced_exit_step") is not None and not 1 <= int(config.forced_exit_step) <= 4:
        raise ValueError("forced_exit_step must be in [1,4]")


def resolve_weights(cfg, lifecycle):
    from methods.core.weights import wan_model, checkpoint
    if lifecycle == "infer" and cfg.mode == "router" and not cfg.weights.checkpoint and cfg.forced_exit_step is None:
        raise ValueError("LongLive router inference requires weights.checkpoint")
    cfg.assets_path = str(wan_model(cfg.weights.assets, assets=lifecycle != "prepare"))
    if lifecycle == "prepare" or cfg.weight_mode == "official_lora":
        cfg.base_path = str(checkpoint(cfg.weights.model, filename="models/longlive_base.pt"))
        cfg.lora_path = str(checkpoint(cfg.weights.model, filename="models/lora.pt"))
    source = None
    if lifecycle == "train" and not cfg.resume:
        source = cfg.weights.init
    elif lifecycle == "infer":
        source = cfg.weights.checkpoint
    cfg.trained_path = str(checkpoint(source)) if source else None
    return cfg
