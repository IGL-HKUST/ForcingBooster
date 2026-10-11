"""Configuration for local LingBot example preparation and H3 inference."""
from pathlib import Path
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


def path(value):
    p = Path(str(value)).expanduser()
    return p if p.is_absolute() else ROOT / p


def load_config(filename, runtime=None, overrides=()):
    pieces = [OmegaConf.load(Path(__file__).parent / "config/defaults.yaml"),
              OmegaConf.load(path(filename))]
    if runtime:
        pieces.append(OmegaConf.load(path(runtime)))
    pieces.append(OmegaConf.from_dotlist(list(overrides)))
    cfg = OmegaConf.merge(*pieces)
    OmegaConf.resolve(cfg)
    if cfg.mode not in ("native", "router"):
        raise ValueError("mode must be native or router")
    if cfg.forced_exit_step is not None:
        if cfg.forced_exit_step not in (1, 2, 3, 4) or cfg.mode != "router":
            raise ValueError("forced_exit_step requires mode=router and a step in 1..4")
    if not 0 <= cfg.exit_threshold <= 1:
        raise ValueError("exit_threshold must lie in [0,1]")
    return cfg


def resolve_weights(cfg, lifecycle):
    from methods.core.weights import directory, checkpoint, require, validate_shards
    base = "SolarWM-h3-33B-base"
    stage2 = "SolarWM-h3-33B-sgf-stage2-158f-fix"
    if lifecycle == "infer" and cfg.mode == "router" and cfg.forced_exit_step is None and not cfg.weights.checkpoint:
        raise ValueError("H3 router inference requires weights.checkpoint or forced_exit_step")
    components = (["transformer"] + (["vae"] if cfg.decode else [])
                  if lifecycle == "infer" else
                  ["text_encoder", "processor", "tokenizer", "vae", "audio_vae", "scheduler"])
    patterns = [f"{base}/{part}/*" for part in components] + [f"{base}/*.json"]
    if lifecycle == "infer":
        patterns += [f"{stage2}/ema.pt", f"{stage2}/*.json"]
    root = directory(cfg.weights.model, patterns=patterns)
    cfg.base_path = str(require(root / base))
    for part in components:
        require(root / base / part)
        if part in {"transformer", "text_encoder", "vae", "audio_vae"}:
            require(root / base / part / "config.json")
            validate_shards(root / base / part)
    if lifecycle == "infer":
        cfg.stage2_path = str(require(root / stage2))
        for name in ("ema.pt", "COMPLETE.json", "checkpoint-manifest.json"):
            require(root / stage2 / name)
    cfg.trained_path = (str(checkpoint(cfg.weights.checkpoint))
                        if lifecycle == "infer" and cfg.mode == "router"
                        and cfg.forced_exit_step is None and cfg.weights.checkpoint else None)
    return cfg
