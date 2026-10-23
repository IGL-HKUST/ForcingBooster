"""Recipe merging shared by H3 variants."""
from pathlib import Path
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]


def path(value):
    p = Path(str(value)).expanduser()
    return p if p.is_absolute() else ROOT / p


def load_config(defaults, filename, runtime=None, overrides=()):
    pieces = [OmegaConf.load(defaults), OmegaConf.load(path(filename))]
    if runtime:
        pieces.append(OmegaConf.load(path(runtime)))
    pieces.append(OmegaConf.from_dotlist(list(overrides)))
    cfg = OmegaConf.merge(*pieces)
    OmegaConf.resolve(cfg)
    return cfg
