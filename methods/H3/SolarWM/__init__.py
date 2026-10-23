"""SolarWM MiniMax-H3 Stage2 inference and router training with patch early exits."""
from pathlib import Path
from .config import resolve_weights, validate, validate_train

DEFAULTS = Path(__file__).parent / "config/defaults.yaml"


def prepare(cfg):
    from .data import prepare
    return prepare(cfg)


def infer(cfg):
    from .inference import infer
    return infer(cfg)


def train(cfg):
    from .trainer import train
    return train(cfg)
