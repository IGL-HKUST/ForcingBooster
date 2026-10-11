"""Config-driven LingBot-World lifecycle entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig, OmegaConf


def load_config(
    config_path: str | Path,
    *,
    runtime: str | Path | None = None,
    overrides: Sequence[str] = (),
) -> DictConfig:
    """Merge a recipe, optional runtime YAML, and dotted command-line values."""

    config = OmegaConf.load(config_path)
    if runtime is not None:
        config = OmegaConf.merge(config, OmegaConf.load(runtime))
    return OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m methods.LingBotWorld.cli",
        description="LingBot-World training and inference from YAML configuration",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "train", "infer"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True, help="recipe YAML path")
        child.add_argument("--runtime", help="optional runtime YAML path")
        child.add_argument(
            "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
        )
        child.add_argument(
            "--dry-run", action="store_true", help="print merged config without loading models"
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, runtime=args.runtime, overrides=args.overrides)
    if args.dry_run:
        print(OmegaConf.to_yaml(config, resolve=True), end="")
        return 0

    from methods.LingBotWorld.shared.adapter import LingBotV2Adapter

    result = getattr(LingBotV2Adapter(), args.command)(config)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "load_config", "main"]
