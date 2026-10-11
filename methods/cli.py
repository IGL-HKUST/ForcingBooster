"""Unified model/stage command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from typing import Any, Sequence

from methods.core.config import LoadedConfig, load_experiment_config
from methods.core.errors import CoreError
from methods.core.registry import ModelRegistry
from methods.core.runtime import load_and_run


LIFECYCLE_COMMANDS = ("prepare", "train", "infer", "validate")


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="registered model ID")
    parser.add_argument("--stage", required=True, help="tf, cd, dmd, adaptive_dmd, or router")
    parser.add_argument(
        "--config", "--recipe", dest="recipe",
        help="config name or YAML path (defaults to config/<stage>.yaml)",
    )
    parser.add_argument(
        "--runtime",
        help="runtime profile name or YAML path (defaults to manifest.defaults.runtime)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="structured dotted override; may be repeated",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print resolved config without importing the model adapter",
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="re-raise lifecycle errors with a full traceback for smoke-test diagnosis",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m methods.cli",
        description="Causal-Forcing multi-model lifecycle CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (*LIFECYCLE_COMMANDS, "config"):
        child = subparsers.add_parser(command)
        _add_config_arguments(child)
    return parser


def _load(args: argparse.Namespace) -> LoadedConfig:
    registry = ModelRegistry.discover()
    return load_experiment_config(
        registry,
        args.model,
        args.stage,
        recipe=args.recipe,
        runtime=args.runtime,
        overrides=args.overrides,
    )


def _print_result(value: Any) -> None:
    if value is None:
        return
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, (dict, list, tuple, str, int, float, bool)):
        print(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
    else:
        print(str(value))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        loaded = _load(args)
        if args.command == "config" or args.dry_run:
            print(loaded.to_yaml(resolve=True), end="")
            return 0
        invocation = load_and_run(args.command, loaded)
        _print_result(invocation.result)
        return 0
    except CoreError as exc:
        if getattr(args, "traceback", False):
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["LIFECYCLE_COMMANDS", "build_parser", "main"]
