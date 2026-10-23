"""python -m methods.H3.cli {prepare,infer,train} --config methods/H3/<variant>/config/<recipe>.yaml."""
import argparse
import importlib
import json
import os
from omegaconf import OmegaConf
from .shared.config import load_config, path


def main(argv=None):
    parser = argparse.ArgumentParser(description="MiniMax-H3 patch router methods")
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "infer", "train"):
        p = sub.add_parser(action)
        p.add_argument("--config", required=True)
        p.add_argument("--runtime")
        p.add_argument("--set", action="append", default=[])
        p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    variant = OmegaConf.load(path(args.config)).get("variant")
    if not isinstance(variant, str) or not variant.isidentifier():
        raise ValueError(f"{args.config} must name its H3 variant, e.g. 'variant: SolarWM'")
    method = importlib.import_module(f"{__package__}.{variant}")
    check = method.validate_train if args.action == "train" else method.validate
    cfg = check(load_config(method.DEFAULTS, args.config, args.runtime, args.set))
    if args.dry_run:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return
    if args.action in ("infer", "train"):
        # prepare downloads every file; the infer/train ranks only read the local cache.
        os.environ["HF_HUB_OFFLINE"] = "1"
    cfg = method.resolve_weights(cfg, args.action)
    result = getattr(method, args.action)(cfg)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
