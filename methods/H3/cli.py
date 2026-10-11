"""python -m methods.H3.cli {prepare,infer}."""
import argparse
import json
from omegaconf import OmegaConf
from .config import load_config, resolve_weights


def main(argv=None):
    parser = argparse.ArgumentParser(description="SolarWM-H3 Stage2 patch router")
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "infer"):
        p = sub.add_parser(action)
        p.add_argument("--config", required=True)
        p.add_argument("--runtime")
        p.add_argument("--set", action="append", default=[])
        p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.runtime, args.set)
    if args.dry_run:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return
    cfg = resolve_weights(cfg, args.action)
    if args.action == "prepare":
        from .data import prepare
        result = prepare(cfg)
    else:
        from .inference import infer
        result = infer(cfg)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

