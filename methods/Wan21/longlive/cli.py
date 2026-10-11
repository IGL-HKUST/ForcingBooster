"""python -m methods.Wan21.longlive.cli {prepare,train,infer}."""
import argparse
import json

from omegaconf import OmegaConf

from .config import load_config, resolve_weights


def main(argv=None):
    parser = argparse.ArgumentParser(description="LongLive 1.0 and patch router")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "train", "infer"):
        child = sub.add_parser(name)
        child.add_argument("--config", required=True)
        child.add_argument("--runtime")
        child.add_argument("--set", action="append", default=[])
        child.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.runtime, args.set)
    if args.dry_run:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return
    cfg = resolve_weights(cfg, args.command)
    if args.command == "prepare":
        from .weights import prepare
        result = prepare(cfg)
    elif args.command == "train":
        from .trainer import train
        result = train(cfg)
    else:
        from .inference import infer
        result = infer(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
