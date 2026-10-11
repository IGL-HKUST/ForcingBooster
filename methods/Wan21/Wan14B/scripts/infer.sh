#!/usr/bin/env bash
set -x

# Configuration (run from the project root)
MODEL="wan21-14b"
STAGE="router"
CONFIG="methods/Wan21/Wan14B/config/$STAGE.yaml"

python \
  -m methods.cli infer \
  --model "$MODEL" \
  --stage "$STAGE" \
  --config "$CONFIG" \
  "$@"
