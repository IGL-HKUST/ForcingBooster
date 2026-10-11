#!/usr/bin/env bash
set -x

# Configuration (run from the project root)
MODEL="wan21-1b3"
STAGE="router"
CONFIG="methods/Wan21/Wan1B3/config/$STAGE.yaml"

python \
  -m methods.cli prepare \
  --model "$MODEL" \
  --stage "$STAGE" \
  --config "$CONFIG" \
  "$@"
