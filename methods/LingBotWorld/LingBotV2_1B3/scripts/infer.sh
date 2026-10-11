#!/usr/bin/env bash
set -x

# Configuration (run from the project root)
CONFIG="methods/LingBotWorld/LingBotV2_1B3/config/baseline.yaml"
NPROC_PER_NODE=1
MASTER_PORT=29610

torchrun --standalone \
  --nproc-per-node="$NPROC_PER_NODE" \
  --master-port="$MASTER_PORT" \
  -m methods.LingBotWorld.cli infer \
  --config "$CONFIG" \
  "$@"
