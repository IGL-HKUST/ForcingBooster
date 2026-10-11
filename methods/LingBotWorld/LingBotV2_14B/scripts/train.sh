#!/usr/bin/env bash
set -x

# Configuration (run from the project root)
CONFIG="methods/LingBotWorld/LingBotV2_14B/config/router.yaml"
NPROC_PER_NODE=8
MASTER_PORT=29608
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

torchrun --standalone \
  --nproc-per-node="$NPROC_PER_NODE" \
  --master-port="$MASTER_PORT" \
  -m methods.LingBotWorld.cli train \
  --config "$CONFIG" \
  "$@"
