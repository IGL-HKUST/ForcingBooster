#!/usr/bin/env bash
set -x

# Configuration (run from the project root)
MODEL="wan21-14b"
STAGE="router"
CONFIG="methods/Wan21/Wan14B/config/$STAGE.yaml"
NNODES=1
NPROC_PER_NODE=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29500

torchrun \
  --nnodes="$NNODES" \
  --nproc-per-node="$NPROC_PER_NODE" \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  -m methods.cli train \
  --model "$MODEL" \
  --stage "$STAGE" \
  --config "$CONFIG" \
  "$@"
