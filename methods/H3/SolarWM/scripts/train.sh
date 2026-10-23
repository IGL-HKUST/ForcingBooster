#!/usr/bin/env bash
set -x

# Configuration (run from the project root). The eight GPUs form one sequence-parallel group
# that trains the replicated FP32 router head on one sample per attempt.
CONFIG="methods/H3/SolarWM/config/train.yaml"
NPROC_PER_NODE=8
MASTER_PORT="${MASTER_PORT:-29621}"

"${PYTHON:-python}" -m torch.distributed.run --standalone \
  --nproc-per-node="$NPROC_PER_NODE" \
  --master-port="$MASTER_PORT" \
  -m methods.H3.cli train \
  --config "$CONFIG" \
  "$@"
