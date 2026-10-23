#!/usr/bin/env bash
set -x

# Configuration (run from the project root). Sequence parallel over every GPU of one node:
# NPROC_PER_NODE=8 by default, 2 or 4 also work.
CONFIG="methods/H3/SolarWM/config/router.yaml"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29611}"

"${PYTHON:-python}" -m torch.distributed.run --standalone \
  --nproc-per-node="$NPROC_PER_NODE" \
  --master-port="$MASTER_PORT" \
  -m methods.H3.cli infer \
  --config "$CONFIG" \
  "$@"
