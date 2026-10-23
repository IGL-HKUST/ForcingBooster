#!/usr/bin/env bash
set -x

# Configuration (run from the project root). Downloads every weight file that infer
# needs, then encodes the LingBot examples on one GPU.
CONFIG="methods/H3/SolarWM/config/router.yaml"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON:-python}" \
  -m methods.H3.cli prepare \
  --config "$CONFIG" \
  "$@"
