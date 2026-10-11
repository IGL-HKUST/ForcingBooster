#!/usr/bin/env bash
set -x

# Configuration (run from the project root)
CONFIG="methods/LingBotWorld/LingBotV2_1B3/config/baseline.yaml"

python \
  -m methods.LingBotWorld.cli prepare \
  --config "$CONFIG" \
  "$@"
