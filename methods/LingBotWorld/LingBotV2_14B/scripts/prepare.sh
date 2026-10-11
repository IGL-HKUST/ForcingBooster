#!/usr/bin/env bash
set -x

# Configuration (run from the project root)
CONFIG="methods/LingBotWorld/LingBotV2_14B/config/router.yaml"

python \
  -m methods.LingBotWorld.cli prepare \
  --config "$CONFIG" \
  "$@"
