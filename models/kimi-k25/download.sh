#!/usr/bin/env bash
# Download Kimi K2.5 NVFP4 weights.
# Large model (~600 GB) — ensure target volume has 700+ GB free.

set -euo pipefail
MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$MODEL_DIR/../_download_lib.sh"

REPO=${REPO:-moonshotai/Kimi-K2.5-NVFP4}
SIZE_GB=${SIZE_GB:-620}
INCLUDE_PAT=${INCLUDE_PAT:-"*.safetensors *.json *.txt tokenizer*"}

download_model "$REPO" "$MODEL_DIR/model" "$SIZE_GB" "$INCLUDE_PAT"
