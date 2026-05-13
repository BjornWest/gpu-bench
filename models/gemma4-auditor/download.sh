#!/usr/bin/env bash
# Download Gemma 4 (auditor role).
#
# If Gemma 4 hasn't shipped by run time, REPO falls back to Gemma 3 27B,
# the closest stable substitute. Override REPO when Gemma 4 lands.

set -euo pipefail
MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$MODEL_DIR/../_download_lib.sh"

REPO=${REPO:-google/gemma-3-27b-it}   # placeholder; swap to gemma-4-* when released
SIZE_GB=${SIZE_GB:-55}
INCLUDE_PAT=${INCLUDE_PAT:-"*.safetensors *.json *.txt tokenizer*"}

download_model "$REPO" "$MODEL_DIR/model" "$SIZE_GB" "$INCLUDE_PAT"
