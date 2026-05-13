#!/usr/bin/env bash
# Download GLM-5 NVFP4 weights.

set -euo pipefail
MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$MODEL_DIR/../_download_lib.sh"

REPO=${REPO:-zai-org/GLM-5-NVFP4}
SIZE_GB=${SIZE_GB:-460}
INCLUDE_PAT=${INCLUDE_PAT:-"*.safetensors *.json *.txt tokenizer*"}

download_model "$REPO" "$MODEL_DIR/model" "$SIZE_GB" "$INCLUDE_PAT"
