#!/usr/bin/env bash
# Download MiniMax M2.7 NVFP4 weights.
#
# Default: nvidia/MiniMax-M2.7-NVFP4 — the official NVIDIA quant
# (verified to exist on HuggingFace; 230B params, 10B active, MoE).
# Only MoE expert MLPs are quantized to NVFP4; other layers stay BF16.

set -euo pipefail
MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$MODEL_DIR/../_download_lib.sh"

REPO=${REPO:-nvidia/MiniMax-M2.7-NVFP4}
SIZE_GB=${SIZE_GB:-150}
INCLUDE_PAT=${INCLUDE_PAT:-"*.safetensors *.json *.txt tokenizer*"}

download_model "$REPO" "$MODEL_DIR/model" "$SIZE_GB" "$INCLUDE_PAT"
