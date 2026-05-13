#!/usr/bin/env bash
# Download Mistral Medium 3.5 128B (FP8 / NVFP4).
# Replaces the older devstral2 entry in our matrix per the deployment plan.
#
# Mistral hasn't always published official NVFP4 quants on HF; if the
# default repo path doesn't exist at run time, check for a community quant
# or fall back to FP8 weights and quant on the rack.

set -euo pipefail
MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$MODEL_DIR/../_download_lib.sh"

REPO=${REPO:-mistralai/Mistral-Medium-3.5-128B}
SIZE_GB=${SIZE_GB:-260}
INCLUDE_PAT=${INCLUDE_PAT:-"*.safetensors *.json *.txt tokenizer* *.model"}

download_model "$REPO" "$MODEL_DIR/model" "$SIZE_GB" "$INCLUDE_PAT"
