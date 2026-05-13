#!/usr/bin/env bash
# Download Qwen 3.6 (smaller variant — the "we'll only use Qwen 3.6" choice
# from the deployment plan, NOT the 397B 3.5).
#
# Confirm exact size at run time; the Qwen team typically publishes several
# variants (Instruct, MoE, etc.) under different repos.

set -euo pipefail
MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$MODEL_DIR/../_download_lib.sh"

REPO=${REPO:-Qwen/Qwen3.6-32B-Instruct}
SIZE_GB=${SIZE_GB:-70}
INCLUDE_PAT=${INCLUDE_PAT:-"*.safetensors *.json *.txt tokenizer*"}

download_model "$REPO" "$MODEL_DIR/model" "$SIZE_GB" "$INCLUDE_PAT"
