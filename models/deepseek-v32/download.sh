#!/usr/bin/env bash
# Download DeepSeek V3.2 NVFP4 weights.
#
# CPU-only node: this script doesn't install vllm or torch. It only fetches
# the weights to disk. Run on a cheap CPU node, then attach the volume to
# the B200 instance.
#
# Override the repo via REPO env var (e.g. for a different quant or fork).

set -euo pipefail
MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$MODEL_DIR/../_download_lib.sh"

# Default: best available NVFP4 quant. The "deepseek-v4-pro" label is what
# InferenceX uses internally for what appears as dsv4; use the canonical
# DeepSeek V3.2 NVFP4 repo path here. Override via REPO if a newer quant
# ships closer to run date.
REPO=${REPO:-deepseek-ai/DeepSeek-V3.2-Exp-NVFP4}
SIZE_GB=${SIZE_GB:-420}

# Skip community GGUF shards if the repo carries them — we want safetensors.
INCLUDE_PAT=${INCLUDE_PAT:-"*.safetensors *.json *.txt tokenizer*"}

download_model "$REPO" "$MODEL_DIR/model" "$SIZE_GB" "$INCLUDE_PAT"
