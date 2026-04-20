#!/usr/bin/env bash
# venv setup + model download for Devstral-2-123B
# Dense BF16 model; no special RC needed.

set -euo pipefail
MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

uv venv "$MODEL_DIR/venv"
source "$MODEL_DIR/venv/bin/activate"

uv pip install "vllm>=0.9.0"
uv pip install hf-transfer huggingface_hub

# Download weights to local directory
HF_HUB_ENABLE_HF_TRANSFER=1 hf download \
    "mistralai/Devstral-2-123B" \
    --include "model-*.safetensors" "model.safetensors.index.json" "config.json" "tokenizer*" "*.jinja" "generation_config.json" \
    --exclude "consolidated*" \
    --local-dir "$MODEL_DIR/model"

echo "Done: $MODEL_DIR/venv  weights: $MODEL_DIR/model"
