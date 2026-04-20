#!/usr/bin/env bash
# venv setup + model download for Qwen3.5-397B-A17B
# TODO: pin to the RC validated for this model if a nightly is needed

set -euo pipefail
MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

uv venv "$MODEL_DIR/venv"
source "$MODEL_DIR/venv/bin/activate"

uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
uv pip install hf-transfer huggingface_hub

# Download weights to local directory
HF_HUB_ENABLE_HF_TRANSFER=1 hf download \
    "Qwen/Qwen3.5-397B-A17B-FP8" \
    --local-dir "$MODEL_DIR/model"

echo "Done: $MODEL_DIR/venv  weights: $MODEL_DIR/model"
