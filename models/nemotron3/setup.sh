#!/usr/bin/env bash
# venv setup + model download for NVIDIA Nemotron-3 Super 120B-A12B (FP8)
# Pre-quantized FP8 checkpoint; standard stable vLLM works.

set -euo pipefail
MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

uv venv "$MODEL_DIR/venv"
source "$MODEL_DIR/venv/bin/activate"

pip install vllm==0.17.1
uv pip install hf-transfer huggingface_hub

curl -O https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8/raw/main/super_v3_reasoning_parser.py


# Download weights to local directory
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8" \
    --local-dir "$MODEL_DIR/model"

echo "Done: $MODEL_DIR/venv  weights: $MODEL_DIR/model"
