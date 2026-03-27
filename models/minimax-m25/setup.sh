#!/usr/bin/env bash
# venv setup + model download for MiniMax-M2.5
#
# The minimax_m2 tool-call and reasoning parsers must be present in the
# installed vLLM version. Check the vLLM changelog or the MiniMax model card
# for the minimum required version / RC and pin accordingly.

set -euo pipefail
MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

uv venv "$MODEL_DIR/venv"
source "$MODEL_DIR/venv/bin/activate"

# TODO: pin to the version that includes minimax_m2 parsers
uv pip install "vllm>=0.9.0"
uv pip install hf-transfer huggingface_hub

# Download weights to local directory
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    "MiniMaxAI/MiniMax-M2.5" \
    --local-dir "$MODEL_DIR/model"

echo "Done: $MODEL_DIR/venv  weights: $MODEL_DIR/model"
