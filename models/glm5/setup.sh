#!/usr/bin/env bash
# venv setup + model download for GLM-5 (AWQ quantized via QuantTrio)
# TODO: pin to the RC validated for this model if a nightly is needed

set -euo pipefail
MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

uv venv "$MODEL_DIR/venv"
source "$MODEL_DIR/venv/bin/activate"

pip install -U vllm --pre --index-url https://pypi.org/simple --extra-index-url https://wheels.vllm.ai/nightly
pip install git+https://github.com/huggingface/transformers.git
pip install git+https://github.com/deepseek-ai/DeepGEMM.git@v2.1.1.post3 --no-build-isolation


# Download weights to local directory
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    "QuantTrio/GLM-5-AWQ" \
    --local-dir "$MODEL_DIR/model"

echo "Done: $MODEL_DIR/venv  weights: $MODEL_DIR/model"
