#!/usr/bin/env bash
# venv setup + model download for DeepSeek-V3.2 (AWQ quantized via QuantTrio)
#
# Uses vLLM nightly and DeepGEMM custom kernels per DeepSeek's recommended setup.
# DeepGEMM versions: https://github.com/deepseek-ai/DeepGEMM/releases

set -euo pipefail
MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

uv venv "$MODEL_DIR/venv"
source "$MODEL_DIR/venv/bin/activate"

uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly
uv pip install git+https://github.com/deepseek-ai/DeepGEMM.git@v2.1.1.post3 --no-build-isolation
uv pip install hf-transfer huggingface_hub

# Download weights to local directory
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    "QuantTrio/DeepSeek-V3.2-AWQ" \
    --local-dir "$MODEL_DIR/model"

echo "Done: $MODEL_DIR/venv  weights: $MODEL_DIR/model"
