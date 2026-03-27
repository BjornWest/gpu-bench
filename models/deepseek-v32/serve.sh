#!/usr/bin/env bash
# Launch vLLM for DeepSeek-V3.2 (AWQ, QuantTrio checkpoint)
# Edit this file directly to tune any argument.
#
# Notes:
#   --quantization awq    load the AWQ-quantized weights; vLLM may auto-detect
#                         this from quantize_config.json — remove flag if redundant
#   --enable-expert-parallel  MoE expert routing across GPUs

set -euo pipefail
MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$MODEL_DIR/venv"

if [ ! -d "$VENV" ]; then
    echo "venv not found. Run: bash models/deepseek-v32/setup.sh" >&2
    exit 1
fi
source "$VENV/bin/activate"

export VLLM_USE_DEEP_GEMM=0  # ATM, this line is a "must" for Hopper devices
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export VLLM_USE_FLASHINFER_MOE_FP16=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=4
export VLLM_SERVER_DEV_MODE=1

CONTEXT_LENGTH=32768
SPEC_CONFIG="{\"model\": \"${MODEL_DIR}/model\", \"num_speculative_tokens\": 1}"
vllm serve \
    "$MODEL_DIR/model" \
    --served-model-name MY_MODEL_NAME \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v31 \
    --reasoning-parser deepseek_v3 \
    --swap-space 16 \
    --max-num-seqs 32 \
    --max-model-len $CONTEXT_LENGTH \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \  # optional
    --speculative-config "$SPEC_CONFIG" \  # optional, 50%+- throughput increase is observed
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-sleep-mode

