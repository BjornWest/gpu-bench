#!/usr/bin/env bash
# Launch vLLM for GLM-5 (AWQ, QuantTrio checkpoint)
# Edit this file directly to tune any argument.
#
# Notes:
#   --trust-remote-code   required for all GLM model architectures
#   --quantization awq    may be auto-detected; remove if vLLM warns about it

set -euo pipefail
MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$MODEL_DIR/venv"

if [ ! -d "$VENV" ]; then
    echo "venv not found. Run: bash models/glm5/setup.sh" >&2
    exit 1
fi
source "$VENV/bin/activate"

export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_MOE_FP16=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=4
export VLLM_SERVER_DEV_MODE=1


exec vllm serve \
    "$MODEL_DIR/model" \
    --served-model-name MY_MODEL \
    --swap-space 16 \
    --max-num-seqs 32 \
    --max-model-len 32768  \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \ 
    --enable-auto-tool-choice \    
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-prompt-tokens-details \
    --enable-sleep-mode
