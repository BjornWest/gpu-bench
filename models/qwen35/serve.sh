#!/usr/bin/env bash
# Launch vLLM for Qwen3.5-397B-A17B
# Edit this file directly to tune any argument.
#
# Notes:
#   --dtype auto          detects FP8 weights from config.json; change to
#                         bfloat16 if loading the BF16 checkpoint instead
#   --enable-expert-parallel  routes expert tokens across GPUs (MoE-aware);
#                         faster than pure tensor-parallel for sparse MoE
#   --max-num-seqs 512    server-side ceiling; actual batch size is controlled
#                         by the benchmark client (asyncio.Semaphore)

set -euo pipefail
MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$MODEL_DIR/venv"

if [ ! -d "$VENV" ]; then
    echo "venv not found. Run: bash models/qwen35/setup.sh" >&2
    exit 1
fi
source "$VENV/bin/activate"

export VLLM_SERVER_DEV_MODE=1


vllm serve Qwen/Qwen3.5-397B-A17B-FP8 
    --port 8000 \
    --tensor-parallel-size 8 \
    --max-model-len 262144 \
    --reasoning-parser qwen3 \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
    --enable-sleep-mode

