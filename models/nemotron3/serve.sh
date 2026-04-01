#!/usr/bin/env bash
# Launch vLLM for NVIDIA Nemotron-3 Super 120B-A12B (FP8)
# Edit this file directly to tune any argument.
#
# Notes:
#   MoE model (120B total / 12B active); --enable-expert-parallel is beneficial.
#   FP8 weights are detected automatically from config.json (--dtype auto).
#   TP=4 is sufficient for the active parameter footprint; using 8 reduces
#   latency further at some efficiency cost.

set -euo pipefail
MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$MODEL_DIR/venv"

if [ ! -d "$VENV" ]; then
    echo "venv not found. Run: bash models/nemotron3/setup.sh" >&2
    exit 1
fi
source "$VENV/bin/activate"

export VLLM_SERVER_DEV_MODE=1


vllm serve "$MODEL_DIR/model" \
  --served-model-name nvidia/nemotron-3-super \
  --async-scheduling \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --enable-expert-parallel \
  --swap-space 0 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --max-cudagraph-capture-size 128 \
  --enable-chunked-prefill \
  --mamba-ssm-cache-dtype float16 \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-prompt-tokens-details \
  --enable-sleep-mode
