#!/usr/bin/env bash
# Launch vLLM for gpt-oss-20b (local debug model — single GPU, no setup needed).
# Model weights are expected at /home/bjowes/code/models/gpt-oss-20b.
# Edit MODEL_PATH if your weights live elsewhere.
#
# This model is used for local benchmarking/debugging without requiring the
# H100 cluster. It has no per-model venv — it uses the system uv environment.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/bjowes/code/models/gpt-oss-20b}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Model not found at $MODEL_PATH" >&2
    echo "Set MODEL_PATH env var to override." >&2
    exit 1
fi

export TIKTOKEN_ENCODINGS_BASE="/home/bjowes/code/models/encodings/"
export TIKTOKEN_RS_CACHE_DIR="/home/bjowes/code/models/encodings/"
export VLLM_SERVER_DEV_MODE=1

uv run python -m vllm.entrypoints.openai.api_server \
  --max-model-len 40480 \
  --model /home/bjowes/code/models/gpt-oss-20b \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 600 \
  --served-model-name gpt-oss-20b \
  --trust-remote-code \
  --tool-call-parser openai \
  --enable-auto-tool-choice \
  --reasoning-parser openai_gptoss \
  --enable-prompt-tokens-details \
  --enable-sleep-mode \
