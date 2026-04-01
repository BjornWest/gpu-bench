#!/usr/bin/env bash
# Launch vLLM for Devstral-2-123B
# Edit this file directly to tune any argument.
#
# Notes:
#   Dense model; no expert parallelism flag needed.
#   TP=4 uses half the pod GPUs, leaving the other 4 free for a second model
#   if you want to run two servers simultaneously. Change to 8 for lower latency.

set -euo pipefail
MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$MODEL_DIR/venv"

if [ ! -d "$VENV" ]; then
    echo "venv not found. Run: bash models/devstral2/setup.sh" >&2
    exit 1
fi
source "$VENV/bin/activate"

export VLLM_SERVER_DEV_MODE=1
exec vllm serve "$MODEL_DIR/model" \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --max-num-seqs 512 \
    --port 8000 \
    --served-model-name devstral2 \
    --enable-prompt-tokens-details \
    --enable-sleep-mode \
    --disable-log-requests