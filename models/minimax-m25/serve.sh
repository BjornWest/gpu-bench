#!/usr/bin/env bash
# Launch vLLM for MiniMax-M2.5
# Edit this file directly to tune any argument.
#
# Notes:
#   --enable-auto-tool-choice   enables structured tool-call output
#   --tool-call-parser minimax_m2        MiniMax-specific tool-call parser
#   --reasoning-parser minimax_m2_append_think  captures <think> blocks
#   These parsers must exist in the installed vLLM version; see setup.sh.
#
#   For a pure throughput benchmark you can omit the tool-call / reasoning
#   parser flags — they only affect output parsing, not raw TPS.

set -euo pipefail
MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$MODEL_DIR/venv"

if [ ! -d "$VENV" ]; then
    echo "venv not found. Run: bash models/minimax-m25/setup.sh" >&2
    exit 1
fi
source "$VENV/bin/activate"

export VLLM_SERVER_DEV_MODE=1

exec vllm serve "$MODEL_DIR/model" \
    --tensor-parallel-size 8 \
    --dtype auto \
    --max-model-len 8192 \
    --max-num-seqs 512 \
    --port 8000 \
    --served-model-name minimax-m25 \
    --enable-expert-parallel \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think \
    --enable-prompt-tokens-details
