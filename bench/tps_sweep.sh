#!/usr/bin/env bash
# Sweep batch sizes for a running vLLM server.
# Runs two test modes:
#   1. Decode — shared prefix, sweep batch sizes (isolates decode throughput)
#   2. Prefill — unique prompts, sweep input lengths × batch sizes (isolates prefill)
#
# No restart between tests — concurrency controlled by --max-concurrency.
# Stops early when the server's KV cache fills.
#
# Usage:  bash bench/tps_sweep.sh [--decode|--prefill] <model_name> [results_dir]
# Requires: server already running and healthy on $HOST:$PORT

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
source "$ROOT_DIR/sweep.conf"

# ── Parse flags ──────────────────────────────────────────────────────
RUN_DECODE=0
RUN_PREFILL=0
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --decode)  RUN_DECODE=1; shift ;;
        --prefill) RUN_PREFILL=1; shift ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

# Default: run both
if [[ $RUN_DECODE -eq 0 && $RUN_PREFILL -eq 0 ]]; then
    RUN_DECODE=1
    RUN_PREFILL=1
fi

MODEL_NAME="${1:?Usage: tps_sweep.sh [--decode|--prefill] <model_name> [results_dir]}"
RESULTS_DIR="${2:-$ROOT_DIR/results/$MODEL_NAME}"

MINIMAL_SCRIPT="$SCRIPT_DIR/tps_sweep_minimal.py"
BENCH_VENV="$SCRIPT_DIR/venv"

# Use the bench venv if it exists, otherwise fall back to uv run.
if [ -d "$BENCH_VENV" ]; then
    source "$BENCH_VENV/bin/activate"
    PY="python"
else
    PY="uv run"
fi

mkdir -p "$RESULTS_DIR"

# ── Decode test ──────────────────────────────────────────────────────
if [[ $RUN_DECODE -eq 1 ]]; then
    echo ""
    echo "=========================================="
    echo "  DECODE TEST — $MODEL_NAME"
    echo "  Shared prefix, sweep batch sizes"
    echo "  ISL: ${DECODE_INPUT_LEN}  OSL: ${DECODE_OUTPUT_LEN}"
    echo "  Batch sizes: ${DECODE_BATCH_SIZES[*]}"
    echo "=========================================="

    $PY "$MINIMAL_SCRIPT" decode \
        --model "$MODEL_NAME" \
        --base-url "http://${HOST}:${PORT}" \
        --batch-sizes "$(IFS=,; echo "${DECODE_BATCH_SIZES[*]}")" \
        --input-len "$DECODE_INPUT_LEN" \
        --output-len "$DECODE_OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --num-warmup "$NUM_WARMUP" \
        --results-dir "$RESULTS_DIR/decode"
fi

# ── Prefill test ─────────────────────────────────────────────────────
if [[ $RUN_PREFILL -eq 1 ]]; then
    echo ""
    echo "=========================================="
    echo "  PREFILL TEST — $MODEL_NAME"
    echo "  Unique prompts, sweep input lengths"
    echo "  ISLs: ${PREFILL_INPUT_LENS[*]}  OSL: ${PREFILL_OUTPUT_LEN}"
    echo "  Batch sizes: ${PREFILL_BATCH_SIZES[*]}"
    echo "=========================================="

    $PY "$MINIMAL_SCRIPT" prefill \
        --model "$MODEL_NAME" \
        --base-url "http://${HOST}:${PORT}" \
        --input-lens "$(IFS=,; echo "${PREFILL_INPUT_LENS[*]}")" \
        --batch-sizes "$(IFS=,; echo "${PREFILL_BATCH_SIZES[*]}")" \
        --output-len "$PREFILL_OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --num-warmup "$NUM_WARMUP" \
        --results-dir "$RESULTS_DIR/prefill"
fi

echo ""
echo "Sweep complete: $RESULTS_DIR"
