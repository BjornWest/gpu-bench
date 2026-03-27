#!/usr/bin/env bash
# Sweep batch sizes for a running vLLM server.
# No restart between batch sizes — concurrency controlled by --max-concurrency.
#
# Usage:  bash bench/tps_sweep.sh <model_name> [results_dir]
# Requires: server already running and healthy on $HOST:$PORT

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
source "$ROOT_DIR/models.conf"

MODEL_NAME="${1:?Usage: tps_sweep.sh <model_name> [results_dir]}"
RESULTS_DIR="${2:-$ROOT_DIR/results/$MODEL_NAME}"

BENCH_SCRIPT="$ROOT_DIR/InferenceX/utils/bench_serving/benchmark_serving.py"
MINIMAL_SCRIPT="$SCRIPT_DIR/tps_sweep_minimal.py"
BENCH_VENV="$SCRIPT_DIR/venv"

# Activate the bench client venv if present
if [ -d "$BENCH_VENV" ]; then
    source "$BENCH_VENV/bin/activate"
fi

mkdir -p "$RESULTS_DIR"

if [ -f "$BENCH_SCRIPT" ]; then
    echo "Using InferenceX benchmark_serving.py"
    USE_INFERENCEX=1
else
    echo "InferenceX not found — using tps_sweep_minimal.py (run setup.sh to get InferenceX)"
    USE_INFERENCEX=0
fi

echo "Model: $MODEL_NAME  batch sizes: ${BATCH_SIZES[*]}"
echo "Input: ${INPUT_LEN} tokens  Output: ${OUTPUT_LEN} tokens  Prompts: ${NUM_PROMPTS}"
echo "Results: $RESULTS_DIR"
echo ""

if [ "$USE_INFERENCEX" -eq 1 ]; then
    for BS in "${BATCH_SIZES[@]}"; do
        echo "--- batch_size=$BS ---"
        python "$BENCH_SCRIPT" \
            --backend vllm \
            --base-url "http://${HOST}:${PORT}" \
            --model "$MODEL_NAME" \
            --random-input-len "$INPUT_LEN" \
            --random-output-len "$OUTPUT_LEN" \
            --max-concurrency "$BS" \
            --num-prompts "$NUM_PROMPTS" \
            --num-warmups "$NUM_WARMUP" \
            --save-result "$RESULTS_DIR/bs$(printf '%04d' "$BS").json" \
            --percentile-metrics ttft,tpot,itl,e2el \
            --request-rate inf
        echo ""
    done
else
    python "$MINIMAL_SCRIPT" \
        --model "$MODEL_NAME" \
        --base-url "http://${HOST}:${PORT}" \
        --batch-sizes "$(IFS=,; echo "${BATCH_SIZES[*]}")" \
        --input-len "$INPUT_LEN" \
        --output-len "$OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --num-warmup "$NUM_WARMUP" \
        --results-dir "$RESULTS_DIR"
fi

echo "Sweep complete: $RESULTS_DIR"
