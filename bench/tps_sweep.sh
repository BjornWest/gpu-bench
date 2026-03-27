#!/usr/bin/env bash
# Sweep batch sizes for a running vLLM server.
# No restart between batch sizes — concurrency controlled by --max-concurrency.
# Stops early when the server's KV cache fills (requests start waiting).
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

if [ -d "$BENCH_VENV" ]; then
    source "$BENCH_VENV/bin/activate"
fi

mkdir -p "$RESULTS_DIR"

if [ -f "$BENCH_SCRIPT" ]; then
    USE_INFERENCEX=1
    echo "Using InferenceX benchmark_serving.py"
else
    USE_INFERENCEX=0
    echo "InferenceX not found — using tps_sweep_minimal.py (run setup.sh to get InferenceX)"
fi

echo "Model: $MODEL_NAME  batch sizes: ${BATCH_SIZES[*]}"
echo "ISL: ${INPUT_LEN} tokens  OSL: ${OUTPUT_LEN} tokens  Prompts: ${NUM_PROMPTS}"
echo "Results: $RESULTS_DIR"
echo ""

# ── Saturation polling ────────────────────────────────────────────────
# Polls /metrics every 0.5s in a background subshell.
# Writes "1" to SAT_FILE when num_requests_waiting > 0 for 3+ consecutive polls.
start_sat_poller() {
    local sat_file="$1"
    (
        CONSEC=0
        while true; do
            RAW=$(curl -sf "http://${HOST}:${PORT}/metrics" 2>/dev/null || true)
            # Sum all num_requests_waiting gauge values (one per model name label)
            WAITING=$(echo "$RAW" | awk '
                /^vllm:num_requests_waiting\{/ { s += $NF }
                END { print (s > 0) ? 1 : 0 }
            ')
            if [ "${WAITING:-0}" -eq 1 ]; then
                CONSEC=$((CONSEC + 1))
                if [ $CONSEC -ge 3 ]; then
                    echo "1" > "$sat_file"
                fi
            else
                CONSEC=0
            fi
            sleep 0.5
        done
    ) &
    echo $!  # return poller PID
}

stop_sat_poller() {
    local pid="$1"
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

# ── Sweep ─────────────────────────────────────────────────────────────
for BS in "${BATCH_SIZES[@]}"; do
    echo "--- batch_size=$BS ---"

    SAT_FILE=$(mktemp)
    POLLER_PID=$(start_sat_poller "$SAT_FILE")

    if [ "$USE_INFERENCEX" -eq 1 ]; then
        python "$BENCH_SCRIPT" \
            --backend vllm \
            --base-url "http://${HOST}:${PORT}" \
            --model "$MODEL_NAME" \
            --dataset-name random \
            --random-input-len "$INPUT_LEN" \
            --random-output-len "$OUTPUT_LEN" \
            --random-prefix-len 0 \
            --max-concurrency "$BS" \
            --num-prompts "$NUM_PROMPTS" \
            --num-warmups "$NUM_WARMUP" \
            --save-result "$RESULTS_DIR/bs$(printf '%04d' "$BS").json" \
            --percentile-metrics ttft,tpot,itl,e2el \
            --request-rate inf
    else
        python "$MINIMAL_SCRIPT" \
            --model "$MODEL_NAME" \
            --base-url "http://${HOST}:${PORT}" \
            --batch-sizes "$BS" \
            --input-len "$INPUT_LEN" \
            --output-len "$OUTPUT_LEN" \
            --num-prompts "$NUM_PROMPTS" \
            --num-warmup "$NUM_WARMUP" \
            --results-dir "$RESULTS_DIR"
    fi

    stop_sat_poller "$POLLER_PID"

    if [ "$(cat "$SAT_FILE" 2>/dev/null)" = "1" ]; then
        rm -f "$SAT_FILE"
        echo ""
        echo "KV cache saturated at batch_size=$BS — stopping sweep."
        break
    fi
    rm -f "$SAT_FILE"
    echo ""
done

echo "Sweep complete: $RESULTS_DIR"
