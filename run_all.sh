#!/usr/bin/env bash
# Run the TPS sweep for every model in ALL_MODELS (defined in sweep.conf).
#
# Usage:  bash run_all.sh
# Single model: bash run_bench.sh <model_name>

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/sweep.conf"

FAILED=()

for MODEL in "${ALL_MODELS[@]}"; do
    echo ""
    echo "========================================"
    echo "  $MODEL"
    echo "========================================"
    if bash "$SCRIPT_DIR/run_bench.sh" "$MODEL"; then
        echo "  OK"
    else
        echo "  FAILED — continuing"
        FAILED+=("$MODEL")
    fi
done

echo ""
echo "All models processed. Results: $SCRIPT_DIR/results/"
[ ${#FAILED[@]} -gt 0 ] && echo "Failed: ${FAILED[*]}"

BENCH_VENV="$SCRIPT_DIR/bench/venv"
if [ -d "$BENCH_VENV" ]; then
    source "$BENCH_VENV/bin/activate"
fi
python "$SCRIPT_DIR/bench/summarize.py" --results-dir "$SCRIPT_DIR/results"

[ ${#FAILED[@]} -gt 0 ] && exit 1
exit 0
