#!/usr/bin/env bash
# Top-level setup: installs benchmark client dependencies (InferenceX, aiohttp).
# This does NOT set up model venvs — run each model's own setup.sh for that.
#
# Run once per pod session:
#   bash gpu-bench/setup.sh
#
# Then set up whichever models you need:
#   bash gpu-bench/models/qwen35/setup.sh
#   bash gpu-bench/models/deepseek-v32/setup.sh
#   ...

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_VENV="$SCRIPT_DIR/bench/venv"

echo "==> Setting up benchmark client venv ..."
uv venv "$BENCH_VENV"
source "$BENCH_VENV/bin/activate"
uv pip install aiohttp openai numpy matplotlib pandas tabulate

# InferenceX provides benchmark_serving.py which is the primary sweep runner.
INFERENCEX_DIR="$SCRIPT_DIR/InferenceX"
if [ ! -d "$INFERENCEX_DIR" ]; then
    echo "==> Cloning InferenceX ..."
    git clone https://github.com/SemiAnalysisAI/InferenceX.git "$INFERENCEX_DIR"
else
    echo "==> InferenceX already present; pulling latest ..."
    git -C "$INFERENCEX_DIR" pull --ff-only
fi

BENCH_REQ="$INFERENCEX_DIR/utils/bench_serving/requirements.txt"
if [ -f "$BENCH_REQ" ]; then
    uv pip install -r "$BENCH_REQ"
fi

echo ""
echo "Benchmark client ready: $BENCH_VENV"
echo ""
echo "Model venvs — run whichever you need:"
for model_dir in "$SCRIPT_DIR/models"/*/; do
    model="$(basename "$model_dir")"
    echo "  bash $SCRIPT_DIR/models/$model/setup.sh"
done
