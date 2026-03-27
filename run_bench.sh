#!/usr/bin/env bash
# Start the server for one model, run the full batch-size sweep, stop the server.
#
# Usage: bash run_bench.sh <model_name>
# Example: bash run_bench.sh qwen35

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/models.conf"

MODEL_NAME="${1:?Usage: run_bench.sh <model_name>}"
SERVE_SCRIPT="$SCRIPT_DIR/models/$MODEL_NAME/serve.sh"

if [ ! -f "$SERVE_SCRIPT" ]; then
    echo "No serve script for '$MODEL_NAME'. Available:" >&2
    ls "$SCRIPT_DIR/models/" >&2
    exit 1
fi

HEALTH_URL="http://${HOST}:${PORT}/health"
SERVER_PID=""

cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "==> Stopping server (PID $SERVER_PID) ..."
        kill "$SERVER_PID"
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "==> Starting server for $MODEL_NAME ..."
bash "$SERVE_SCRIPT" &
SERVER_PID=$!

echo "==> Waiting for server to become healthy (max 20 min) ..."
WAITED=0
MAX_WAIT=1200
until curl -sf "$HEALTH_URL" > /dev/null 2>&1; do
    sleep 10
    WAITED=$((WAITED + 10))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "Server did not become healthy after ${MAX_WAIT}s." >&2
        exit 1
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server process exited unexpectedly." >&2
        exit 1
    fi
    echo "  ... ${WAITED}s"
done
echo "Server healthy (${WAITED}s)."

bash "$SCRIPT_DIR/bench/tps_sweep.sh" "$MODEL_NAME"

echo "==> Done: $MODEL_NAME"
# cleanup() kills server on EXIT
