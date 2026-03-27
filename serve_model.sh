#!/usr/bin/env bash
# Thin dispatcher — delegates to models/<name>/serve.sh
# Usually called by run_bench.sh rather than directly.
#
# Usage: bash serve_model.sh <model_name>

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_NAME="${1:?Usage: serve_model.sh <model_name>}"
SERVE_SCRIPT="$SCRIPT_DIR/models/$MODEL_NAME/serve.sh"

if [ ! -f "$SERVE_SCRIPT" ]; then
    echo "No serve script found at $SERVE_SCRIPT" >&2
    echo "Available models:" >&2
    ls "$SCRIPT_DIR/models/" >&2
    exit 1
fi

exec bash "$SERVE_SCRIPT"
