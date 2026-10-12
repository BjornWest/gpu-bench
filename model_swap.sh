#!/usr/bin/env bash
# model_swap.sh — suspend/resume a vLLM instance via sleep-L2 + cuda-checkpoint
#
# The model must be launched with --enable-sleep-mode AND VLLM_SERVER_DEV_MODE=1
# for /sleep to work (the sleep router only registers in dev mode).
#
# Usage:
#   ./model_swap.sh suspend [port]
#   ./model_swap.sh resume  [port]
#   ./model_swap.sh status  [port]
#
# Default port: 8000 (gpt-oss-20b)
#
# Suspend flow: POST /sleep?level=2 → SIGSTOP API → cuda-checkpoint engine
#   For gpt-oss-20b: sleep level 2 frees KV cache only (~6 GiB); weights stay
#   in VRAM. cuda-checkpoint moves the residual CUDA ctx to host RAM.
#
# Resume flow: cuda-checkpoint engine → SIGCONT API → POST /wake_up
#             → POST /reset_prefix_cache
#   wake_up re-maps KV pool allocations. reload_weights is skipped: weights
#   never left VRAM, and gpt-oss-20b has a vLLM bug in its mxfp4 reload path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${2:-8000}"
STATE_FILE="/tmp/model_swap_${PORT}.state"
CUDA_CKPT="${SCRIPT_DIR}/cuda-checkpoint"

[[ -x "$CUDA_CKPT" ]] || { echo "cuda-checkpoint not found at $CUDA_CKPT"; exit 1; }

http_post() {
    local url="$1" body="${2:-}" timeout="${3:-60}"
    if [[ -n "$body" ]]; then
        curl -sf --max-time "$timeout" -X POST "$url" \
            -H "Content-Type: application/json" -d "$body" > /dev/null
    else
        curl -sf --max-time "$timeout" -X POST "$url" > /dev/null
    fi
}

find_api_pid() {
    lsof -ti ":${PORT}" -sTCP:LISTEN 2>/dev/null | head -1
}

find_engine_pid() {
    local api_pid="$1"
    local gpu_pids
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    # Prefer a child of the API server that is using the GPU
    for child in $(pgrep -P "$api_pid" 2>/dev/null); do
        if echo "$gpu_pids" | grep -qx "$child"; then
            echo "$child"
            return 0
        fi
    done
    # Fallback: first GPU process (works when only one model is loaded)
    echo "$gpu_pids" | head -1
}

vram_mib() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

case "${1:-}" in
    suspend)
        API_PID=$(find_api_pid)
        [[ -n "$API_PID" ]] || { echo "Nothing listening on port $PORT"; exit 1; }

        # Warn if --enable-sleep-mode is missing
        if ! tr '\0' ' ' < /proc/"$API_PID"/cmdline 2>/dev/null | grep -q "enable-sleep-mode"; then
            echo "WARNING: process $API_PID was not launched with --enable-sleep-mode"
            echo "The /sleep endpoint will not work. Relaunch with that flag."
            exit 1
        fi

        ENGINE_PID=$(find_engine_pid "$API_PID")
        [[ -n "$ENGINE_PID" ]] || { echo "No GPU compute process found for API pid $API_PID"; exit 1; }

        echo "Suspending  port=$PORT  api=$API_PID  engine=$ENGINE_PID  vram=$(vram_mib) MiB"

        # Drop weights + KV from VRAM (keeps ~2 GB CUDA ctx + graphs)
        http_post "http://localhost:${PORT}/sleep?level=2"
        sleep 1

        # Freeze API server (stops accepting new requests)
        kill -STOP "$API_PID"

        # Move residual VRAM state to host RAM
        "$CUDA_CKPT" --toggle --pid "$ENGINE_PID"

        printf 'API_PID=%s\nENGINE_PID=%s\n' "$API_PID" "$ENGINE_PID" > "$STATE_FILE"

        echo "Suspended.  vram=$(vram_mib) MiB  (checkpoint in host RAM)"
        ;;

    resume)
        [[ -f "$STATE_FILE" ]] || { echo "No state at $STATE_FILE — model not suspended?"; exit 1; }
        # shellcheck source=/dev/null
        source "$STATE_FILE"
        # Delete state file immediately so a concurrent or accidental second
        # resume attempt fails fast rather than calling cuda-checkpoint --toggle
        # on an already-running engine (which would checkpoint it and kill it).
        rm -f "$STATE_FILE"

        echo "Resuming    port=$PORT  api=$API_PID  engine=$ENGINE_PID"

        # Restore CUDA ctx + graphs from host RAM to GPU
        "$CUDA_CKPT" --toggle --pid "$ENGINE_PID"

        # Unfreeze API server
        kill -CONT "$API_PID"

        # Re-map weight/KV pool allocations; wake_up restores both weights and
        # KV cache. reload_weights (disk reload) is intentionally skipped: for
        # gpt-oss-20b, sleep level 2 keeps weights in VRAM so wake_up is
        # sufficient, and the vLLM reload_weights path has a bug with mxfp4
        # models (TypeError: default_weight_loader() unexpected kwarg 'weight_name').
        http_post "http://localhost:${PORT}/wake_up"
        http_post "http://localhost:${PORT}/reset_prefix_cache"

        echo "Resumed.    vram=$(vram_mib) MiB"
        ;;

    status)
        if [[ -f "$STATE_FILE" ]]; then
            # shellcheck source=/dev/null
            source "$STATE_FILE"
            echo "SUSPENDED   port=$PORT  api=$API_PID  engine=$ENGINE_PID"
        else
            API_PID=$(find_api_pid)
            if [[ -n "$API_PID" ]]; then
                ENGINE_PID=$(find_engine_pid "$API_PID")
                echo "RUNNING     port=$PORT  api=$API_PID  engine=${ENGINE_PID:-?}  vram=$(vram_mib) MiB"
            else
                echo "NOT RUNNING port=$PORT"
            fi
        fi
        ;;

    *)
        echo "Usage: $0 {suspend|resume|status} [port]"
        echo "Default port: 8000"
        exit 1
        ;;
esac
