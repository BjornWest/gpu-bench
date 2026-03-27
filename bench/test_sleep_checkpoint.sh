#!/usr/bin/env bash
# Sleep + cuda-checkpoint round-trip test for the currently active vLLM server.
#
# Sequence:
#   1. Sanity inference  →  record pre-sleep VRAM
#   2. POST /sleep?level=2       (discard weights + KV; keep CUDA ctx ~2 GB)
#   3. cuda-checkpoint --toggle  (move CUDA ctx from VRAM to host RAM)
#   4. Verify VRAM is nearly zero
#   5. cuda-checkpoint --toggle  (restore CUDA ctx to VRAM)
#   6. POST /wake_up             (re-map weight/KV pool)
#   7. POST /collective_rpc reload_weights
#   8. POST /reset_prefix_cache
#   9. Sanity inference  →  verify output matches pre-sleep
#
# Requirements:
#   - Server must be running with --enable-sleep-mode
#   - cuda-checkpoint binary must exist (see $CUDA_CKPT below)
#
# Usage:
#   bash bench/test_sleep_checkpoint.sh [port]
#   PORT=8000 bash bench/test_sleep_checkpoint.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

source "$ROOT_DIR/models.conf"

PORT="${1:-${PORT:-8000}}"
HOST="${HOST:-localhost}"

# cuda-checkpoint binary — place it in the repo root after building/downloading.
# Source: https://github.com/NVIDIA/cuda-checkpoint
# Or obtain the pre-built binary that ships with some vLLM distributions.
CUDA_CKPT="${ROOT_DIR}/cuda-checkpoint"

# ── Colour helpers ────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $*"; }
pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*" >&2; exit 1; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

now_ms()     { echo $(( $(date +%s%N) / 1000000 )); }
elapsed_ms() { echo $(( $(now_ms) - $1 )); }
fmt_ms()     { local ms=$1; printf "%d.%ds (%dms)" $((ms/1000)) $((ms%1000/100)) "$ms"; }

# ── GPU helpers ───────────────────────────────────────────────────────
vram_mib() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk '{s+=$1} END{print s}' || echo "?"
}

find_api_pid() {
    lsof -ti ":${PORT}" -sTCP:LISTEN 2>/dev/null | head -1
}

find_engine_pid() {
    local api_pid="$1"
    local gpu_pids
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    for child in $(pgrep -P "$api_pid" 2>/dev/null); do
        if echo "$gpu_pids" | grep -qx "$child"; then
            echo "$child"; return 0
        fi
    done
    # Fallback: first GPU process (works when only one model is loaded)
    echo "$gpu_pids" | head -1
}

# ── Inference helper ──────────────────────────────────────────────────
# Returns the generated text or exits non-zero on failure.
quick_inference() {
    local port="$1" model_name="$2" timeout="${3:-60}"
    curl -sf --max-time "$timeout" \
        "http://localhost:${port}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${model_name}\",\"prompt\":\"1+1=\",\"max_tokens\":4,\"temperature\":0}" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['text'].strip())"
}

http_post() {
    local url="$1" body="${2:-}" timeout="${3:-120}"
    if [[ -n "$body" ]]; then
        curl -sf --max-time "$timeout" -X POST "$url" \
            -H "Content-Type: application/json" -d "$body" > /dev/null
    else
        curl -sf --max-time "$timeout" -X POST "$url" > /dev/null
    fi
}

# ── Pre-flight checks ─────────────────────────────────────────────────
log "=== vLLM sleep + cuda-checkpoint test  port=${PORT} ==="

[[ -x "$CUDA_CKPT" ]] || fail "cuda-checkpoint not found at $CUDA_CKPT"

API_PID=$(find_api_pid)
[[ -n "$API_PID" ]] || fail "Nothing listening on port $PORT — is the server running?"

# Check --enable-sleep-mode
if ! tr '\0' ' ' < /proc/"$API_PID"/cmdline 2>/dev/null | grep -q "enable-sleep-mode"; then
    fail "Server (PID $API_PID) was not launched with --enable-sleep-mode.\nAdd that flag to models/*/serve.sh and restart."
fi

ENGINE_PID=$(find_engine_pid "$API_PID")
[[ -n "$ENGINE_PID" ]] || fail "No GPU compute process found as child of API PID $API_PID"

# Discover served model name from /v1/models
MODEL_NAME=$(curl -sf "http://localhost:${PORT}/v1/models" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
[[ -n "$MODEL_NAME" ]] || fail "Could not determine model name from /v1/models"

log "api=$API_PID  engine=$ENGINE_PID  model=$MODEL_NAME"
log "VRAM before sleep: $(vram_mib) MiB"

# ── Phase 1: pre-sleep inference ──────────────────────────────────────
log "--- Phase 1: pre-sleep inference ---"
T=$(now_ms)
PRE_OUT=$(quick_inference "$PORT" "$MODEL_NAME") \
    || fail "Pre-sleep inference failed"
pass "Pre-sleep output: '$PRE_OUT'  ($(fmt_ms $(elapsed_ms $T)))"

# ── Phase 2: sleep level 2 ────────────────────────────────────────────
log "--- Phase 2: POST /sleep?level=2 ---"
T=$(now_ms)
http_post "http://localhost:${PORT}/sleep?level=2" || fail "/sleep request failed"
sleep 1
pass "Sleep complete  $(fmt_ms $(elapsed_ms $T))  VRAM: $(vram_mib) MiB"

# ── Phase 3: freeze process to host RAM ──────────────────────────────
log "--- Phase 3: cuda-checkpoint freeze (VRAM → host RAM) ---"
T=$(now_ms)
kill -STOP "$API_PID"
"$CUDA_CKPT" --toggle --pid "$ENGINE_PID"
FROZEN_VRAM=$(vram_mib)
pass "Frozen  $(fmt_ms $(elapsed_ms $T))  VRAM: ${FROZEN_VRAM} MiB (should be ~2–10 MiB)"

# Sanity: VRAM should have dropped substantially
if [[ "$FROZEN_VRAM" =~ ^[0-9]+$ ]] && (( FROZEN_VRAM > 2000 )); then
    warn "VRAM after checkpoint is ${FROZEN_VRAM} MiB — expected <2000 MiB"
fi

# ── Phase 4: restore from host RAM ───────────────────────────────────
log "--- Phase 4: cuda-checkpoint restore (host RAM → VRAM) ---"
T=$(now_ms)
"$CUDA_CKPT" --toggle --pid "$ENGINE_PID"
kill -CONT "$API_PID"
pass "Restored  $(fmt_ms $(elapsed_ms $T))  VRAM: $(vram_mib) MiB"

# ── Phase 5: wake up + reload weights ────────────────────────────────
log "--- Phase 5: wake_up + reload_weights + reset_prefix_cache ---"
T=$(now_ms)
http_post "http://localhost:${PORT}/wake_up"             || fail "/wake_up failed"
http_post "http://localhost:${PORT}/collective_rpc" \
    '{"method":"reload_weights"}' 300                    || fail "reload_weights failed"
http_post "http://localhost:${PORT}/reset_prefix_cache"  || fail "reset_prefix_cache failed"
pass "Wake complete  $(fmt_ms $(elapsed_ms $T))  VRAM: $(vram_mib) MiB"

# ── Phase 6: post-wake inference ─────────────────────────────────────
log "--- Phase 6: post-wake inference ---"
T=$(now_ms)
POST_OUT=$(quick_inference "$PORT" "$MODEL_NAME") \
    || fail "Post-wake inference failed — server may not have recovered"
pass "Post-wake output: '$POST_OUT'  ($(fmt_ms $(elapsed_ms $T)))"

if [[ "$PRE_OUT" == "$POST_OUT" ]]; then
    pass "Output matches pre-sleep  ✓"
else
    warn "Output differs: pre='$PRE_OUT' post='$POST_OUT' (greedy, so these should match)"
fi

log "=== Test complete ==="
