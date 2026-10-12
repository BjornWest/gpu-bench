#!/usr/bin/env bash
# Test: vLLM sleep level 2 + cuda-checkpoint + wake + reload_weights
#
# Validates the core swap mechanism:
#   1. Boot vLLM with --enable-sleep-mode
#   2. Verify inference works
#   3. Sleep level 2 (discard weights + KV from VRAM)
#   4. cuda-checkpoint the sleeping process (should be tiny)
#   5. Restore via cuda-checkpoint
#   6. Wake up + reload weights (from disk initially)
#   7. Verify inference works again
#
# Usage:
#   ./test_sleep_checkpoint.sh [model_path] [model_name]
#
# Defaults to Qwen3-0.6B for fast iteration. Requires:
#   - GPU to be free (no other vLLM processes)
#   - cuda-checkpoint binary available
#
# Set TEST_TMPFS=1 to also test weight reload from tmpfs (phase 2).
# Set TEST_MULTI=1 to test with two models (phase 3).

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────

MODEL_PATH="${1:-../models/Qwen3-0.6B}"
MODEL_NAME="${2:-Qwen/Qwen3-0.6B}"
PORT=8199
VLLM_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CUDA_CKPT="${VLLM_DIR}/cuda-checkpoint"
VLLM_LOG="${VLLM_DIR}/bench/model_swap/sleep_checkpoint_vllm.log"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Use venv python directly to avoid uv lock contention when multiple vLLM processes run
VENV_PYTHON="$(cd "$VLLM_DIR" && uv run which python)"

# Second model for multi-model test
MODEL2_PATH="${MODEL2_PATH:-../models/Qwen3-8B}"
MODEL2_NAME="${MODEL2_NAME:-Qwen/Qwen3-8B}"
MODEL2_MAX_MODEL_LEN="${MODEL2_MAX_MODEL_LEN:-6000}"
PORT2=8200

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ── Helpers ──────────────────────────────────────────────────────────

log()  { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $*"; }
pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; return 1; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
sep()  { echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"; }

gpu_mem() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

# Millisecond timer helpers
now_ms() { echo $(( $(date +%s%N) / 1000000 )); }
elapsed_ms() { echo $(( $(now_ms) - $1 )); }
fmt_ms() { local ms=$1; echo "$((ms / 1000)).$((ms % 1000 / 100))s (${ms}ms)"; }

gpu_procs() {
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null
}

check_inference() {
    local port="$1"
    local model="$2"
    local timeout="${3:-30}"

    local response
    response=$(curl -s --max-time "$timeout" \
        "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${model}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in exactly 3 words.\"}],
            \"max_tokens\": 20
        }" 2>&1)

    if echo "$response" | "$VENV_PYTHON" -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>/dev/null; then
        return 0
    else
        echo "ERROR: $response" >&2
        return 1
    fi
}

poll_ready() {
    local port="$1"
    local model="$2"
    local timeout="${3:-120}"
    local deadline=$((SECONDS + timeout))

    while (( SECONDS < deadline )); do
        if curl -s --max-time 3 "http://localhost:${port}/v1/models" 2>/dev/null | \
           "$VENV_PYTHON" -c "import sys,json; d=json.load(sys.stdin); ids=[m['id'] for m in d['data']]; exit(0 if '${model}' in ids else 1)" 2>/dev/null; then
            return 0
        fi
        sleep 1
    done
    return 1
}

http_post() {
    local url="$1"
    local body="${2:-}"
    local timeout="${3:-30}"

    if [[ -n "$body" ]]; then
        curl -s --max-time "$timeout" -X POST "$url" \
            -H "Content-Type: application/json" \
            -d "$body" 2>&1
    else
        curl -s --max-time "$timeout" -X POST "$url" 2>&1
    fi
}

cleanup() {
    log "Cleaning up..."
    # Kill any vLLM processes we started on our test ports
    for p in $PORT $PORT2; do
        local pids
        pids=$(lsof -ti ":$p" 2>/dev/null || true)
        if [[ -n "$pids" ]]; then
            echo "$pids" | xargs kill -CONT 2>/dev/null || true
            echo "$pids" | xargs kill -TERM 2>/dev/null || true
            sleep 1
            echo "$pids" | xargs kill -KILL 2>/dev/null || true
        fi
    done
    # Also kill any engine cores we might have frozen
    if [[ -n "${ENGINE_PID:-}" ]]; then
        # Try to restore first (cuda-checkpoint toggle), then kill
        "$CUDA_CKPT" --toggle --pid "$ENGINE_PID" 2>/dev/null || true
        kill -CONT "$ENGINE_PID" 2>/dev/null || true
        kill -TERM "$ENGINE_PID" 2>/dev/null || true
        sleep 1
        kill -KILL "$ENGINE_PID" 2>/dev/null || true
    fi
    if [[ -n "${ENGINE_PID2:-}" ]]; then
        "$CUDA_CKPT" --toggle --pid "$ENGINE_PID2" 2>/dev/null || true
        kill -CONT "$ENGINE_PID2" 2>/dev/null || true
        kill -TERM "$ENGINE_PID2" 2>/dev/null || true
        sleep 1
        kill -KILL "$ENGINE_PID2" 2>/dev/null || true
    fi
}

trap cleanup EXIT

# ── Preflight checks ────────────────────────────────────────────────

sep
log "Sleep + cuda-checkpoint integration test"
sep
echo ""

# Check cuda-checkpoint
if [[ ! -x "$CUDA_CKPT" ]]; then
    fail "cuda-checkpoint not found at $CUDA_CKPT"
    exit 1
fi
pass "cuda-checkpoint found: $CUDA_CKPT"

# Check model exists (resolve relative paths against VLLM_DIR)
if [[ "$MODEL_PATH" != /* ]]; then
    MODEL_PATH="$VLLM_DIR/$MODEL_PATH"
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    fail "Model not found: $MODEL_PATH"
    exit 1
fi
pass "Model found: $MODEL_PATH"

# Check GPU is free
EXISTING_PROCS=$(gpu_procs)
if [[ -n "$EXISTING_PROCS" ]]; then
    warn "GPU has existing processes:"
    echo "$EXISTING_PROCS"
    echo ""
    read -rp "Continue anyway? [y/N] " yn
    [[ "$yn" == [yY]* ]] || exit 1
fi

MEM_BASELINE=$(gpu_mem)
log "GPU memory baseline: ${MEM_BASELINE} MiB"
echo ""

# ════════════════════════════════════════════════════════════════════
# TEST 1: Boot with sleep mode
# ════════════════════════════════════════════════════════════════════

sep
log "TEST 1: Boot vLLM with --enable-sleep-mode"
sep

cd "$VLLM_DIR"

VLLM_SERVER_DEV_MODE=1 "$VENV_PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --served-model-name "$MODEL_NAME" \
    --enable-sleep-mode \
    --gpu-memory-utilization 0.85 \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

log "vLLM starting (PID $VLLM_PID), polling for readiness..."

if ! poll_ready "$PORT" "$MODEL_NAME" 180; then
    fail "vLLM did not become ready within 180s"
    echo "Last 20 lines of log:"
    tail -20 "$VLLM_LOG"
    exit 1
fi

MEM_AFTER_BOOT=$(gpu_mem)
log "GPU memory after boot: ${MEM_AFTER_BOOT} MiB"

# Find engine core PID
ENGINE_PID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
if [[ -z "$ENGINE_PID" ]]; then
    fail "Could not find engine core PID"
    exit 1
fi
log "Engine core PID: $ENGINE_PID"

pass "TEST 1: vLLM booted with sleep mode (VRAM: ${MEM_AFTER_BOOT} MiB)"
echo ""

# ════════════════════════════════════════════════════════════════════
# TEST 2: Verify inference works
# ════════════════════════════════════════════════════════════════════

sep
log "TEST 2: Pre-sleep inference check"
sep

RESPONSE=$(check_inference "$PORT" "$MODEL_NAME" 30)
if [[ $? -eq 0 ]]; then
    pass "TEST 2: Inference works. Response: $RESPONSE"
else
    fail "TEST 2: Inference failed before sleep"
    exit 1
fi
echo ""

# ════════════════════════════════════════════════════════════════════
# TEST 3: Sleep level 2
# ════════════════════════════════════════════════════════════════════

sep
log "TEST 3: Sleep level 2 (discard weights + KV)"
sep

SLEEP_RESULT=$(http_post "http://localhost:${PORT}/sleep?level=2")
log "Sleep response: $SLEEP_RESULT"

sleep 2
MEM_AFTER_SLEEP=$(gpu_mem)
log "GPU memory after sleep: ${MEM_AFTER_SLEEP} MiB"

# Check that VRAM dropped significantly
MEM_FREED=$((MEM_AFTER_BOOT - MEM_AFTER_SLEEP))
if (( MEM_FREED > 100 )); then
    pass "TEST 3: Sleep freed ${MEM_FREED} MiB (${MEM_AFTER_BOOT} → ${MEM_AFTER_SLEEP} MiB)"
else
    fail "TEST 3: Sleep only freed ${MEM_FREED} MiB — expected significant reduction"
    exit 1
fi

# Check is_sleeping endpoint
IS_SLEEPING=$(curl -s "http://localhost:${PORT}/is_sleeping" 2>/dev/null)
log "is_sleeping: $IS_SLEEPING"
echo ""

# ════════════════════════════════════════════════════════════════════
# TEST 4: cuda-checkpoint the sleeping process
# ════════════════════════════════════════════════════════════════════

sep
log "TEST 4: cuda-checkpoint (freeze sleeping process)"
sep

log "SIGSTOP API server (PID $VLLM_PID)..."
kill -STOP "$VLLM_PID"

log "Running cuda-checkpoint --toggle --pid $ENGINE_PID..."
T_START=$(now_ms)
"$CUDA_CKPT" --toggle --pid "$ENGINE_PID"
T_CKPT=$(elapsed_ms $T_START)

sleep 1
MEM_AFTER_CKPT=$(gpu_mem)
log "GPU memory after checkpoint: ${MEM_AFTER_CKPT} MiB"
log "Checkpoint took: $(fmt_ms $T_CKPT)"

# Verify GPU is freed
REMAINING_PROCS=$(gpu_procs)
if [[ -z "$REMAINING_PROCS" ]]; then
    pass "TEST 4: GPU fully freed after checkpoint (${MEM_AFTER_CKPT} MiB, took $(fmt_ms $T_CKPT))"
else
    warn "TEST 4: GPU still has processes after checkpoint:"
    echo "$REMAINING_PROCS"
fi
echo ""

# ════════════════════════════════════════════════════════════════════
# TEST 4b: CRIU dump to disk (optional — requires CAP_CHECKPOINT_RESTORE)
# ════════════════════════════════════════════════════════════════════

CRIU_BIN=$(nix-shell -p criu --run "which criu" 2>/dev/null || true)
CRIU_DUMP_DIR="${SCRIPT_DIR}/criu_dump_${MODEL_NAME//\//_}"
CRIU_WORKED=false

if [[ -n "$CRIU_BIN" ]]; then
    sep
    log "TEST 4b: CRIU dump frozen process to disk"
    sep

    rm -rf "$CRIU_DUMP_DIR"
    mkdir -p "$CRIU_DUMP_DIR"

    log "CRIU dumping API server tree (PID $VLLM_PID) to $CRIU_DUMP_DIR..."
    T_START=$(now_ms)

    # CRIU dump: --leave-stopped keeps the process frozen after dump (we'll kill it to prove restore works from disk)
    # --shell-job needed if process was started from a terminal
    # --tcp-established if vLLM has open connections (probably not needed here)
    # -t targets the API server PID (which is parent of the engine core)
    if nix-shell -p criu --run "criu dump -t $VLLM_PID --leave-stopped -D $CRIU_DUMP_DIR -v0 --shell-job" 2>&1; then
        T_CRIU_DUMP=$(elapsed_ms $T_START)
        DUMP_SIZE=$(du -sh "$CRIU_DUMP_DIR" | cut -f1)
        log "CRIU dump took: $(fmt_ms $T_CRIU_DUMP), size on disk: $DUMP_SIZE"

        MEM_AFTER_CRIU=$(gpu_mem)
        log "GPU memory after CRIU dump: ${MEM_AFTER_CRIU} MiB"

        # Kill the original process to prove restore comes from disk
        log "Killing original process to prove disk restore..."
        kill -CONT "$VLLM_PID" 2>/dev/null || true
        kill -KILL "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        sleep 1

        pass "TEST 4b: CRIU dump succeeded (${DUMP_SIZE}, took $(fmt_ms $T_CRIU_DUMP))"
        CRIU_WORKED=true
    else
        warn "TEST 4b: CRIU dump failed (likely needs CAP_CHECKPOINT_RESTORE). Skipping CRIU tests."
        log "Continuing with in-memory cuda-checkpoint restore instead."
    fi
    echo ""
fi

# ════════════════════════════════════════════════════════════════════
# TEST 5: Restore (CRIU from disk if available, else cuda-checkpoint from RAM)
# ════════════════════════════════════════════════════════════════════

sep
if $CRIU_WORKED; then
    log "TEST 5: CRIU restore from disk + cuda-checkpoint restore"
else
    log "TEST 5: cuda-checkpoint restore (from host RAM)"
fi
sep

if $CRIU_WORKED; then
    log "CRIU restoring from $CRIU_DUMP_DIR..."
    T_START=$(now_ms)
    if nix-shell -p criu --run "criu restore -d -D $CRIU_DUMP_DIR -v0 --shell-job" 2>&1; then
        T_CRIU_RESTORE=$(elapsed_ms $T_START)
        log "CRIU restore took: $(fmt_ms $T_CRIU_RESTORE)"

        # After CRIU restore, the process is in the same state as when dumped:
        # - API server is SIGSTOP'd
        # - Engine core is cuda-checkpoint frozen
        # We need to: cuda-checkpoint toggle (restore GPU) → SIGCONT API server

        sleep 1
        # Re-detect engine PID (should be same PID, restored by CRIU)
        log "cuda-checkpoint restoring GPU state (PID $ENGINE_PID)..."
        "$CUDA_CKPT" --toggle --pid "$ENGINE_PID"
    else
        warn "CRIU restore failed. This test requires matching kernel capabilities."
        warn "Falling back to non-CRIU path (process was killed, cannot continue)."
        fail "TEST 5: Cannot restore — CRIU dump succeeded but restore failed"
        exit 1
    fi
else
    log "Running cuda-checkpoint --toggle --pid $ENGINE_PID (restore)..."
    T_START=$(now_ms)
    "$CUDA_CKPT" --toggle --pid "$ENGINE_PID"
fi

T_RESTORE=$(elapsed_ms $T_START)

sleep 1
MEM_AFTER_RESTORE=$(gpu_mem)
log "GPU memory after restore: ${MEM_AFTER_RESTORE} MiB"
log "Restore took: $(fmt_ms $T_RESTORE)"

# SIGCONT the API server
log "SIGCONT API server (PID $VLLM_PID)..."
kill -CONT "$VLLM_PID"

# Wait for API server to respond
if ! poll_ready "$PORT" "$MODEL_NAME" 30; then
    fail "TEST 5: API server did not respond after restore"
    exit 1
fi

if $CRIU_WORKED; then
    pass "TEST 5: Restored from CRIU disk dump (VRAM: ${MEM_AFTER_RESTORE} MiB, CRIU restore: $(fmt_ms $T_CRIU_RESTORE), total: $(fmt_ms $T_RESTORE))"
else
    pass "TEST 5: Restored from checkpoint (VRAM: ${MEM_AFTER_RESTORE} MiB, took $(fmt_ms $T_RESTORE))"
fi
echo ""

# ════════════════════════════════════════════════════════════════════
# TEST 6: Wake up + reload weights
# ════════════════════════════════════════════════════════════════════

sep
log "TEST 6: Wake up + reload weights from disk"
sep

log "Calling /wake_up..."
T_START=$(now_ms)
WAKE_RESULT=$(http_post "http://localhost:${PORT}/wake_up")
log "wake_up response: $WAKE_RESULT"

log "Calling /collective_rpc reload_weights..."
RELOAD_RESULT=$(http_post "http://localhost:${PORT}/collective_rpc" \
    '{"method": "reload_weights"}' 120)
T_RELOAD=$(elapsed_ms $T_START)
log "reload_weights response: $RELOAD_RESULT"

log "Calling /reset_prefix_cache..."
RESET_RESULT=$(http_post "http://localhost:${PORT}/reset_prefix_cache")
log "reset_prefix_cache response: $RESET_RESULT"

sleep 1
MEM_AFTER_WAKE=$(gpu_mem)
log "GPU memory after wake: ${MEM_AFTER_WAKE} MiB"
log "Wake + reload took: $(fmt_ms $T_RELOAD)"

pass "TEST 6: Wake complete (VRAM: ${MEM_AFTER_WAKE} MiB, took $(fmt_ms $T_RELOAD))"
echo ""

# ════════════════════════════════════════════════════════════════════
# TEST 7: Verify inference works after full cycle
# ════════════════════════════════════════════════════════════════════

sep
log "TEST 7: Post-restore inference check"
sep

RESPONSE=$(check_inference "$PORT" "$MODEL_NAME" 60)
if [[ $? -eq 0 ]]; then
    pass "TEST 7: Inference works after full cycle! Response: $RESPONSE"
else
    fail "TEST 7: Inference FAILED after sleep+checkpoint+restore+wake cycle"
    echo "Last 30 lines of vLLM log:"
    tail -30 "$VLLM_LOG"
    exit 1
fi
echo ""

# ════════════════════════════════════════════════════════════════════
# PHASE 2: Boot from tmpfs, full swap cycle simulation
# ════════════════════════════════════════════════════════════════════
#
# Simulates the real DGX flow:
#   1. Stage weights to tmpfs
#   2. Boot vLLM from tmpfs (via symlink)
#   3. Verify inference
#   4. Sleep + checkpoint (model is "swapped out")
#   5. Remove weights from tmpfs (another model would use this space)
#   6. Copy weights back to tmpfs (simulates pre-staging for next round)
#   7. Restore + wake + reload (should read from tmpfs)
#   8. Verify inference again

if [[ "${TEST_TMPFS:-0}" == "1" ]]; then

    # First, shut down the process from phase 1
    log "Shutting down phase 1 vLLM process..."
    kill -TERM "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    ENGINE_PID=""
    sleep 2

    TMPFS_DIR="/dev/shm/test-model-weights"
    MODEL_BASENAME=$(basename "$MODEL_PATH")
    TMPFS_MODEL="$TMPFS_DIR/$MODEL_BASENAME"
    # Symlink that vLLM's --model will point to
    SYMLINK_DIR="${SCRIPT_DIR}/tmpfs_model_link"
    PORT_TMPFS=8198

    sep
    log "PHASE 2: Boot from tmpfs, full swap cycle"
    sep

    # ── Step 1: Stage weights to tmpfs ────────────────────────

    log "Step 1: Copying model to tmpfs..."
    rm -rf "$TMPFS_DIR" "$SYMLINK_DIR"
    mkdir -p "$TMPFS_DIR"

    T_START=$(now_ms)
    rsync -a "$MODEL_PATH/" "$TMPFS_MODEL/"
    T_STAGE=$(elapsed_ms $T_START)
    TMPFS_SIZE=$(du -sh "$TMPFS_MODEL" | cut -f1)
    log "Staged to tmpfs in $(fmt_ms $T_STAGE) (size: $TMPFS_SIZE)"

    # Create symlink: vLLM --model points here, we control where it resolves
    ln -s "$TMPFS_MODEL" "$SYMLINK_DIR"
    log "Symlink: $SYMLINK_DIR → $TMPFS_MODEL"

    # ── Step 2: Boot vLLM from tmpfs (via symlink) ────────────

    log "Step 2: Booting vLLM from tmpfs..."
    VLLM_LOG_TMPFS="${SCRIPT_DIR}/sleep_checkpoint_vllm_tmpfs.log"

    cd "$VLLM_DIR"
    VLLM_SERVER_DEV_MODE=1 "$VENV_PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$SYMLINK_DIR" \
        --port "$PORT_TMPFS" \
        --served-model-name "$MODEL_NAME" \
        --enable-sleep-mode \
        --gpu-memory-utilization 0.85 \
        > "$VLLM_LOG_TMPFS" 2>&1 &
    VLLM_PID=$!

    if ! poll_ready "$PORT_TMPFS" "$MODEL_NAME" 180; then
        fail "PHASE 2: vLLM did not become ready"
        tail -20 "$VLLM_LOG_TMPFS"
        exit 1
    fi

    ENGINE_PID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    MEM_TMPFS_BOOT=$(gpu_mem)
    log "Booted from tmpfs (engine PID $ENGINE_PID, VRAM: ${MEM_TMPFS_BOOT} MiB)"

    # ── Step 3: Verify inference ──────────────────────────────

    log "Step 3: Pre-swap inference check..."
    RESPONSE=$(check_inference "$PORT_TMPFS" "$MODEL_NAME" 30)
    if [[ $? -eq 0 ]]; then
        pass "Inference works (booted from tmpfs). Response: $RESPONSE"
    else
        fail "PHASE 2: Inference failed after tmpfs boot"
        exit 1
    fi

    # ── Step 4: Sleep + checkpoint ────────────────────────────

    log "Step 4: Sleep level 2 + cuda-checkpoint..."
    http_post "http://localhost:${PORT_TMPFS}/sleep?level=2" > /dev/null
    sleep 1

    MEM_TMPFS_SLEEP=$(gpu_mem)
    log "After sleep: ${MEM_TMPFS_SLEEP} MiB"

    kill -STOP "$VLLM_PID"
    "$CUDA_CKPT" --toggle --pid "$ENGINE_PID"
    sleep 1

    MEM_TMPFS_CKPT=$(gpu_mem)
    log "After checkpoint: ${MEM_TMPFS_CKPT} MiB"

    # ── Step 5: Clear tmpfs (simulate another model using the space) ──

    log "Step 5: Removing model from tmpfs (simulating another model's turn)..."
    rm -rf "$TMPFS_MODEL"
    ls "$SYMLINK_DIR" 2>&1 | head -1  # Should show broken symlink / error
    log "tmpfs cleared. Symlink is now dangling."
    log "tmpfs usage: $(df -h /dev/shm | tail -1 | awk '{print $3 " used / " $2 " total"}')"

    # ── Step 6: Re-stage weights to tmpfs (simulate pre-staging) ──

    log "Step 6: Re-staging model to tmpfs (simulating pre-stage for next round)..."
    T_START=$(now_ms)
    rsync -a "$MODEL_PATH/" "$TMPFS_MODEL/"
    T_RESTAGE=$(elapsed_ms $T_START)
    log "Re-staged in $(fmt_ms $T_RESTAGE). Symlink should be valid again."
    ls "$SYMLINK_DIR" > /dev/null 2>&1 && log "Symlink resolves OK" || warn "Symlink still broken!"

    # ── Step 7: Restore + wake + reload ───────────────────────

    log "Step 7: cuda-checkpoint restore..."
    T_START=$(now_ms)
    "$CUDA_CKPT" --toggle --pid "$ENGINE_PID"
    sleep 1
    kill -CONT "$VLLM_PID"

    if ! poll_ready "$PORT_TMPFS" "$MODEL_NAME" 30; then
        fail "PHASE 2: API server did not respond after restore"
        exit 1
    fi

    log "Calling /wake_up..."
    http_post "http://localhost:${PORT_TMPFS}/wake_up" > /dev/null

    log "Calling /collective_rpc reload_weights..."
    RELOAD_RESULT=$(http_post "http://localhost:${PORT_TMPFS}/collective_rpc" \
        '{"method": "reload_weights"}' 120)
    log "reload_weights response: $RELOAD_RESULT"

    log "Calling /reset_prefix_cache..."
    http_post "http://localhost:${PORT_TMPFS}/reset_prefix_cache" > /dev/null

    T_TMPFS_RELOAD=$(elapsed_ms $T_START)
    MEM_TMPFS_WAKE=$(gpu_mem)
    log "After wake+reload from tmpfs: ${MEM_TMPFS_WAKE} MiB (took $(fmt_ms $T_TMPFS_RELOAD))"

    # ── Step 8: Verify inference after full cycle ─────────────

    log "Step 8: Post-swap inference check..."
    RESPONSE=$(check_inference "$PORT_TMPFS" "$MODEL_NAME" 60)
    if [[ $? -eq 0 ]]; then
        pass "PHASE 2: Full tmpfs swap cycle works! Response: $RESPONSE"
    else
        fail "PHASE 2: Inference FAILED after tmpfs swap cycle"
        tail -30 "$VLLM_LOG_TMPFS"
        exit 1
    fi

    echo ""
    log "PHASE 2 Memory Timeline:"
    echo "    Boot from tmpfs:     ${MEM_TMPFS_BOOT} MiB"
    echo "    After sleep L2:      ${MEM_TMPFS_SLEEP} MiB"
    echo "    After checkpoint:    ${MEM_TMPFS_CKPT} MiB"
    echo "    After wake+reload:   ${MEM_TMPFS_WAKE} MiB"
    echo "    Staging time:        $(fmt_ms $T_STAGE) (initial), $(fmt_ms $T_RESTAGE) (re-stage)"
    echo "    Wake+reload time:    $(fmt_ms $T_TMPFS_RELOAD)"
    echo ""

    # Cleanup
    rm -rf "$TMPFS_DIR" "$SYMLINK_DIR"

    # Shut down for phase 3
    log "Shutting down phase 2 vLLM process..."
    kill -TERM "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    ENGINE_PID=""
    sleep 2
fi

# ════════════════════════════════════════════════════════════════════
# PHASE 3: Multi-model test (optional)
# ════════════════════════════════════════════════════════════════════

if [[ "${TEST_MULTI:-0}" == "1" ]]; then
    sep
    log "PHASE 3: Two-model swap test"
    sep

    # Resolve model 2 path
    if [[ "$MODEL2_PATH" != /* ]]; then
        MODEL2_PATH="$VLLM_DIR/$MODEL2_PATH"
    fi
    if [[ ! -d "$MODEL2_PATH" ]]; then
        fail "Model 2 not found: $MODEL2_PATH"
        exit 1
    fi
    log "Model 2: $MODEL2_PATH ($MODEL2_NAME)"

    # ── Boot model 1 fresh ────────────────────────────────────
    log "Booting model 1: $MODEL_NAME"
    VLLM_LOG_M1="${SCRIPT_DIR}/sleep_checkpoint_vllm_m1.log"

    cd "$VLLM_DIR"
    T_START=$(now_ms)
    VLLM_SERVER_DEV_MODE=1 "$VENV_PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --served-model-name "$MODEL_NAME" \
        --enable-sleep-mode \
        --gpu-memory-utilization 0.85 \
        > "$VLLM_LOG_M1" 2>&1 &
    VLLM_PID=$!

    if ! poll_ready "$PORT" "$MODEL_NAME" 180; then
        fail "Model 1 did not become ready"
        tail -20 "$VLLM_LOG_M1"
        exit 1
    fi
    T_BOOT_M1=$(elapsed_ms $T_START)

    ENGINE_PID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    MEM_M1_BOOT=$(gpu_mem)
    log "Model 1 ready (engine PID $ENGINE_PID, VRAM: ${MEM_M1_BOOT} MiB, boot: $(fmt_ms $T_BOOT_M1))"

    # Verify model 1 inference
    RESPONSE=$(check_inference "$PORT" "$MODEL_NAME" 30)
    if [[ $? -eq 0 ]]; then
        pass "Model 1 inference works: $RESPONSE"
    else
        fail "Model 1 inference failed"
        exit 1
    fi

    # ── Sleep+freeze model 1 ──────────────────────────────────
    log "Sleeping + freezing model 1..."
    T_START=$(now_ms)
    http_post "http://localhost:${PORT}/sleep?level=2" > /dev/null
    sleep 1
    kill -STOP "$VLLM_PID"
    "$CUDA_CKPT" --toggle --pid "$ENGINE_PID"
    T_FREEZE_M1=$(elapsed_ms $T_START)
    sleep 1

    MEM_AFTER_FREEZE1=$(gpu_mem)
    log "GPU after freezing model 1: ${MEM_AFTER_FREEZE1} MiB (freeze: $(fmt_ms $T_FREEZE_M1))"

    # ── Cold-start model 2 ────────────────────────────────────
    log "Cold-starting model 2: $MODEL2_NAME"
    VLLM_LOG2="${SCRIPT_DIR}/sleep_checkpoint_vllm2.log"

    T_START=$(now_ms)
    VLLM_SERVER_DEV_MODE=1 "$VENV_PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL2_PATH" \
        --port "$PORT2" \
        --served-model-name "$MODEL2_NAME" \
        --enable-sleep-mode \
        --gpu-memory-utilization 0.85 \
        --max-model-len "$MODEL2_MAX_MODEL_LEN" \
        > "$VLLM_LOG2" 2>&1 &
    VLLM_PID2=$!

    if ! poll_ready "$PORT2" "$MODEL2_NAME" 300; then
        fail "Model 2 did not become ready"
        tail -20 "$VLLM_LOG2"
        exit 1
    fi
    T_BOOT_M2=$(elapsed_ms $T_START)

    ENGINE_PID2=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    MEM_M2_BOOT=$(gpu_mem)
    log "Model 2 ready (engine PID $ENGINE_PID2, VRAM: ${MEM_M2_BOOT} MiB, boot: $(fmt_ms $T_BOOT_M2))"

    # Verify model 2 inference
    RESPONSE2=$(check_inference "$PORT2" "$MODEL2_NAME" 60)
    if [[ $? -eq 0 ]]; then
        pass "Model 2 inference works: $RESPONSE2"
    else
        fail "Model 2 inference failed"
        exit 1
    fi

    # ── Sleep+freeze model 2 ──────────────────────────────────
    log "Sleeping + freezing model 2..."
    T_START=$(now_ms)
    http_post "http://localhost:${PORT2}/sleep?level=2" > /dev/null
    sleep 1
    kill -STOP "$VLLM_PID2"
    "$CUDA_CKPT" --toggle --pid "$ENGINE_PID2"
    T_FREEZE_M2=$(elapsed_ms $T_START)
    sleep 1

    MEM_AFTER_FREEZE2=$(gpu_mem)
    log "GPU after freezing both models: ${MEM_AFTER_FREEZE2} MiB (freeze M2: $(fmt_ms $T_FREEZE_M2))"

    # ── Swap: restore model 1 ─────────────────────────────────
    log "Restoring model 1..."
    T_START=$(now_ms)
    "$CUDA_CKPT" --toggle --pid "$ENGINE_PID"
    kill -CONT "$VLLM_PID"
    poll_ready "$PORT" "$MODEL_NAME" 30
    http_post "http://localhost:${PORT}/wake_up" > /dev/null
    http_post "http://localhost:${PORT}/collective_rpc" '{"method": "reload_weights"}' 120 > /dev/null
    http_post "http://localhost:${PORT}/reset_prefix_cache" > /dev/null
    T_SWAP_TO_M1=$(elapsed_ms $T_START)

    MEM_M1_RESTORED=$(gpu_mem)

    RESPONSE=$(check_inference "$PORT" "$MODEL_NAME" 60)
    if [[ $? -eq 0 ]]; then
        pass "Model 1 restored ($(fmt_ms $T_SWAP_TO_M1), VRAM: ${MEM_M1_RESTORED} MiB). Response: $RESPONSE"
    else
        fail "PHASE 3: Model 1 inference failed after swap cycle"
        exit 1
    fi

    # ── Re-freeze model 1, restore model 2 ────────────────────
    log "Sleeping model 1 again, restoring model 2..."
    T_START=$(now_ms)
    http_post "http://localhost:${PORT}/sleep?level=2" > /dev/null
    sleep 1
    kill -STOP "$VLLM_PID"
    "$CUDA_CKPT" --toggle --pid "$ENGINE_PID"
    T_REFREEZE_M1=$(elapsed_ms $T_START)

    sleep 1

    T_START=$(now_ms)
    "$CUDA_CKPT" --toggle --pid "$ENGINE_PID2"
    kill -CONT "$VLLM_PID2"
    poll_ready "$PORT2" "$MODEL2_NAME" 30
    http_post "http://localhost:${PORT2}/wake_up" > /dev/null
    http_post "http://localhost:${PORT2}/collective_rpc" '{"method": "reload_weights"}' 120 > /dev/null
    http_post "http://localhost:${PORT2}/reset_prefix_cache" > /dev/null
    T_SWAP_TO_M2=$(elapsed_ms $T_START)

    MEM_M2_RESTORED=$(gpu_mem)

    RESPONSE2=$(check_inference "$PORT2" "$MODEL2_NAME" 60)
    if [[ $? -eq 0 ]]; then
        pass "Model 2 restored ($(fmt_ms $T_SWAP_TO_M2), VRAM: ${MEM_M2_RESTORED} MiB). Response: $RESPONSE2"
    else
        fail "PHASE 3: Model 2 inference failed after swap"
        exit 1
    fi

    echo ""
    log "PHASE 3 Timings:"
    echo "    Model 1 cold boot:     $(fmt_ms $T_BOOT_M1)"
    echo "    Model 1 freeze:        $(fmt_ms $T_FREEZE_M1)"
    echo "    Model 2 cold boot:     $(fmt_ms $T_BOOT_M2)"
    echo "    Model 2 freeze:        $(fmt_ms $T_FREEZE_M2)"
    echo "    Swap to model 1:       $(fmt_ms $T_SWAP_TO_M1) (restore+wake+reload)"
    echo "    Re-freeze model 1:     $(fmt_ms $T_REFREEZE_M1)"
    echo "    Swap to model 2:       $(fmt_ms $T_SWAP_TO_M2) (restore+wake+reload)"
    echo ""
    log "PHASE 3 VRAM:"
    echo "    Model 1 booted:        ${MEM_M1_BOOT} MiB"
    echo "    After freeze M1:       ${MEM_AFTER_FREEZE1} MiB"
    echo "    Model 2 booted:        ${MEM_M2_BOOT} MiB"
    echo "    After freeze both:     ${MEM_AFTER_FREEZE2} MiB"
    echo "    Model 1 restored:      ${MEM_M1_RESTORED} MiB"
    echo "    Model 2 restored:      ${MEM_M2_RESTORED} MiB"
    echo ""
fi

# ════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════

sep
log "SUMMARY"
sep
echo ""
echo "  GPU Memory Timeline:"
echo "    Baseline:          ${MEM_BASELINE} MiB"
echo "    After boot:        ${MEM_AFTER_BOOT} MiB"
echo "    After sleep L2:    ${MEM_AFTER_SLEEP} MiB  (residual: $((MEM_AFTER_SLEEP - MEM_BASELINE)) MiB)"
echo "    After checkpoint:  ${MEM_AFTER_CKPT} MiB"
echo "    After restore:     ${MEM_AFTER_RESTORE} MiB"
echo "    After wake+reload: ${MEM_AFTER_WAKE} MiB"
echo ""
echo "  Timings:"
echo "    cuda-checkpoint:   $(fmt_ms $T_CKPT) (freeze)"
if $CRIU_WORKED; then
echo "    CRIU dump:         $(fmt_ms $T_CRIU_DUMP) (to disk, size: ${DUMP_SIZE})"
echo "    CRIU restore:      $(fmt_ms $T_CRIU_RESTORE) (from disk)"
fi
echo "    Restore total:     $(fmt_ms $T_RESTORE)"
echo "    Wake + reload:     $(fmt_ms $T_RELOAD) (disk)"
[[ "${TEST_TMPFS:-0}" == "1" ]] && echo "    Wake + reload:     $(fmt_ms $T_TMPFS_RELOAD) (tmpfs)"
echo ""
echo "  Key observations:"
echo "    Sleep residual:    $((MEM_AFTER_SLEEP - MEM_BASELINE)) MiB (CUDA context + graphs)"
echo "    Checkpoint freed:  $((MEM_AFTER_SLEEP - MEM_AFTER_CKPT)) MiB"
echo "    VRAM recovered:    $((MEM_AFTER_WAKE - MEM_AFTER_CKPT)) MiB (after full restore)"
echo ""

pass "All tests passed!"
log "vLLM log: $VLLM_LOG"
