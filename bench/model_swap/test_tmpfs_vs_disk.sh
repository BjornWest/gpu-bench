#!/usr/bin/env bash
# Benchmark: disk vs tmpfs weight reload speed during model swaps
#
# Runs the same two-model swap cycle twice:
#   Pass 1 (DISK):  reload_weights reads from NVMe/disk
#   Pass 2 (TMPFS): reload_weights reads from tmpfs (/dev/shm)
#
# Both models are pre-staged to tmpfs before pass 2, so tmpfs reads
# have zero disk I/O. Comparison printed at the end.
#
# Usage:
#   ./test_tmpfs_vs_disk.sh [model1_path] [model1_name]
#
# Environment variables:
#   MODEL2_PATH          - path to second model (default: ../models/Qwen3-8B)
#   MODEL2_NAME          - served name for model 2 (default: Qwen/Qwen3-8B)
#   MODEL2_MAX_MODEL_LEN - max context for model 2 (default: 6000)
#
# Page cache is evicted (posix_fadvise DONTNEED) before each disk reload
# to ensure disk reads are real I/O, not cached in RAM.

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────

MODEL_PATH="${1:-../models/Qwen3-0.6B}"
MODEL_NAME="${2:-Qwen/Qwen3-0.6B}"
PORT=8199
VLLM_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CUDA_CKPT="${VLLM_DIR}/cuda-checkpoint"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$(cd "$VLLM_DIR" && uv run which python)"

MODEL2_PATH="${MODEL2_PATH:-../models/Qwen3-8B}"
MODEL2_NAME="${MODEL2_NAME:-Qwen/Qwen3-8B}"
MODEL2_MAX_MODEL_LEN="${MODEL2_MAX_MODEL_LEN:-6000}"
PORT2=8200

TMPFS_DIR="/dev/shm/test-model-weights"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ── Helpers ──────────────────────────────────────────────────────────

log()  { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $*"; }
pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; return 1; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
sep()  { echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"; }

gpu_mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }
now_ms() { echo $(( $(date +%s%N) / 1000000 )); }
elapsed_ms() { echo $(( $(now_ms) - $1 )); }
fmt_ms() { local ms=$1; echo "$((ms / 1000)).$((ms % 1000 / 100))s (${ms}ms)"; }

check_inference() {
    local port="$1" model="$2" timeout="${3:-30}"
    local response
    response=$(curl -s --max-time "$timeout" \
        "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"${model}\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in exactly 3 words.\"}], \"max_tokens\": 20}" 2>&1)
    if echo "$response" | "$VENV_PYTHON" -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>/dev/null; then
        return 0
    else
        echo "ERROR: $response" >&2
        return 1
    fi
}

poll_ready() {
    local port="$1" model="$2" timeout="${3:-120}"
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
    local url="$1" body="${2:-}" timeout="${3:-30}"
    if [[ -n "$body" ]]; then
        curl -s --max-time "$timeout" -X POST "$url" -H "Content-Type: application/json" -d "$body" 2>&1
    else
        curl -s --max-time "$timeout" -X POST "$url" 2>&1
    fi
}

drop_page_cache() {
    # Evict model files from Linux page cache using posix_fadvise(DONTNEED).
    # No root required — works on files owned by the current user.
    local model_dir="$1"
    "$VENV_PYTHON" -c "
import os, sys
d = sys.argv[1]
total = 0
for f in os.listdir(d):
    fp = os.path.join(d, f)
    if os.path.isfile(fp):
        sz = os.path.getsize(fp)
        fd = os.open(fp, os.O_RDONLY)
        os.posix_fadvise(fd, 0, sz, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        total += sz
print(f'Evicted {total / 1e9:.2f} GB from page cache ({d})')
" "$model_dir"
}

sleep_and_freeze() {
    local port="$1" api_pid="$2" engine_pid="$3"
    http_post "http://localhost:${port}/sleep?level=2" > /dev/null
    sleep 1
    kill -STOP "$api_pid"
    "$CUDA_CKPT" --toggle --pid "$engine_pid"
    sleep 1
}

restore_and_wake() {
    local port="$1" model="$2" api_pid="$3" engine_pid="$4"
    "$CUDA_CKPT" --toggle --pid "$engine_pid"
    kill -CONT "$api_pid"
    poll_ready "$port" "$model" 30
    http_post "http://localhost:${port}/wake_up" > /dev/null
    http_post "http://localhost:${port}/collective_rpc" '{"method": "reload_weights"}' 120 > /dev/null
    http_post "http://localhost:${port}/reset_prefix_cache" > /dev/null
}

shutdown_model() {
    local api_pid="$1" engine_pid="$2"
    # Attempt to unfreeze if frozen
    "$CUDA_CKPT" --toggle --pid "$engine_pid" 2>/dev/null || true
    kill -CONT "$api_pid" 2>/dev/null || true
    kill -TERM "$api_pid" 2>/dev/null || true
    wait "$api_pid" 2>/dev/null || true
}

# Tracked PIDs for cleanup
ENGINE_PID=""
ENGINE_PID2=""
VLLM_PID=""
VLLM_PID2=""

cleanup() {
    log "Cleaning up..."
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
    for epid in "$ENGINE_PID" "$ENGINE_PID2"; do
        if [[ -n "$epid" ]]; then
            "$CUDA_CKPT" --toggle --pid "$epid" 2>/dev/null || true
            kill -CONT "$epid" 2>/dev/null || true
            kill -TERM "$epid" 2>/dev/null || true
        fi
    done
    rm -rf "$TMPFS_DIR" "${SCRIPT_DIR}/m1_link" "${SCRIPT_DIR}/m2_link"
}
trap cleanup EXIT

# ── Preflight ────────────────────────────────────────────────────────

sep
log "Disk vs tmpfs reload benchmark"
sep
echo ""

[[ -x "$CUDA_CKPT" ]] || { fail "cuda-checkpoint not found: $CUDA_CKPT"; exit 1; }

# Resolve model paths
if [[ "$MODEL_PATH" != /* ]]; then MODEL_PATH="$VLLM_DIR/$MODEL_PATH"; fi
if [[ "$MODEL2_PATH" != /* ]]; then MODEL2_PATH="$VLLM_DIR/$MODEL2_PATH"; fi
[[ -d "$MODEL_PATH" ]] || { fail "Model 1 not found: $MODEL_PATH"; exit 1; }
[[ -d "$MODEL2_PATH" ]] || { fail "Model 2 not found: $MODEL2_PATH"; exit 1; }

M1_SIZE=$(du -sh "$MODEL_PATH" | cut -f1)
M2_SIZE=$(du -sh "$MODEL2_PATH" | cut -f1)
log "Model 1: $MODEL_NAME ($M1_SIZE) — $MODEL_PATH"
log "Model 2: $MODEL2_NAME ($M2_SIZE) — $MODEL2_PATH"
log "GPU baseline: $(gpu_mem) MiB"
echo ""

# ════════════════════════════════════════════════════════════════════
# PASS 1: DISK — reload weights from NVMe
# ════════════════════════════════════════════════════════════════════

sep
log "PASS 1: Swap benchmark — weights from DISK"
sep

# Evict model files from page cache so initial boot reads from actual disk
log "Evicting model files from page cache..."
drop_page_cache "$MODEL_PATH"
drop_page_cache "$MODEL2_PATH"

# ── Boot M1 from disk ─────────────────────────────────────────
log "DISK: Booting M1..."
cd "$VLLM_DIR"
T_START=$(now_ms)
VLLM_SERVER_DEV_MODE=1 "$VENV_PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --served-model-name "$MODEL_NAME" \
    --enable-sleep-mode \
    --gpu-memory-utilization 0.85 \
    > "${SCRIPT_DIR}/bench_disk_m1.log" 2>&1 &
VLLM_PID=$!

if ! poll_ready "$PORT" "$MODEL_NAME" 180; then
    fail "DISK: M1 did not become ready"; tail -20 "${SCRIPT_DIR}/bench_disk_m1.log"; exit 1
fi
DISK_T_BOOT_M1=$(elapsed_ms $T_START)
ENGINE_PID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
log "DISK: M1 ready (PID $ENGINE_PID, boot: $(fmt_ms $DISK_T_BOOT_M1))"

check_inference "$PORT" "$MODEL_NAME" 30 > /dev/null || { fail "DISK: M1 inference failed"; exit 1; }
pass "DISK: M1 inference OK"

# ── Freeze M1 ─────────────────────────────────────────────────
log "DISK: Freezing M1..."
T_START=$(now_ms)
sleep_and_freeze "$PORT" "$VLLM_PID" "$ENGINE_PID"
DISK_T_FREEZE_M1=$(elapsed_ms $T_START)
log "DISK: M1 frozen ($(fmt_ms $DISK_T_FREEZE_M1), GPU: $(gpu_mem) MiB)"

# ── Boot M2 from disk ─────────────────────────────────────────
log "DISK: Booting M2..."
T_START=$(now_ms)
VLLM_SERVER_DEV_MODE=1 "$VENV_PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL2_PATH" \
    --port "$PORT2" \
    --served-model-name "$MODEL2_NAME" \
    --enable-sleep-mode \
    --gpu-memory-utilization 0.85 \
    --max-model-len "$MODEL2_MAX_MODEL_LEN" \
    > "${SCRIPT_DIR}/bench_disk_m2.log" 2>&1 &
VLLM_PID2=$!

if ! poll_ready "$PORT2" "$MODEL2_NAME" 300; then
    fail "DISK: M2 did not become ready"; tail -20 "${SCRIPT_DIR}/bench_disk_m2.log"; exit 1
fi
DISK_T_BOOT_M2=$(elapsed_ms $T_START)
ENGINE_PID2=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
log "DISK: M2 ready (PID $ENGINE_PID2, boot: $(fmt_ms $DISK_T_BOOT_M2))"

check_inference "$PORT2" "$MODEL2_NAME" 60 > /dev/null || { fail "DISK: M2 inference failed"; exit 1; }
pass "DISK: M2 inference OK"

# ── Freeze M2 ─────────────────────────────────────────────────
log "DISK: Freezing M2..."
T_START=$(now_ms)
sleep_and_freeze "$PORT2" "$VLLM_PID2" "$ENGINE_PID2"
DISK_T_FREEZE_M2=$(elapsed_ms $T_START)
log "DISK: Both frozen (GPU: $(gpu_mem) MiB)"

# ── Swap to M1 (reload from disk) ─────────────────────────────
log "DISK: Evicting M1 from page cache before swap..."
drop_page_cache "$MODEL_PATH"
log "DISK: Swapping to M1 (reload from disk)..."
T_START=$(now_ms)
restore_and_wake "$PORT" "$MODEL_NAME" "$VLLM_PID" "$ENGINE_PID"
DISK_T_SWAP_M1=$(elapsed_ms $T_START)
log "DISK: M1 swapped in ($(fmt_ms $DISK_T_SWAP_M1))"

check_inference "$PORT" "$MODEL_NAME" 60 > /dev/null || { fail "DISK: M1 post-swap inference failed"; exit 1; }
pass "DISK: M1 swap+inference OK"

# ── Re-freeze M1, swap to M2 (reload from disk) ──────────────
log "DISK: Re-freezing M1..."
T_START=$(now_ms)
sleep_and_freeze "$PORT" "$VLLM_PID" "$ENGINE_PID"
DISK_T_REFREEZE_M1=$(elapsed_ms $T_START)

log "DISK: Evicting M2 from page cache before swap..."
drop_page_cache "$MODEL2_PATH"
log "DISK: Swapping to M2 (reload from disk)..."
T_START=$(now_ms)
restore_and_wake "$PORT2" "$MODEL2_NAME" "$VLLM_PID2" "$ENGINE_PID2"
DISK_T_SWAP_M2=$(elapsed_ms $T_START)
log "DISK: M2 swapped in ($(fmt_ms $DISK_T_SWAP_M2))"

check_inference "$PORT2" "$MODEL2_NAME" 60 > /dev/null || { fail "DISK: M2 post-swap inference failed"; exit 1; }
pass "DISK: M2 swap+inference OK"

# ── Shutdown disk pass ────────────────────────────────────────
log "DISK: Shutting down..."
shutdown_model "$VLLM_PID2" "$ENGINE_PID2"
# M1 is frozen — unfreeze first
"$CUDA_CKPT" --toggle --pid "$ENGINE_PID" 2>/dev/null || true
kill -CONT "$VLLM_PID" 2>/dev/null || true
shutdown_model "$VLLM_PID" "$ENGINE_PID"
ENGINE_PID=""
ENGINE_PID2=""
sleep 2

echo ""
log "DISK pass complete:"
echo "    M1 boot:      $(fmt_ms $DISK_T_BOOT_M1)"
echo "    M2 boot:      $(fmt_ms $DISK_T_BOOT_M2)"
echo "    M1 swap:      $(fmt_ms $DISK_T_SWAP_M1)"
echo "    M2 swap:      $(fmt_ms $DISK_T_SWAP_M2)"
echo ""

# ════════════════════════════════════════════════════════════════════
# STAGING: Copy both models to tmpfs
# ════════════════════════════════════════════════════════════════════

sep
log "Staging both models to tmpfs..."
sep

rm -rf "$TMPFS_DIR"
mkdir -p "$TMPFS_DIR"

M1_BASENAME=$(basename "$MODEL_PATH")
M2_BASENAME=$(basename "$MODEL2_PATH")
TMPFS_M1="$TMPFS_DIR/$M1_BASENAME"
TMPFS_M2="$TMPFS_DIR/$M2_BASENAME"

T_START=$(now_ms)
rsync -a "$MODEL_PATH/" "$TMPFS_M1/"
STAGE_T_M1=$(elapsed_ms $T_START)
log "M1 staged to tmpfs in $(fmt_ms $STAGE_T_M1)"

T_START=$(now_ms)
rsync -a "$MODEL2_PATH/" "$TMPFS_M2/"
STAGE_T_M2=$(elapsed_ms $T_START)
log "M2 staged to tmpfs in $(fmt_ms $STAGE_T_M2)"

log "tmpfs usage: $(df -h /dev/shm | tail -1 | awk '{print $3 " used / " $2 " total"}')"

# Create symlinks for vLLM --model
M1_LINK="${SCRIPT_DIR}/m1_link"
M2_LINK="${SCRIPT_DIR}/m2_link"
rm -f "$M1_LINK" "$M2_LINK"
ln -s "$TMPFS_M1" "$M1_LINK"
ln -s "$TMPFS_M2" "$M2_LINK"
log "Symlinks: $M1_LINK → $TMPFS_M1"
log "          $M2_LINK → $TMPFS_M2"
echo ""

# ════════════════════════════════════════════════════════════════════
# PASS 2: TMPFS — reload weights from /dev/shm
# ════════════════════════════════════════════════════════════════════

sep
log "PASS 2: Swap benchmark — weights from TMPFS"
sep

# ── Boot M1 from tmpfs ────────────────────────────────────────
log "TMPFS: Booting M1..."
cd "$VLLM_DIR"
T_START=$(now_ms)
VLLM_SERVER_DEV_MODE=1 "$VENV_PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$M1_LINK" \
    --port "$PORT" \
    --served-model-name "$MODEL_NAME" \
    --enable-sleep-mode \
    --gpu-memory-utilization 0.85 \
    > "${SCRIPT_DIR}/bench_tmpfs_m1.log" 2>&1 &
VLLM_PID=$!

if ! poll_ready "$PORT" "$MODEL_NAME" 180; then
    fail "TMPFS: M1 did not become ready"; tail -20 "${SCRIPT_DIR}/bench_tmpfs_m1.log"; exit 1
fi
TMPFS_T_BOOT_M1=$(elapsed_ms $T_START)
ENGINE_PID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
log "TMPFS: M1 ready (PID $ENGINE_PID, boot: $(fmt_ms $TMPFS_T_BOOT_M1))"

check_inference "$PORT" "$MODEL_NAME" 30 > /dev/null || { fail "TMPFS: M1 inference failed"; exit 1; }
pass "TMPFS: M1 inference OK"

# ── Freeze M1 ─────────────────────────────────────────────────
log "TMPFS: Freezing M1..."
T_START=$(now_ms)
sleep_and_freeze "$PORT" "$VLLM_PID" "$ENGINE_PID"
TMPFS_T_FREEZE_M1=$(elapsed_ms $T_START)
log "TMPFS: M1 frozen ($(fmt_ms $TMPFS_T_FREEZE_M1), GPU: $(gpu_mem) MiB)"

# ── Boot M2 from tmpfs ────────────────────────────────────────
log "TMPFS: Booting M2..."
T_START=$(now_ms)
VLLM_SERVER_DEV_MODE=1 "$VENV_PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$M2_LINK" \
    --port "$PORT2" \
    --served-model-name "$MODEL2_NAME" \
    --enable-sleep-mode \
    --gpu-memory-utilization 0.85 \
    --max-model-len "$MODEL2_MAX_MODEL_LEN" \
    > "${SCRIPT_DIR}/bench_tmpfs_m2.log" 2>&1 &
VLLM_PID2=$!

if ! poll_ready "$PORT2" "$MODEL2_NAME" 300; then
    fail "TMPFS: M2 did not become ready"; tail -20 "${SCRIPT_DIR}/bench_tmpfs_m2.log"; exit 1
fi
TMPFS_T_BOOT_M2=$(elapsed_ms $T_START)
ENGINE_PID2=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
log "TMPFS: M2 ready (PID $ENGINE_PID2, boot: $(fmt_ms $TMPFS_T_BOOT_M2))"

check_inference "$PORT2" "$MODEL2_NAME" 60 > /dev/null || { fail "TMPFS: M2 inference failed"; exit 1; }
pass "TMPFS: M2 inference OK"

# ── Freeze M2 ─────────────────────────────────────────────────
log "TMPFS: Freezing M2..."
T_START=$(now_ms)
sleep_and_freeze "$PORT2" "$VLLM_PID2" "$ENGINE_PID2"
TMPFS_T_FREEZE_M2=$(elapsed_ms $T_START)
log "TMPFS: Both frozen (GPU: $(gpu_mem) MiB)"

# ── Swap to M1 (reload from tmpfs) ────────────────────────────
log "TMPFS: Swapping to M1 (reload from tmpfs)..."
T_START=$(now_ms)
restore_and_wake "$PORT" "$MODEL_NAME" "$VLLM_PID" "$ENGINE_PID"
TMPFS_T_SWAP_M1=$(elapsed_ms $T_START)
log "TMPFS: M1 swapped in ($(fmt_ms $TMPFS_T_SWAP_M1))"

check_inference "$PORT" "$MODEL_NAME" 60 > /dev/null || { fail "TMPFS: M1 post-swap inference failed"; exit 1; }
pass "TMPFS: M1 swap+inference OK"

# ── Re-freeze M1, swap to M2 (reload from tmpfs) ─────────────
log "TMPFS: Re-freezing M1..."
T_START=$(now_ms)
sleep_and_freeze "$PORT" "$VLLM_PID" "$ENGINE_PID"
TMPFS_T_REFREEZE_M1=$(elapsed_ms $T_START)

log "TMPFS: Swapping to M2 (reload from tmpfs)..."
T_START=$(now_ms)
restore_and_wake "$PORT2" "$MODEL2_NAME" "$VLLM_PID2" "$ENGINE_PID2"
TMPFS_T_SWAP_M2=$(elapsed_ms $T_START)
log "TMPFS: M2 swapped in ($(fmt_ms $TMPFS_T_SWAP_M2))"

check_inference "$PORT2" "$MODEL2_NAME" 60 > /dev/null || { fail "TMPFS: M2 post-swap inference failed"; exit 1; }
pass "TMPFS: M2 swap+inference OK"

# ── Shutdown tmpfs pass ───────────────────────────────────────
log "TMPFS: Shutting down..."
shutdown_model "$VLLM_PID2" "$ENGINE_PID2"
"$CUDA_CKPT" --toggle --pid "$ENGINE_PID" 2>/dev/null || true
kill -CONT "$VLLM_PID" 2>/dev/null || true
shutdown_model "$VLLM_PID" "$ENGINE_PID"
ENGINE_PID=""
ENGINE_PID2=""

echo ""
log "TMPFS pass complete:"
echo "    M1 boot:      $(fmt_ms $TMPFS_T_BOOT_M1)"
echo "    M2 boot:      $(fmt_ms $TMPFS_T_BOOT_M2)"
echo "    M1 swap:      $(fmt_ms $TMPFS_T_SWAP_M1)"
echo "    M2 swap:      $(fmt_ms $TMPFS_T_SWAP_M2)"
echo ""

# ════════════════════════════════════════════════════════════════════
# COMPARISON
# ════════════════════════════════════════════════════════════════════

sep
log "COMPARISON: Disk vs tmpfs"
sep
echo ""
echo "  Model sizes: M1=$M1_SIZE, M2=$M2_SIZE"
echo "  Staging to tmpfs: M1=$(fmt_ms $STAGE_T_M1), M2=$(fmt_ms $STAGE_T_M2)"
echo ""
printf "  %-28s %20s %20s %10s\n" "" "DISK" "TMPFS" "SPEEDUP"
printf "  %-28s %20s %20s %10s\n" "---" "---" "---" "---"
printf "  %-28s %20s %20s %10s\n" \
    "M1 swap (restore+wake+load)" "$(fmt_ms $DISK_T_SWAP_M1)" "$(fmt_ms $TMPFS_T_SWAP_M1)" \
    "$(echo "scale=1; $DISK_T_SWAP_M1 * 100 / ($TMPFS_T_SWAP_M1 + 1) / 100" | bc)x"
printf "  %-28s %20s %20s %10s\n" \
    "M2 swap (restore+wake+load)" "$(fmt_ms $DISK_T_SWAP_M2)" "$(fmt_ms $TMPFS_T_SWAP_M2)" \
    "$(echo "scale=1; $DISK_T_SWAP_M2 * 100 / ($TMPFS_T_SWAP_M2 + 1) / 100" | bc)x"
printf "  %-28s %20s %20s\n" \
    "M1 cold boot" "$(fmt_ms $DISK_T_BOOT_M1)" "$(fmt_ms $TMPFS_T_BOOT_M1)"
printf "  %-28s %20s %20s\n" \
    "M2 cold boot" "$(fmt_ms $DISK_T_BOOT_M2)" "$(fmt_ms $TMPFS_T_BOOT_M2)"
echo ""
echo "  NOTE: Page cache was evicted (POSIX_FADV_DONTNEED) before each disk"
echo "  swap to force real I/O. tmpfs reads bypass disk entirely."
echo ""

pass "Benchmark complete!"
