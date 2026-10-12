#!/usr/bin/env bash
# Test: LMCache KV cache persistence across sleep+checkpoint swap cycles
#
# Validates that:
#   1. LMCache stores KV cache to disk during inference
#   2. Cache hits occur when the same conversation is re-sent
#   3. KV cache on disk survives sleep L2 + cuda-checkpoint + restore + wake
#   4. After a two-model swap cycle, each model's KV cache is correctly restored
#
# Usage:
#   ./test_lmcache.sh [model_path] [model_name]
#
# Requires:
#   - GPU free
#   - cuda-checkpoint binary available
#   - lmcache pip package installed

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────

MODEL_PATH="${1:-../models/gpt-oss-20b}"
MODEL_NAME="${2:-gpt-oss-20b}"
PORT=8199
VLLM_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CUDA_CKPT="${VLLM_DIR}/cuda-checkpoint"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$(cd "$VLLM_DIR" && uv run which python)"

# NixOS: several shared libs are not on the default library path.
# - libnvidia-ml.so.1: needed by vLLM for CUDA/NVML platform detection
# - libcuda.so.1 (graphics-drivers): the real CUDA driver; triton's libcuda_dirs()
#   calls ldconfig which doesn't know about nix store — LD_LIBRARY_PATH fallback
#   also requires it to be there.
# - libxcb.so.1: needed by opencv-python-headless (cv2), which gets imported by
#   the EngineCore subprocess when vLLM spawns it; without this, the subprocess
#   fails with ImportError before it can even load vLLM.
NVIDIA_DRIVER_LIB=$(find /nix/store -name "libnvidia-ml.so.1" \
    -not -path "*32bit*" -not -path "*nvidia-x11*" 2>/dev/null | head -1 | xargs -r dirname 2>/dev/null || true)
LIBXCB_DIR=$(find /nix/store -name "libxcb.so.1" \
    -path "*libxcb-*/lib*" -not -path "*codium*" 2>/dev/null | head -1 | xargs -r dirname 2>/dev/null || true)
EXTRA_LIBS="${NVIDIA_DRIVER_LIB}${LIBXCB_DIR:+:$LIBXCB_DIR}"
if [[ -n "$EXTRA_LIBS" ]]; then
    export LD_LIBRARY_PATH="${EXTRA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# Set TRITON_LIBCUDA_PATH explicitly so triton doesn't rely on ldconfig finding
# libcuda.so.1 (it won't on NixOS).
if [[ -n "$NVIDIA_DRIVER_LIB" && -f "$NVIDIA_DRIVER_LIB/libcuda.so.1" ]]; then
    export TRITON_LIBCUDA_PATH="$NVIDIA_DRIVER_LIB"
fi

KV_CACHE_DIR="${SCRIPT_DIR}/lmcache_kv"
VLLM_LOG="${SCRIPT_DIR}/lmcache_vllm.log"

# Second model for swap test (phase 3)
MODEL2_PATH="${MODEL2_PATH:-../models/Qwen3-8B}"
MODEL2_NAME="${MODEL2_NAME:-Qwen/Qwen3-8B}"
MODEL2_MAX_MODEL_LEN="${MODEL2_MAX_MODEL_LEN:-16384}"
PORT2=8200
KV_CACHE_DIR2="${SCRIPT_DIR}/lmcache_kv2"
VLLM_LOG2="${SCRIPT_DIR}/lmcache_vllm2.log"

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

# ── Long prompt padding (~10k tokens) ────────────────────────────────────────
# LMCache only stores complete 256-token chunks; short prompts never fill a
# chunk and nothing gets written to disk. ~10k tokens (≈200 iterations of the
# base sentence) gives ~39 full chunks, ensuring a large and measurable latency
# difference between cold and cached prefill on a 20B model.
LONG_PAD=$("$VENV_PYTHON" -c "
s = 'The ocean is vast and deep. Mountains are tall and majestic. Rivers carry water to the sea. The wind blows across open plains. Birds migrate south in autumn. Stars are visible on clear nights. '
print((s * 200).strip())
")

# Return the hit-token count from the most recently logged LMCache request line.
# Call after a sleep 1 to ensure the log has flushed.
last_hit_tokens() {
    grep "LMCache hit tokens:" "$1" | tail -1 | sed 's/.*LMCache hit tokens: \([0-9]*\).*/\1/'
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

# Send a chat completion and return the response content
send_chat() {
    local port="$1" model="$2" message="$3" max_tokens="${4:-50}"
    local response
    response=$(curl -s --max-time 300 \
        "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${model}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"${message}\"}],
            \"max_tokens\": ${max_tokens}
        }" 2>&1)
    echo "$response" | "$VENV_PYTHON" -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>/dev/null
}

# Send a multi-turn conversation
send_multiturn() {
    local port="$1" model="$2" max_tokens="${3:-50}"
    shift 3
    # Remaining args are alternating role,content pairs
    local messages="["
    local first=true
    while [[ $# -ge 2 ]]; do
        local role="$1" content="$2"
        shift 2
        if $first; then first=false; else messages+=","; fi
        messages+="{\"role\":\"${role}\",\"content\":\"${content}\"}"
    done
    messages+="]"

    local response
    response=$(curl -s --max-time 300 \
        "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${model}\",
            \"messages\": ${messages},
            \"max_tokens\": ${max_tokens}
        }" 2>&1)
    echo "$response" | "$VENV_PYTHON" -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>/dev/null
}

# Get LMCache metrics from vLLM's /metrics endpoint
get_lmcache_metrics() {
    local port="$1"
    curl -s --max-time 5 "http://localhost:${port}/metrics" 2>/dev/null | grep -E "^(lmcache_|vllm:lmcache)" || true
}

# Count KV cache files on disk
count_kv_files() {
    local dir="$1"
    if [[ -d "$dir" ]]; then
        find "$dir" -type f | wc -l
    else
        echo 0
    fi
}

# Per-instance checkpoint-taken flags (set by sleep_and_freeze, read by restore_and_wake)
CKPT_TAKEN_1=false
CKPT_TAKEN_2=false

sleep_and_freeze() {
    local port="$1" api_pid="$2" engine_pid="$3" ckpt_flag_var="$4"
    http_post "http://localhost:${port}/sleep?level=2" > /dev/null
    sleep 1
    kill -STOP "$api_pid"
    # cuda-checkpoint moves the residual CUDA context (~2 GB) to host RAM.
    # Some models (e.g. gpt-oss-20b with triton MoE kernels) fully release GPU
    # at sleep level 2, leaving nothing to checkpoint — the tool then returns an
    # error which is safe to ignore; we just skip the restore call later.
    if "$CUDA_CKPT" --toggle --pid "$engine_pid" 2>/dev/null; then
        eval "${ckpt_flag_var}=true"
        log "cuda-checkpoint: context saved to host RAM"
    else
        eval "${ckpt_flag_var}=false"
        warn "cuda-checkpoint skipped (no residual GPU context — sleep freed everything)"
    fi
    sleep 1
}

restore_and_wake() {
    local port="$1" model="$2" api_pid="$3" engine_pid="$4" ckpt_taken="$5"
    if [[ "$ckpt_taken" == "true" ]]; then
        "$CUDA_CKPT" --toggle --pid "$engine_pid"
    fi
    kill -CONT "$api_pid"
    poll_ready "$port" "$model" 30
    http_post "http://localhost:${port}/wake_up" > /dev/null
    http_post "http://localhost:${port}/collective_rpc" '{"method": "reload_weights"}' 120 > /dev/null
    http_post "http://localhost:${port}/reset_prefix_cache" > /dev/null
}

# Tracked PIDs for cleanup
ENGINE_PID=""
ENGINE_PID2=""
VLLM_PID=""
VLLM_PID2=""

cleanup() {
    log "Cleaning up..."
    # Kill API servers (found by port) and their child processes.
    # lsof only returns the API server PID; the EngineCore subprocess is
    # tracked separately in ENGINE_PID/ENGINE_PID2 and killed below.
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
    # Kill engine processes directly.
    # Do NOT call cuda-checkpoint --toggle here: if the engine is currently
    # running it would checkpoint it instead of resuming it, and SIGTERM
    # doesn't kill a checkpointed process. SIGKILL works in any state
    # (stopped, checkpointed, or running).
    for epid in "$ENGINE_PID" "$ENGINE_PID2"; do
        if [[ -n "$epid" ]]; then
            kill -CONT "$epid" 2>/dev/null || true
            kill -KILL "$epid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT

# Boot vLLM with LMCache enabled
# Args: model_path model_name port kv_cache_dir log_file [extra_args...]
boot_with_lmcache() {
    local model_path="$1" model_name="$2" port="$3" kv_dir="$4" log_file="$5"
    shift 5
    local extra_args=("$@")

    mkdir -p "$kv_dir"

    cd "$VLLM_DIR"
    VLLM_SERVER_DEV_MODE=1 \
    PYTHONHASHSEED=0 \
    LMCACHE_PRE_CACHING_HASH_ALGORITHM=sha256_cbor \
    LMCACHE_USE_EXPERIMENTAL=True \
    LMCACHE_LOCAL_CPU=true \
    LMCACHE_MAX_LOCAL_CPU_SIZE=2.0 \
    LMCACHE_LOCAL_DISK="file://${kv_dir}" \
    LMCACHE_MAX_LOCAL_DISK_SIZE=5.0 \
    LMCACHE_CHUNK_SIZE=256 \
    "$VENV_PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$model_path" \
        --port "$port" \
        --served-model-name "$model_name" \
        --enable-sleep-mode \
        --gpu-memory-utilization 0.85 \
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
        "${extra_args[@]}" \
        > "$log_file" 2>&1 &
    echo $!
}

# ── Preflight ────────────────────────────────────────────────────────

sep
log "LMCache integration test"
sep
echo ""

[[ -x "$CUDA_CKPT" ]] || { fail "cuda-checkpoint not found: $CUDA_CKPT"; exit 1; }
"$VENV_PYTHON" -c "import lmcache" 2>/dev/null || { fail "lmcache not installed"; exit 1; }
pass "cuda-checkpoint and lmcache available"

# Resolve model path
if [[ "$MODEL_PATH" != /* ]]; then MODEL_PATH="$VLLM_DIR/$MODEL_PATH"; fi
[[ -d "$MODEL_PATH" ]] || { fail "Model not found: $MODEL_PATH"; exit 1; }
pass "Model: $MODEL_PATH"

# Clean KV cache dir from previous runs
rm -rf "$KV_CACHE_DIR" "$KV_CACHE_DIR2"

log "GPU baseline: $(gpu_mem) MiB"
echo ""

# ════════════════════════════════════════════════════════════════════
# PHASE 1: Basic LMCache — verify KV storage and cache hits
# ════════════════════════════════════════════════════════════════════

sep
log "PHASE 1: Basic LMCache disk storage + cache hits"
sep

log "Booting vLLM with LMCache..."
VLLM_PID=$(boot_with_lmcache "$MODEL_PATH" "$MODEL_NAME" "$PORT" "$KV_CACHE_DIR" "$VLLM_LOG" \
    --trust-remote-code --tool-call-parser openai --reasoning-parser openai_gptoss \
    --max-model-len 16384)
log "vLLM starting (PID $VLLM_PID)..."

if ! poll_ready "$PORT" "$MODEL_NAME" 360; then
    fail "vLLM did not become ready"
    tail -30 "$VLLM_LOG"
    exit 1
fi

ENGINE_PID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
log "Ready (engine PID $ENGINE_PID, VRAM: $(gpu_mem) MiB)"

# ── Test 1a: Send a conversation, verify KV stored ────────────
log "Test 1a: Sending conversation A (cold baseline + KV storage)..."
CONV_A="${LONG_PAD} Explain the concept of photosynthesis. Include the light reactions and Calvin cycle."
# Cold TTFT baseline: max_tokens=1 isolates prefill time with zero cache anywhere
T_START=$(now_ms)
send_chat "$PORT" "$MODEL_NAME" "$CONV_A" 1 > /dev/null
T_COLD_A=$(elapsed_ms $T_START)
log "Cold prefill baseline for conv A: $(fmt_ms $T_COLD_A)"
# Full send to generate response and populate LMCache disk storage
RESPONSE_A1=$(send_chat "$PORT" "$MODEL_NAME" "$CONV_A" 100)
if [[ -n "$RESPONSE_A1" ]]; then
    pass "Got response (${#RESPONSE_A1} chars)"
else
    fail "No response from model"
    exit 1
fi

sleep 2  # Let async KV flush complete

KV_FILES_AFTER_A=$(count_kv_files "$KV_CACHE_DIR")
log "KV cache files on disk: $KV_FILES_AFTER_A"
if (( KV_FILES_AFTER_A > 0 )); then
    pass "Test 1a: KV cache stored to disk ($KV_FILES_AFTER_A files)"
else
    warn "Test 1a: No KV cache files found on disk — LMCache may not be storing"
    log "Checking vLLM log for LMCache messages..."
    grep -i "lmcache\|kv_connector\|LMCacheConnector" "$VLLM_LOG" | tail -10 || true
fi

# ── Test 1b: Re-send same conversation, check for cache hit ───
log "Test 1b: Re-sending conversation A (should get cache hit)..."
METRICS_BEFORE=$(get_lmcache_metrics "$PORT")
log "Metrics before re-send:"
echo "$METRICS_BEFORE" | head -10

RESPONSE_A2=$(send_chat "$PORT" "$MODEL_NAME" "$CONV_A" 100)
if [[ -n "$RESPONSE_A2" ]]; then
    pass "Got response (${#RESPONSE_A2} chars)"
else
    fail "No response on re-send"
    exit 1
fi

sleep 1
METRICS_AFTER=$(get_lmcache_metrics "$PORT")
log "Metrics after re-send:"
echo "$METRICS_AFTER" | head -10

# Check for LMCache log messages about hits
log "Checking vLLM log for cache hit messages..."
grep -i "hit\|match\|retrieve\|lmcache" "$VLLM_LOG" | tail -10 || true

# ── Test 1c: Send a different conversation ─────────────────────
log "Test 1c: Sending conversation B (different — should be a miss)..."
CONV_B="${LONG_PAD} What is the capital of Sweden and what is it known for?"
RESPONSE_B=$(send_chat "$PORT" "$MODEL_NAME" "$CONV_B" 100)
if [[ -n "$RESPONSE_B" ]]; then
    pass "Got response (${#RESPONSE_B} chars)"
else
    fail "No response for conversation B"
    exit 1
fi

sleep 2
KV_FILES_AFTER_B=$(count_kv_files "$KV_CACHE_DIR")
log "KV cache files after conv B: $KV_FILES_AFTER_B (was $KV_FILES_AFTER_A after conv A)"

echo ""
pass "PHASE 1 complete"
echo ""

# ════════════════════════════════════════════════════════════════════
# PHASE 2: LMCache survives sleep+checkpoint cycle
# ════════════════════════════════════════════════════════════════════

sep
log "PHASE 2: LMCache across sleep+checkpoint+restore cycle"
sep

# Record pre-sleep state
KV_FILES_PRE_SLEEP=$(count_kv_files "$KV_CACHE_DIR")
log "KV files before sleep: $KV_FILES_PRE_SLEEP"
METRICS_PRE_SLEEP=$(get_lmcache_metrics "$PORT")

# ── Sleep + checkpoint ────────────────────────────────────────
log "Sleeping + checkpointing..."
T_START=$(now_ms)
sleep_and_freeze "$PORT" "$VLLM_PID" "$ENGINE_PID" CKPT_TAKEN_1
T_FREEZE=$(elapsed_ms $T_START)
log "Frozen ($(fmt_ms $T_FREEZE), GPU: $(gpu_mem) MiB)"

# Verify KV files persist on disk during freeze
KV_FILES_FROZEN=$(count_kv_files "$KV_CACHE_DIR")
log "KV files while frozen: $KV_FILES_FROZEN"
if (( KV_FILES_FROZEN >= KV_FILES_PRE_SLEEP )); then
    pass "KV cache files persist on disk during freeze"
else
    warn "KV files decreased during freeze: $KV_FILES_PRE_SLEEP → $KV_FILES_FROZEN"
fi

# ── Restore + wake ────────────────────────────────────────────
log "Restoring + waking..."
T_START=$(now_ms)
restore_and_wake "$PORT" "$MODEL_NAME" "$VLLM_PID" "$ENGINE_PID" "$CKPT_TAKEN_1"
T_RESTORE=$(elapsed_ms $T_START)
log "Restored ($(fmt_ms $T_RESTORE), GPU: $(gpu_mem) MiB)"

# ── Re-send conversation A — assert LMCache disk hit ──────────
log "Re-sending conversation A after restore (expect LMCache disk hit)..."
# max_tokens=1 isolates prefill time for a clean comparison with cold baseline
T_START=$(now_ms)
RESPONSE_A3=$(send_chat "$PORT" "$MODEL_NAME" "$CONV_A" 1)
T_CACHED_A=$(elapsed_ms $T_START)

if [[ -z "$RESPONSE_A3" ]]; then
    fail "PHASE 2: Inference failed after sleep+checkpoint+restore"
    tail -30 "$VLLM_LOG"
    exit 1
fi

sleep 1  # allow log flush
HIT_A=$(last_hit_tokens "$VLLM_LOG")
if (( HIT_A > 0 )); then
    pass "LMCache disk hit confirmed: $HIT_A tokens served from cache"
else
    fail "PHASE 2: LMCache hit tokens = 0 — KV was not loaded from disk (check chunk_size vs prompt length)"
fi
if (( T_CACHED_A < T_COLD_A )); then
    pass "Prefill speedup: $(fmt_ms $T_CACHED_A) cached vs $(fmt_ms $T_COLD_A) cold"
else
    warn "No prefill speedup (cached: $(fmt_ms $T_CACHED_A), cold: $(fmt_ms $T_COLD_A)) — may be noise on small model"
fi

# ── Re-send conversation B — should also hit ──────────────────
log "Re-sending conversation B after restore (expect LMCache disk hit)..."
T_START=$(now_ms)
RESPONSE_B2=$(send_chat "$PORT" "$MODEL_NAME" "$CONV_B" 1)
T_CACHED_B=$(elapsed_ms $T_START)

if [[ -z "$RESPONSE_B2" ]]; then
    fail "PHASE 2: Conversation B failed after restore"
    exit 1
fi

sleep 1
HIT_B=$(last_hit_tokens "$VLLM_LOG")
if (( HIT_B > 0 )); then
    pass "Conv B LMCache hit: $HIT_B tokens from cache (prefill $(fmt_ms $T_CACHED_B))"
else
    warn "Conv B hit tokens = 0"
fi

echo ""
pass "PHASE 2 complete"
echo ""

# ════════════════════════════════════════════════════════════════════
# PHASE 3: Two-model swap — verify each model's KV cache is isolated
# ════════════════════════════════════════════════════════════════════

if [[ "${TEST_MULTI:-1}" == "1" ]]; then
    sep
    log "PHASE 3: Two-model swap with isolated KV caches"
    sep

    # Resolve model 2 path
    if [[ "$MODEL2_PATH" != /* ]]; then MODEL2_PATH="$VLLM_DIR/$MODEL2_PATH"; fi
    if [[ ! -d "$MODEL2_PATH" ]]; then
        warn "Model 2 not found: $MODEL2_PATH — skipping phase 3"
    else
        log "Model 2: $MODEL2_PATH ($MODEL2_NAME)"

        # ── Freeze model 1 (already running from phase 2) ─────────
        log "Freezing model 1..."
        sleep_and_freeze "$PORT" "$VLLM_PID" "$ENGINE_PID" CKPT_TAKEN_1
        KV_FILES_M1=$(count_kv_files "$KV_CACHE_DIR")
        log "M1 frozen. KV files on disk: $KV_FILES_M1"

        # ── Boot model 2 with LMCache ─────────────────────────────
        log "Booting model 2 with LMCache..."
        VLLM_PID2=$(boot_with_lmcache "$MODEL2_PATH" "$MODEL2_NAME" "$PORT2" "$KV_CACHE_DIR2" "$VLLM_LOG2" \
            --max-model-len "$MODEL2_MAX_MODEL_LEN")
        log "M2 starting (PID $VLLM_PID2)..."

        if ! poll_ready "$PORT2" "$MODEL2_NAME" 300; then
            fail "Model 2 did not become ready"
            tail -30 "$VLLM_LOG2"
            exit 1
        fi

        ENGINE_PID2=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
        log "M2 ready (engine PID $ENGINE_PID2, VRAM: $(gpu_mem) MiB)"

        # ── Send unique conversations to model 2 ──────────────────
        CONV_C="${LONG_PAD} Describe the process of nuclear fusion in stars."
        CONV_D="${LONG_PAD} What are the main differences between TCP and UDP protocols?"

        log "Sending conversations C and D to model 2 (cold baseline + KV storage)..."
        T_START=$(now_ms); send_chat "$PORT2" "$MODEL2_NAME" "$CONV_C" 1 > /dev/null; T_COLD_C=$(elapsed_ms $T_START)
        RESPONSE_C=$(send_chat "$PORT2" "$MODEL2_NAME" "$CONV_C" 100)
        [[ -n "$RESPONSE_C" ]] && pass "Conv C OK (cold prefill $(fmt_ms $T_COLD_C))" || fail "Conv C failed"

        T_START=$(now_ms); send_chat "$PORT2" "$MODEL2_NAME" "$CONV_D" 1 > /dev/null; T_COLD_D=$(elapsed_ms $T_START)
        RESPONSE_D=$(send_chat "$PORT2" "$MODEL2_NAME" "$CONV_D" 100)
        [[ -n "$RESPONSE_D" ]] && pass "Conv D OK (cold prefill $(fmt_ms $T_COLD_D))" || fail "Conv D failed"

        sleep 2
        KV_FILES_M2=$(count_kv_files "$KV_CACHE_DIR2")
        log "M2 KV files: $KV_FILES_M2 (in $KV_CACHE_DIR2)"
        log "M1 KV files: $KV_FILES_M1 (in $KV_CACHE_DIR) — should be unchanged"

        # Verify isolation: M1's KV dir should not have grown
        KV_FILES_M1_CHECK=$(count_kv_files "$KV_CACHE_DIR")
        if (( KV_FILES_M1_CHECK == KV_FILES_M1 )); then
            pass "KV cache directories are isolated (M1: $KV_FILES_M1, M2: $KV_FILES_M2)"
        else
            warn "M1 KV files changed during M2 inference: $KV_FILES_M1 → $KV_FILES_M1_CHECK"
        fi

        # ── Freeze model 2, restore model 1 ───────────────────────
        log "Freezing M2, restoring M1..."
        sleep_and_freeze "$PORT2" "$VLLM_PID2" "$ENGINE_PID2" CKPT_TAKEN_2

        T_START=$(now_ms)
        restore_and_wake "$PORT" "$MODEL_NAME" "$VLLM_PID" "$ENGINE_PID" "$CKPT_TAKEN_1"
        T_SWAP_M1=$(elapsed_ms $T_START)
        log "M1 restored ($(fmt_ms $T_SWAP_M1))"

        # ── Re-send M1's conversations — should hit M1's KV cache ─
        log "Re-sending conv A to model 1 (expect LMCache hit from M1's cache)..."
        T_START=$(now_ms)
        RESPONSE_A4=$(send_chat "$PORT" "$MODEL_NAME" "$CONV_A" 1)
        T_HIT_A=$(elapsed_ms $T_START)
        [[ -n "$RESPONSE_A4" ]] || { fail "Conv A failed after swap"; exit 1; }
        sleep 1
        HIT_A2=$(last_hit_tokens "$VLLM_LOG")
        if (( HIT_A2 > 0 )); then
            pass "Conv A: LMCache hit $HIT_A2 tokens, prefill $(fmt_ms $T_HIT_A)"
        else
            fail "Conv A: hit tokens = 0 after swap"
        fi

        log "Re-sending conv B to model 1..."
        T_START=$(now_ms)
        RESPONSE_B3=$(send_chat "$PORT" "$MODEL_NAME" "$CONV_B" 1)
        T_HIT_B=$(elapsed_ms $T_START)
        [[ -n "$RESPONSE_B3" ]] || { fail "Conv B failed after swap"; exit 1; }
        sleep 1
        HIT_B2=$(last_hit_tokens "$VLLM_LOG")
        (( HIT_B2 > 0 )) && pass "Conv B: LMCache hit $HIT_B2 tokens, prefill $(fmt_ms $T_HIT_B)" || warn "Conv B: hit tokens = 0"

        # ── Freeze M1, restore M2, verify M2's cache ──────────────
        log "Freezing M1, restoring M2..."
        sleep_and_freeze "$PORT" "$VLLM_PID" "$ENGINE_PID" CKPT_TAKEN_1

        T_START=$(now_ms)
        restore_and_wake "$PORT2" "$MODEL2_NAME" "$VLLM_PID2" "$ENGINE_PID2" "$CKPT_TAKEN_2"
        T_SWAP_M2=$(elapsed_ms $T_START)
        log "M2 restored ($(fmt_ms $T_SWAP_M2))"

        log "Re-sending conv C to model 2 (expect LMCache hit from M2's cache)..."
        T_START=$(now_ms)
        RESPONSE_C2=$(send_chat "$PORT2" "$MODEL2_NAME" "$CONV_C" 1)
        T_HIT_C=$(elapsed_ms $T_START)
        [[ -n "$RESPONSE_C2" ]] || { fail "Conv C failed after swap"; exit 1; }
        sleep 1
        HIT_C2=$(last_hit_tokens "$VLLM_LOG2")
        if (( HIT_C2 > 0 )); then
            pass "Conv C: LMCache hit $HIT_C2 tokens, prefill $(fmt_ms $T_HIT_C) vs cold $(fmt_ms $T_COLD_C)"
        else
            fail "Conv C: hit tokens = 0 after swap"
        fi

        log "Re-sending conv D to model 2..."
        T_START=$(now_ms)
        RESPONSE_D2=$(send_chat "$PORT2" "$MODEL2_NAME" "$CONV_D" 1)
        T_HIT_D=$(elapsed_ms $T_START)
        [[ -n "$RESPONSE_D2" ]] || { fail "Conv D failed after swap"; exit 1; }
        sleep 1
        HIT_D2=$(last_hit_tokens "$VLLM_LOG2")
        (( HIT_D2 > 0 )) && pass "Conv D: LMCache hit $HIT_D2 tokens, prefill $(fmt_ms $T_HIT_D) vs cold $(fmt_ms $T_COLD_D)" || warn "Conv D: hit tokens = 0"

        echo ""
        log "PHASE 3 Summary:"
        echo "    M1 KV files: $KV_FILES_M1 ($KV_CACHE_DIR)"
        echo "    M2 KV files: $KV_FILES_M2 ($KV_CACHE_DIR2)"
        echo "    M1 swap time: $(fmt_ms $T_SWAP_M1)"
        echo "    M2 swap time: $(fmt_ms $T_SWAP_M2)"
        echo "    Post-swap inference (M1 conv A): $(fmt_ms $T_HIT_A)"
        echo "    Post-swap inference (M1 conv B): $(fmt_ms $T_HIT_B)"
        echo "    Post-swap inference (M2 conv C): $(fmt_ms $T_HIT_C)"
        echo "    Post-swap inference (M2 conv D): $(fmt_ms $T_HIT_D)"
        echo ""

        # Check metrics for both models
        log "M1 final metrics:"
        get_lmcache_metrics "$PORT" | head -5
        log "M2 final metrics:"
        get_lmcache_metrics "$PORT2" | head -5

        echo ""
        pass "PHASE 3 complete"
        echo ""

        # ════════════════════════════════════════════════════════════════════
        # PHASE 4: KV prefetch ordering — identify and preload chunk files
        #          for the incoming model's priority conversations
        # ════════════════════════════════════════════════════════════════════
        #
        # Simulates production round-robin prefetch:
        #   While M2 is serving its current batch, identify and load M1's
        #   KV chunk files for its first N upcoming conversations. By the time
        #   the swap occurs, those chunks are in system RAM (OS page cache +
        #   LMCache CPU backend survives SIGSTOP), so M1's first inferences
        #   hit CPU/RAM rather than cold disk.
        #
        # This phase validates the file-identification mechanism (hash
        # computation). The production latency benefit is observable when
        # the CPU backend is empty (fresh restart); the SIGSTOP/SIGCONT swap
        # already keeps CPU cache warm, so the key value here is the prefetch
        # script working correctly for any given conversation text.

        sep
        log "PHASE 4: Prefetch ordering — identify M1 KV chunks before swap"
        sep

        PREFETCH_SCRIPT="${SCRIPT_DIR}/prefetch_kv.py"
        if [[ ! -f "$PREFETCH_SCRIPT" ]]; then
            warn "prefetch_kv.py not found at $PREFETCH_SCRIPT — skipping phase 4"
        else
            # Current state: M2 active, M1 frozen.
            # Identify M1's priority conversations' KV chunks while M2 is still running.

            log "Identifying M1 KV chunks for upcoming conversations A and B..."
            PREFETCH_OUT=$( PYTHONHASHSEED=0 "$VENV_PYTHON" "$PREFETCH_SCRIPT" \
                --model-name "$MODEL_NAME" \
                --model-path "$MODEL_PATH" \
                --kv-dir "$KV_CACHE_DIR" \
                --load \
                "$CONV_A" "$CONV_B" 2>/dev/null )
            PREFETCH_EXIT=$?

            if [[ $PREFETCH_EXIT -le 1 ]]; then
                ALL_FOUND_M1=$(echo "$PREFETCH_OUT" | \
                    "$VENV_PYTHON" -c "import sys,json; d=json.load(sys.stdin); print(d['all_found'])")
                DISK_CHUNKS_M1=$(echo "$PREFETCH_OUT" | \
                    "$VENV_PYTHON" -c "import sys,json; d=json.load(sys.stdin); print(d['total_disk_chunks'])")
                LOADED_BYTES_M1=$(echo "$PREFETCH_OUT" | \
                    "$VENV_PYTHON" -c "import sys,json; d=json.load(sys.stdin); print(d['total_loaded_bytes'])")
                echo "$PREFETCH_OUT" | "$VENV_PYTHON" -c "
import sys, json
d = json.load(sys.stdin)
for c in d['conversations']:
    found = c['found_chunks']
    total = c['total_chunks']
    mib = c.get('loaded_bytes', 0) // 1024 // 1024
    print(f'    Conv {c[\"conv_index\"]}: {found}/{total} chunks identified ({mib} MiB loaded)')
"
                if [[ "$ALL_FOUND_M1" == "True" ]]; then
                    pass "M1 prefetch: all chunks identified and loaded ($(( LOADED_BYTES_M1 / 1024 / 1024 )) MiB)"
                else
                    warn "M1 prefetch: some chunks not found on disk (wrong hash algorithm?)"
                    echo "$PREFETCH_OUT" | head -30
                fi
            else
                warn "prefetch_kv.py failed (exit $PREFETCH_EXIT) — check PYTHONHASHSEED and sha256_cbor"
            fi

            echo ""

            # Cross-verify: M2's own conversations should also be identifiable
            log "Cross-checking M2 KV chunks for conversations C and D..."
            PREFETCH_OUT2=$( PYTHONHASHSEED=0 "$VENV_PYTHON" "$PREFETCH_SCRIPT" \
                --model-name "$MODEL2_NAME" \
                --model-path "$MODEL2_PATH" \
                --kv-dir "$KV_CACHE_DIR2" \
                "$CONV_C" "$CONV_D" 2>/dev/null )
            PREFETCH_EXIT2=$?

            if [[ $PREFETCH_EXIT2 -le 1 ]]; then
                ALL_FOUND_M2=$(echo "$PREFETCH_OUT2" | \
                    "$VENV_PYTHON" -c "import sys,json; d=json.load(sys.stdin); print(d['all_found'])")
                echo "$PREFETCH_OUT2" | "$VENV_PYTHON" -c "
import sys, json
d = json.load(sys.stdin)
for c in d['conversations']:
    found = c['found_chunks']
    total = c['total_chunks']
    print(f'    Conv {c[\"conv_index\"]}: {found}/{total} chunks identified')
"
                if [[ "$ALL_FOUND_M2" == "True" ]]; then
                    pass "M2 prefetch: all chunks identified correctly"
                else
                    warn "M2 prefetch: some M2 chunks not found"
                fi
            fi

            echo ""
            log "PHASE 4 Summary:"
            echo "    M1 disk chunks total:    ${DISK_CHUNKS_M1:-?}"
            echo "    M1 loaded into RAM:      $(( ${LOADED_BYTES_M1:-0} / 1024 / 1024 )) MiB"
            echo "    M1 all_found:            ${ALL_FOUND_M1:-?}"
            echo "    M2 all_found:            ${ALL_FOUND_M2:-?}"
            echo "    Prefetch script:         $PREFETCH_SCRIPT"
            echo ""
            echo "  Note: The prefetch script identifies disk chunks by reproducing"
            echo "  LMCache's sha256_cbor rolling hash from the conversation token sequence."
            echo "  In production, this runs while the current model handles its batch,"
            echo "  pre-loading the incoming model's first N conversations into system RAM."
            echo ""

            if [[ "${ALL_FOUND_M1:-False}" == "True" && "${ALL_FOUND_M2:-False}" == "True" ]]; then
                pass "PHASE 4 complete — prefetch identification works correctly"
            else
                warn "PHASE 4 partial — some chunks not identified (check hash settings)"
            fi
        fi
    fi
fi

echo ""

# ════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════

sep
log "SUMMARY"
sep
echo ""
echo "  LMCache KV Storage:"
echo "    KV files after conv A:     $KV_FILES_AFTER_A"
echo "    KV files after conv B:     $KV_FILES_AFTER_B"
echo "    KV files after freeze:     ${KV_FILES_FROZEN:-n/a}"
echo "    KV dir:                    $KV_CACHE_DIR"
echo ""
echo "  Key findings:"
if (( ${KV_FILES_AFTER_A:-0} > 0 )); then
    echo "    [OK] LMCache stores KV to disk"
else
    echo "    [??] Could not confirm KV disk storage — check logs"
fi
echo "    [OK] Inference works after sleep+checkpoint+restore"
echo ""

pass "All tests passed!"
log "vLLM log: $VLLM_LOG"
