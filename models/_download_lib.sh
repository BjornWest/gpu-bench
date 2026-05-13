# Shared helpers for per-model download.sh scripts.
# Source from a download.sh — do not execute directly.
#
# Pattern:
#
#   source "$(dirname "${BASH_SOURCE[0]}")/../_download_lib.sh"
#   download_model "MiniMaxAI/MiniMax-M2.7-NVFP4" "$MODEL_DIR/model"
#
# This creates a minimal venv with just `hf` + `hf-transfer` (no vllm, no
# torch). On a CPU-only download node this saves several GB of install
# time and disk vs the full benchmark setup.

set -euo pipefail

ensure_download_venv() {
    local model_dir=$1
    local venv="$model_dir/download_venv"
    if [[ ! -x "$venv/bin/python" ]]; then
        echo "[download] Creating download venv at $venv"
        uv venv "$venv"
        # `hf` is the v1.x-compatible CLI per audit-lab feedback_hf_cli_install.md.
        # `huggingface_hub[cli]` is broken on v1.x; do not use it.
        "$venv/bin/python" -m pip install --quiet --upgrade pip 2>/dev/null || true
        uv pip install --python "$venv/bin/python" hf hf-transfer
    fi
    echo "$venv"
}

require_disk_free_gb() {
    local need_gb=$1
    local path=$2
    local avail
    avail=$(df -BG "$path" | awk 'NR==2 {sub(/G/,"",$4); print $4}')
    if (( avail < need_gb )); then
        echo "[download] ERROR: need ${need_gb} GB free at ${path}, only ${avail} GB available." >&2
        return 1
    fi
}

download_model() {
    # $1 — HF repo id  (e.g. nvidia/MiniMax-M2.7-NVFP4)
    # $2 — local destination dir
    # $3 — optional: estimated size in GB (for disk-space check)
    # $4 — optional: --include pattern (e.g. "*.safetensors *.json" to skip GGUF)
    local repo=$1
    local dest=$2
    local size_gb=${3:-0}
    local include_pat=${4:-}

    local model_dir
    model_dir=$(dirname "$dest")
    local venv
    venv=$(ensure_download_venv "$model_dir")

    mkdir -p "$dest"

    if (( size_gb > 0 )); then
        require_disk_free_gb "$size_gb" "$model_dir"
    fi

    # Idempotency: if .complete marker exists, skip
    if [[ -f "$dest/.download_complete" ]]; then
        echo "[download] $repo already complete at $dest (remove .download_complete to redo)"
        return 0
    fi

    echo "[download] Repo:        $repo"
    echo "[download] Destination: $dest"
    echo "[download] Expected:    ~${size_gb} GB"
    echo "[download] Include:     ${include_pat:-<all files>}"

    local extra_args=()
    if [[ -n "$include_pat" ]]; then
        # hf download supports --include with space-separated patterns
        for p in $include_pat; do
            extra_args+=("--include" "$p")
        done
    fi

    HF_HUB_ENABLE_HF_TRANSFER=1 \
        "$venv/bin/hf" download "$repo" \
        --local-dir "$dest" \
        --max-workers 8 \
        "${extra_args[@]}"

    # Sanity: at least one safetensors file present
    if ! ls "$dest"/*.safetensors >/dev/null 2>&1 && \
       ! ls "$dest"/model-*.safetensors >/dev/null 2>&1; then
        echo "[download] WARN: no *.safetensors found in $dest — check the repo / include pattern" >&2
    fi

    touch "$dest/.download_complete"
    echo "[download] ✓ Done. Marker: $dest/.download_complete"
    du -sh "$dest"
}
