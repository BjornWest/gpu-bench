#!/usr/bin/env bash
# Convenience: download every model in the current matrix sequentially.
# Comment out models you don't need.
#
# Run on a CPU node before provisioning the B200 rack. Total disk needed:
# ~1.7 TB if you download everything in the matrix.

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS=(
    deepseek-v32
    glm5
    kimi-k25
    minimax-m27
    mistral-med35
    qwen36
    gemma4-auditor
)

for m in "${MODELS[@]}"; do
    echo ""
    echo "═══ Downloading $m ═══"
    if [[ -x "$DIR/$m/download.sh" ]]; then
        bash "$DIR/$m/download.sh"
    else
        echo "[skip] $m has no download.sh"
    fi
done

echo ""
echo "═══ All downloads complete ═══"
du -sh "$DIR"/*/model 2>/dev/null
