#!/usr/bin/env bash
# Run the same image and prompt at four image-token budgets.
#
#   ./gemma-budget-sweep.sh city-streets.jpg
#   ./gemma-budget-sweep.sh city-streets.jpg "detect person and car, output only json"
#
# Use a larger model for sharper detection:
#   MODEL_REF=unsloth/gemma-4-12b-it-GGUF:Q8_0 \
#   MEDIA_REF=unsloth/gemma-4-12b-it-GGUF/mmproj-F32.gguf \
#     ./gemma-budget-sweep.sh photo.jpg
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
img="${1:?usage: gemma-budget-sweep.sh <image> [prompt]}"
prompt="${2:-detect person and car, output only json}"
args=("$img" "$prompt")
[[ -n "${MODEL_REF:-}" ]] && args+=("$MODEL_REF" "${MEDIA_REF:?set MEDIA_REF alongside MODEL_REF}")

for budget in 70 140 280 560; do
    printf '\n== %s image tokens ==\n' "$budget"
    jbang -Djinfer.gemma4.imageTokenBudget="$budget" "$here/GemmaVision.java" "${args[@]}"
done
