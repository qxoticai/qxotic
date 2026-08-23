#!/usr/bin/env bash
# Reproduces the Gemma docs' "token budget comparison" (snippets 5-6): run the SAME detection
# prompt on the SAME image at each image-token budget and compare how the answer sharpens with
# resolution. Uses the jinfer.gemma4.imageTokenBudget knob (one JVM per budget - the property is
# read once per process).
#
#   ./gemma-budget-sweep.sh city-streets.jpg
#   ./gemma-budget-sweep.sh city-streets.jpg "detect person and car, output only json"
#
# Use a larger model for sharper detection:
#   MODEL_REF=hf.co/unsloth/gemma-4-12b-it-GGUF:Q8_0 \
#   MEDIA_REF=hf.co/unsloth/gemma-4-12b-it-GGUF/mmproj-F32.gguf \
#     ./gemma-budget-sweep.sh photo.jpg
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
img="${1:?usage: gemma-budget-sweep.sh <image> [prompt]}"
prompt="${2:-detect person and car, output only json}"
args=("$img" "$prompt")
[[ -n "${MODEL_REF:-}" ]] && args+=("$MODEL_REF" "${MEDIA_REF:?set MEDIA_REF alongside MODEL_REF}")

for budget in 70 140 280 560; do
    echo "════════════════════ budget = $budget tokens ════════════════════"
    jbang -Djinfer.gemma4.imageTokenBudget="$budget" "$here/GemmaVision.java" "${args[@]}"
    echo
done
