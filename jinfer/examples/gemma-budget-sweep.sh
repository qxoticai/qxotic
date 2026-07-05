#!/usr/bin/env bash
# Reproduces the Gemma docs' "token budget comparison" (snippets 5-6): run the SAME detection
# prompt on the SAME image at each image-token budget and compare how the answer sharpens with
# resolution. Uses the jinfer.gemma4.imageTokenBudget knob (one JVM per budget - the property is
# read once per process).
#
#   ./gemma-budget-sweep.sh city-streets.jpg
#   ./gemma-budget-sweep.sh city-streets.jpg "detect person and car, output only json"
#
# Point it at a bigger model for sharper detection by exporting GGUF/MMPROJ:
#   GGUF=~/models/unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-Q8_0.gguf \
#   MMPROJ=~/models/unsloth/gemma-4-12b-it-GGUF/mmproj-F32.gguf ./gemma-budget-sweep.sh photo.jpg
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
img="${1:?usage: gemma-budget-sweep.sh <image> [prompt]}"
prompt="${2:-detect person and car, output only json}"
args=("$img" "$prompt")
[[ -n "${GGUF:-}" ]] && args+=("$GGUF" "${MMPROJ:?set MMPROJ alongside GGUF}")

for budget in 70 140 280 560; do
    echo "════════════════════ budget = $budget tokens ════════════════════"
    jbang -Djinfer.gemma4.imageTokenBudget="$budget" "$here/GemmaVision.java" "${args[@]}"
    echo
done
