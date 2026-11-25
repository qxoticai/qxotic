#!/usr/bin/env bash
# Refreshes src/test/resources/stablehlo-samples from a local openxla/stablehlo checkout.
# Usage: tools/refresh-samples.sh /path/to/openxla/stablehlo
# Files are copied unmodified (Apache-2.0, see NOTICE); the source commit goes to SAMPLES_VERSION.
set -euo pipefail
repo="${1:?usage: $0 /path/to/openxla/stablehlo}"
here="$(cd "$(dirname "$0")/.." && pwd)"
out="$here/src/test/resources/stablehlo-samples"
src="$repo/stablehlo"
rm -rf "$out/text" "$out/portable"
mkdir -p "$out/text/testdata" "$out/text/interpret" "$out/text/syntax" "$out/portable"

# testdata: the smallest file of every op family (abs_*, add_*, gather_*, ...).
declare -A best
while IFS= read -r f; do
  name="$(basename "$f")"
  family="$(sed -E 's/_(bfloat16|float|int|uint|bool|complex|shape|dtypes|lhs|rhs|operand|axis|dimensions|in|nan|inf|[0-9]).*$//' <<<"$name")"
  size=$(stat -c %s "$f")
  if [[ -z "${best[$family]:-}" ]] || (( size < $(stat -c %s "${best[$family]}") )); then
    best[$family]="$f"
  fi
done < <(find "$src/testdata" -maxdepth 1 -name '*.mlir' | sort)
for f in "${best[@]}"; do cp "$f" "$out/text/testdata/"; done

cp "$src"/tests/interpret/*.mlir "$out/text/interpret/"
for f in print_stablehlo ops_stablehlo_roundtrip ops_stablehlo_quantized ops_stablehlo_bounded_dynamism \
         print_reduce print_while print_conv_dimension_numbers_raw; do
  cp "$src/tests/$f.mlir" "$out/text/syntax/"
done

cp "$src"/tests/vhlo/*.mlir.bc "$out/portable/"
oldest="$(ls "$src"/tests/vhlo/stablehlo_legalize_to_vhlo.0_9_0.mlir)"
for f in "$oldest" "$src/tests/vhlo/stablehlo_legalize_to_vhlo.1_0_0.mlir" \
         "$(ls "$src"/tests/vhlo/stablehlo_legalize_to_vhlo.1_*_0.mlir | sort -t_ -k5,5n -k6,6n | tail -1)"; do
  cp "$f" "$out/portable/"
done
for f in "$src"/tests/vhlo/invalid_vhlo_*.mlir; do cp "$f" "$out/portable/"; done

{
  echo "openxla/stablehlo $(git -C "$repo" rev-parse HEAD) ($(git -C "$repo" log -1 --format=%cs))"
  echo "copied by tools/refresh-samples.sh; files are unmodified."
} > "$out/SAMPLES_VERSION"
echo "text/testdata: $(ls "$out/text/testdata" | wc -l), text/interpret: $(ls "$out/text/interpret" | wc -l)," \
     "text/syntax: $(ls "$out/text/syntax" | wc -l), portable: $(ls "$out/portable" | wc -l)"
