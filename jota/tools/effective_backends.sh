#!/usr/bin/env bash

set -euo pipefail

KNOWN_BACKENDS=(panama c hip opencl metal)
available_csv="${JOTA_AVAILABLE_BACKENDS:-}"
include_csv="${JOTA_BACKENDS_INCLUDE:-}"
exclude_csv="${JOTA_BACKENDS_EXCLUDE:-}"

usage() {
  cat <<'EOF'
Usage: effective_backends.sh [options]

Compute effective backend set using the same precedence as Jota runtime filters.

Options:
  --available <csv>  Available backends before filtering (default: all known)
  --include <csv>    Include filter (same as -Djota.backends.include)
  --exclude <csv>    Exclude filter (same as -Djota.backends.exclude)
  --help             Show this help

Notes:
  - Exclude has priority over include.
  - Input is comma-separated and case-insensitive.
  - Also supports env vars:
      JOTA_AVAILABLE_BACKENDS
      JOTA_BACKENDS_INCLUDE
      JOTA_BACKENDS_EXCLUDE

Example:
  ./tools/effective_backends.sh --available "c,hip,opencl" --include "c,opencl" --exclude "opencl"
EOF
}

normalize_csv() {
  local raw="$1"
  local out=()
  if [[ -z "${raw//[[:space:],]/}" ]]; then
    echo ""
    return
  fi
  IFS=',' read -r -a tokens <<< "$raw"
  for t in "${tokens[@]}"; do
    local n
    n="$(printf '%s' "$t" | tr '[:upper:]' '[:lower:]' | xargs)"
    if [[ -n "$n" ]]; then
      out+=("$n")
    fi
  done
  (IFS=','; echo "${out[*]}")
}

csv_contains() {
  local csv="$1"
  local needle="$2"
  [[ ",${csv}," == *",${needle},"* ]]
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --available)
      available_csv="${2:-}"
      shift 2
      ;;
    --include)
      include_csv="${2:-}"
      shift 2
      ;;
    --exclude)
      exclude_csv="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

available_csv="$(normalize_csv "$available_csv")"
include_csv="$(normalize_csv "$include_csv")"
exclude_csv="$(normalize_csv "$exclude_csv")"

if [[ -z "$available_csv" ]]; then
  available_csv="$(IFS=','; echo "${KNOWN_BACKENDS[*]}")"
fi

effective=()
IFS=',' read -r -a available <<< "$available_csv"

for backend in "${available[@]}"; do
  [[ -z "$backend" ]] && continue

  included=true
  if [[ -n "$include_csv" ]]; then
    if csv_contains "$include_csv" "$backend"; then
      included=true
    else
      included=false
    fi
  fi

  excluded=false
  if [[ -n "$exclude_csv" ]] && csv_contains "$exclude_csv" "$backend"; then
    excluded=true
  fi

  if [[ "$included" == true && "$excluded" == false ]]; then
    effective+=("$backend")
  fi
done

echo "available=${available_csv}"
echo "include=${include_csv:-<all>}"
echo "exclude=${exclude_csv:-<none>}"
echo "priority=exclude_over_include"
if [[ ${#effective[@]} -eq 0 ]]; then
  echo "effective=<none>"
else
  (IFS=','; echo "effective=${effective[*]}")
fi
