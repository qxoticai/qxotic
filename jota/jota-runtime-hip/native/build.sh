#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
OUT_DIR="$ROOT_DIR/build"
LIB_NAME="libjota_hip.so"

ROCM_PATH=${ROCM_PATH:-/opt/rocm}
HIP_INCLUDE="$ROCM_PATH/include"
HIP_LIB="$ROCM_PATH/lib"

if [[ -z "${JAVA_HOME:-}" ]]; then
  echo "JAVA_HOME must be set to build JNI library." >&2
  exit 1
fi

INCLUDE_DIRS=("$JAVA_HOME/include" "$JAVA_HOME/include/linux")

mkdir -p "$OUT_DIR"

gcc -shared -fPIC -O2 \
  -D__HIP_PLATFORM_AMD__ \
  -I"${INCLUDE_DIRS[0]}" \
  -I"${INCLUDE_DIRS[1]}" \
  -I"$HIP_INCLUDE" \
  -L"$HIP_LIB" \
  -Wl,-rpath,"$HIP_LIB" \
  -lamdhip64 \
  -o "$OUT_DIR/$LIB_NAME" \
  "$ROOT_DIR/jota_hip_jni.c"

echo "Built $OUT_DIR/$LIB_NAME"
