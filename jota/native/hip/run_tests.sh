#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
HIP_DIR="$ROOT_DIR/native/hip"

JAVA_HOME=${JAVA_HOME:-/usr/lib/jvm/java-25-graalvm}
ROCM_PATH=${ROCM_PATH:-/opt/rocm}

export JAVA_HOME
export LD_LIBRARY_PATH="$HIP_DIR/build:$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"

bash "$HIP_DIR/build.sh"

env -u HIP_HSACO_PATH -u HIP_KERNEL_NAME -u HIP_META_HSACO_PATH -u HIP_META_KERNEL_NAME \
  mvn test -Dtest=HipRuntimeTest,HipKernelSmokeTest,HipKernelMetadataSmokeTest,HipUnaryKernelSmokeTest,HipBackendHostOutputTest
