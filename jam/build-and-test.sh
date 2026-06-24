#!/usr/bin/env bash
# Build jam, run the comprehensive correctness suite, and benchmark every backend.
# Intended for Apple Silicon (exercises neon / dotprod / i8mm / metal) but works anywhere.
#
#   ./build-and-test.sh              # build + test + bench at the default size
#   ./build-and-test.sh 2048 30      # bench size 2048, 30 iters
set -euo pipefail

cd "$(dirname "$0")"
SIZE="${1:-1024}"
ITERS="${2:-20}"
BUILD=build

# threads = performance cores on Apple (best for compute-bound prefill), else all physical cores
if THREADS=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null); then :
elif THREADS=$(sysctl -n hw.physicalcpu 2>/dev/null); then :
elif THREADS=$(nproc 2>/dev/null); then :
else THREADS=4; fi

hr(){ printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

hr "configure + build (Release)"
cmake -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j

hr "correctness — every supported kernel × {1,3} threads + global"
# ctest runs jam_test (all capped contexts); --output-on-failure prints any [FAIL] lines
( cd "$BUILD" && ctest --output-on-failure )

hr "backends present on this machine"
# jam_test's header lists the contexts it built (one per kernel the CPU/GPU supports)
"$BUILD/jam_test" | sed -n '1,2p'

hr "benchmark — all backends, SQUARE ${SIZE}³ (compute-bound -> GMAC/s is the metric)"
JAM_NUM_THREADS="$THREADS" "$BUILD/jam_bench" "$SIZE" "$ITERS"

hr "benchmark — all backends, GEMV 16384×1×8192 (bandwidth-bound -> GB/s approaches peak DRAM)"
# weight (512MB F32 / 136MB Q8) far exceeds cache, so GB/s reflects real DRAM. F32 pins at peak and is
# ISA-independent; Q8_0 finishes faster by streaming ~4x fewer weight bytes.
JAM_NUM_THREADS="$THREADS" "$BUILD/jam_bench" 16384 1 8192 20

# Multi-thread numbers above are thermally coupled (kernels run back-to-back). For a clean per-backend
# reading, isolate each — Metal especially (GPU, untiled first cut) is worth seeing on its own.
for ISA in i8mm metal; do
    hr "isolated: $ISA  (${SIZE}x${SIZE}x${SIZE}, ${THREADS} threads)"
    JAM_ISA="$ISA" JAM_NUM_THREADS="$THREADS" "$BUILD/jam_bench" "$SIZE" "$ITERS" \
        || echo "  ($ISA not available on this machine — skipped)"
done

hr "done"
echo "Tip: JAM_ISA=<neon|dotprod|i8mm|metal> JAM_NUM_THREADS=N $BUILD/jam_bench <size> <iters>"
