#!/bin/sh
# Build the jam-vs-tinyBLAS benchmark. Requires both projects already built:
#   - jam:        qxotic/jam/build/libjam.so            (cmake --build jam/build, or mvn -pl jam package)
#   - llama.cpp:  llama.cpp/build/bin/lib{ggml-cpu,ggml-base}.so
#                 (cmake -B build -DGGML_NATIVE=ON && cmake --build build --target ggml-cpu -j)
# Override the checkout locations with JAM=... LLAMA=... if they live elsewhere;
# LLAMA_BUILD=build-avx2 selects a different llama.cpp build dir (default: build).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
JAM=${JAM:-$(cd "$HERE/.." && pwd)}
: "${LLAMA:?set LLAMA to the llama.cpp checkout}"
LLAMA_BUILD=${LLAMA_BUILD:-build}

cc -O3 -march=native -o "$HERE/jam_vs_tinyblas" "$HERE/jam_vs_tinyblas.c" \
   -I"$JAM/include" -I"$JAM/tests" \
   -L"$JAM/build" -ljam \
   -L"$LLAMA/$LLAMA_BUILD/bin" -lggml-cpu -lggml-base \
   -Wl,-rpath,"$JAM/build" -Wl,-rpath,"$LLAMA/$LLAMA_BUILD/bin" \
   -lstdc++ -lpthread -lm
echo "built $HERE/jam_vs_tinyblas"
