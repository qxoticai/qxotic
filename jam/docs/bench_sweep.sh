#!/usr/bin/env bash
# pp512 sweep behind docs/bench_plot.py: jinfer (native jam) and llama-bench, matched ISA per tier.
# Expects pure quants <MODELS>/gemma-4-E2B-it-{Q4_0,Q8_0,Q4_K,Q5_K,Q6_K}_pure.gguf (llama-quantize --pure
# from BF16) and llama.cpp built once per tier with GGML_NATIVE=OFF and only that tier's GGML_* flags.
# JAVA_HOME, when set, selects the JDK. Run from the repository root after `mvn install -DskipTests` and
# `mvn -q -pl jinfer/jinfer-bench dependency:build-classpath -Dmdep.outputFile=target/cp.txt`.
set -euo pipefail
MODELS=${MODELS:?path to the model directory}
LLAMA=${LLAMA:?path to the llama.cpp checkout}
OUT=${OUT:-bench-results/sweep}
THREADS=${THREADS:-16}
JAVA=${JAVA_HOME:+$JAVA_HOME/bin/}java
mkdir -p "$OUT"
BENCH_FLAGS="--enable-preview --add-modules jdk.incubator.vector,jdk.httpserver --enable-native-access=ALL-UNNAMED -XX:CompileCommandFile=jinfer/hotspot_compile_commands -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0"
CP="jinfer/jinfer-bench/target/classes:$(< jinfer/jinfer-bench/target/cp.txt)"
declare -A LB=([sse3]=build-sse3 [avx2]=build-avx2 [avx_vnni]=build-avxvnni [avx512_vnni]=build-avx512)
for tier in sse3 avx2 avx_vnni avx512_vnni; do
  for q in Q4_0 Q8_0 Q4_K Q5_K Q6_K; do
    f=$MODELS/gemma-4-E2B-it-${q}_pure.gguf
    echo "== $tier $q"
    JAM_DEBUG=1 JAM_ISA=$tier "$JAVA" $BENCH_FLAGS -cp "$CP" com.qxotic.jinfer.bench.JinferBench \
      -m "$f" -p 512 -n 0 -r 5 -w 2 -t "$THREADS" > "$OUT/jinfer-$tier-$q.log" 2> "$OUT/jinfer-$tier-$q.err"
    grep -E '^\|.*pp512' "$OUT/jinfer-$tier-$q.log"
    "$LLAMA/${LB[$tier]}/bin/llama-bench" -m "$f" -p 512 -n 0 -r 5 -t "$THREADS" \
      > "$OUT/llama-$tier-$q.log" 2> "$OUT/llama-$tier-$q.err"
    grep -E '^\|.*pp512' "$OUT/llama-$tier-$q.log"
  done
done
