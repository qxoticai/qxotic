#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"

BENCH_MODELS="${BENCH_MODELS:-gpt2,mistral-tekken}"
BENCH_SIZES="${BENCH_SIZES:-1k,32k}"
BENCH_CORPORA="${BENCH_CORPORA:-chat}"
BENCH_IMPLEMENTATIONS="${BENCH_IMPLEMENTATIONS:-classic,fast}"
BENCH_WARMUP_ITERS="${BENCH_WARMUP_ITERS:-2}"
BENCH_MEASURE_ITERS="${BENCH_MEASURE_ITERS:-3}"
BENCH_WARMUP_SECS="${BENCH_WARMUP_SECS:-1}"
BENCH_MEASURE_SECS="${BENCH_MEASURE_SECS:-1}"
PY_BENCH_DURATION_SECS="${PY_BENCH_DURATION_SECS:-1.5}"
PY_BENCH_WARMUP_SECS="${PY_BENCH_WARMUP_SECS:-0.3}"
PY_BENCH_REPEATS="${PY_BENCH_REPEATS:-3}"

if [[ ! -d "$SCRIPT_DIR/.venv" ]]; then
  echo "[setup] creating local virtual environment with uv"
  uv venv "$SCRIPT_DIR/.venv"
else
  echo "[setup] using existing .venv"
fi

echo "[setup] installing Python dependencies"
uv pip install --python "$PYTHON_BIN" -r "$SCRIPT_DIR/requirements.txt" matplotlib

echo "[java] compiling benchmark classes"
mvn -pl toknroll-benchmarks -am -DskipTests test-compile

echo "[java] running ModelTokenizerBenchmark"
mvn -pl toknroll-benchmarks \
  -Dexec.mainClass=com.qxotic.toknroll.benchmarks.TokenizerBenchmarkRunner \
  -Dexec.classpathScope=test \
  -Dexec.args="ModelTokenizerBenchmark.encode ModelTokenizerBenchmark.decode \
    -p implementation=${BENCH_IMPLEMENTATIONS} \
    -p model=${BENCH_MODELS} \
    -p corpus=${BENCH_CORPORA} \
    -p size=${BENCH_SIZES} \
    -rf json -rff toknroll-benchmarks/target/jmh-model-tokenizers.json \
    -wi ${BENCH_WARMUP_ITERS} -i ${BENCH_MEASURE_ITERS} \
    -w ${BENCH_WARMUP_SECS}s -r ${BENCH_MEASURE_SECS}s -f 0" \
  exec:java

echo "[python] running benchmark_model_tokenizers.py"
"$PYTHON_BIN" "$SCRIPT_DIR/benchmark_model_tokenizers.py" \
  --duration "$PY_BENCH_DURATION_SECS" \
  --warmup "$PY_BENCH_WARMUP_SECS" \
  --repeats "$PY_BENCH_REPEATS" \
  --sizes "$BENCH_SIZES" \
  --corpora "$BENCH_CORPORA" \
  --models "$BENCH_MODELS" \
  --implementations tiktoken,tokie,hf-tokenizers,mistral-common \
  --csv "$ROOT_DIR/target/python-tokenizers.csv"

echo "[charts] building cross-runtime plots"
"$PYTHON_BIN" "$SCRIPT_DIR/plot_cross_runtime.py" \
  --java-json "$ROOT_DIR/toknroll-benchmarks/target/jmh-model-tokenizers.json" \
  --python-csv "$ROOT_DIR/target/python-tokenizers.csv" \
  --output-dir "$ROOT_DIR/target/benchmarks/charts" \
  --merged-csv "$ROOT_DIR/target/benchmarks/merged-cross-runtime.csv"

echo "[done] charts available in target/benchmarks/charts"
