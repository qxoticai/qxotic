# Benchmarks

This directory contains the benchmark harness and reporting tooling for Tok'n'Roll.
It covers two flows:

- Wiki-corpus Java JMH benchmarks (`enwik8` / `enwik9`)
- Cross-runtime comparisons (Java JMH + Python tokenizer libraries)

Run all commands from the repository root unless noted otherwise.

## Prerequisites

- Java 17+
- Maven (`mvn`)
- Python 3.11+
- `uv` installed locally
- Python deps installed in a local `.venv`

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r benchmarks/requirements.txt
uv pip install --python .venv/bin/python matplotlib
```

Tip: use `.venv/bin/python` for benchmark scripts to keep environments isolated.

## Scripts

- `benchmark_model_tokenizers.py`: Python-side tokenizer benchmark harness; writes CSV.
- `run_wiki_benchmarks.py`: helper to download corpora and run wiki JMH benchmarks.
- `run_cross_runtime_benchmarks.sh`: end-to-end Java + Python run with merged outputs/charts.
- `plot_cross_runtime.py`: PNG charts from Java JSON + Python CSV.
- `openai_apples_report.py`: apples-to-apples markdown report from Java JSON + Python CSV.
- `jmh_to_markdown.py`: converts JMH JSON to markdown tables.
- `generate_ground_truth.py`: regenerates tokenizer ground-truth fixtures used by tests.
- `generate_enwik8_ground_truth.py`: regenerates enwik8 golden test artifacts.

## Wiki Corpus Benchmarks (enwik8/enwik9)

### 1) Download corpora

```bash
python benchmarks/run_wiki_benchmarks.py --download-only --datasets enwik8
```

To also fetch `enwik9`:

```bash
python benchmarks/run_wiki_benchmarks.py --download-only --datasets enwik8,enwik9
```

Data is stored under `~/.cache/qxotic/tokenizers/corpus/`.

### 2) Run wiki JMH benchmarks

Parallel (default):

```bash
python benchmarks/run_wiki_benchmarks.py \
  --corpus enwik8 \
  --parallel true \
  --encodings r50k_base,cl100k_base,o200k_base \
  --modes encode,decode \
  --warmup 3 \
  --iterations 5 \
  --forks 1
```

Single-threaded:

```bash
python benchmarks/run_wiki_benchmarks.py \
  --corpus enwik8 \
  --parallel false \
  --encodings r50k_base,cl100k_base,o200k_base \
  --modes encode,decode
```

### 3) Interpret JMH output

JMH reports time per operation (`s/op`). Lower is better.

Example:

```text
Benchmark                     (corpus)   (encoding)  (parallel)  Mode  Cnt  Score   Error  Units
WikiEncodingBenchmark.decode    enwik8    r50k_base       false  avgt    5  0.568 +- 0.013   s/op
WikiEncodingBenchmark.encode    enwik8    r50k_base       false  avgt    5  3.476 +- 0.125   s/op
```

### Helper flags (`run_wiki_benchmarks.py`)

| Flag | Description | Default |
|------|-------------|---------|
| `--datasets` | Corpora to download (`enwik8`, `enwik9`) | `enwik8` |
| `--download-only` | Download and exit | `false` |
| `--corpus` | Corpus passed to the benchmark | `enwik8` |
| `--parallel` | `parallel` JMH parameter (`true`/`false`) | `true` |
| `--encodings` | Comma-separated encodings | `r50k_base,cl100k_base,o200k_base` |
| `--modes` | Benchmark methods (`encode`, `decode`) | `encode,decode` |
| `--warmup` | JMH warmup iterations | `3` |
| `--iterations` | JMH measurement iterations | `5` |
| `--forks` | JMH forks | `1` |

## Cross-Runtime Benchmark Flow

### One-command data collection + charts

```bash
./benchmarks/run_cross_runtime_benchmarks.sh
```

You can override scope with env vars:

```bash
BENCH_CORPORA=chat,code,json BENCH_SIZES=1k,32k BENCH_MODELS=gpt2,mistral-tekken ./benchmarks/run_cross_runtime_benchmarks.sh
```

Useful tuning knobs:

```bash
BENCH_IMPLEMENTATIONS=classic,fast PY_BENCH_DURATION_SECS=1.0 PY_BENCH_REPEATS=2 ./benchmarks/run_cross_runtime_benchmarks.sh
```

Outputs:

- Java JMH JSON: `toknroll-core/target/jmh-model-tokenizers.json`
- Python CSV: `target/python-tokenizers.csv`
- Merged CSV: `target/benchmarks/merged-cross-runtime.csv`
- Charts: `target/benchmarks/charts/*.png`

Chart metrics:

- encode: `chars/s`
- decode: `ops/s`

`ops/s` for decode avoids JMH `AuxCounters` aggregation pitfalls and keeps Java/Python units aligned.

### Manual step-by-step flow

1. Compile benchmark classes:

```bash
mvn -pl toknroll-core -DskipTests test-compile
```

2. Run Java JMH:

```bash
mvn -pl toknroll-core -Dexec.mainClass=com.qxotic.toknroll.benchmarks.TokenizerBenchmarkRunner -Dexec.classpathScope=test exec:java
```

3. Optional: convert JMH JSON to markdown:

```bash
python benchmarks/jmh_to_markdown.py \
  --input toknroll-core/target/jmh-model-tokenizers.json \
  --output toknroll-core/target/jmh-model-tokenizers.md
```

4. Run Python harness:

```bash
.venv/bin/python benchmarks/benchmark_model_tokenizers.py \
  --implementations tiktoken,tokie,hf-tokenizers,mistral-common \
  --csv target/python-tokenizers.csv
```

Common focused runs:

```bash
# GPT-2 Python libs only
.venv/bin/python benchmarks/benchmark_model_tokenizers.py \
  --models gpt2 \
  --implementations tiktoken,tokie,hf-tokenizers

# Model-native tokenizer libs only
.venv/bin/python benchmarks/benchmark_model_tokenizers.py \
  --models llama3,qwen35,mistral-tekken \
  --implementations hf-tokenizers,mistral-common
```

5. Build apples-to-apples markdown report:

```bash
.venv/bin/python benchmarks/openai_apples_report.py \
  --java-json toknroll-core/target/jmh-openai-encodings.json \
  --python-csv target/python-tokenizers.csv \
  --output target/openai-apples.md \
  --ops encode,decode
```

6. Generate charts:

```bash
.venv/bin/python benchmarks/plot_cross_runtime.py \
  --java-json toknroll-core/target/jmh-model-tokenizers.json \
  --python-csv target/python-tokenizers.csv \
  --output-dir target/benchmarks/charts \
  --merged-csv target/benchmarks/merged-cross-runtime.csv
```

## Ground-Truth Fixture Regeneration

Tokenizer fixtures:

```bash
python benchmarks/generate_ground_truth.py
```

Enwik8 corpus fixtures:

```bash
python benchmarks/generate_enwik8_ground_truth.py
```

## Java Benchmark Sources

- `toknroll-core/src/test/java/com/qxotic/toknroll/benchmarks/WikiEncodingBenchmark.java`
- `toknroll-core/src/test/java/com/qxotic/toknroll/benchmarks/WikiBenchmarkSupport.java`
- `toknroll-core/src/test/java/com/qxotic/toknroll/benchmarks/WikiCorpusPaths.java`
