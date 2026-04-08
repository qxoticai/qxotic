# Benchmarks

This folder contains the Python tooling used to run and compare tokenizer benchmarks across Java (JMH) and Python backends.

## Scripts

- `benchmark_model_tokenizers.py`: runs Python tokenizer benchmarks and writes CSV output.
- `jmh_to_markdown.py`: converts JMH JSON output into markdown tables.
- `openai_apples_report.py`: builds an apples-to-apples markdown report using Java JMH JSON + Python CSV.
- `plot_cross_runtime.py`: generates PNG comparison charts for Tok'n'Roll vs Python backends.
- `run_cross_runtime_benchmarks.sh`: end-to-end runner using `uv` + `.venv`.
- `generate-ground-truth.py`: regenerates tokenizer ground-truth fixtures used by tests.

## Prerequisites

- Java 17+
- Maven
- Python 3.10+
- `uv` installed locally
- Python deps installed from repo root (inside `.venv`):

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r benchmarks/requirements.txt
uv pip install --python .venv/bin/python matplotlib
```

Tip: use `.venv/bin/python` for all benchmark scripts to keep the environment isolated.

## One-command data collection + charts

From the repository root:

```bash
./benchmarks/run_cross_runtime_benchmarks.sh
```

By default, this runs a fairness-focused quick profile on `corpus=chat` for both Java and Python.
You can override scope with environment variables:

```bash
BENCH_CORPORA=chat,code,json BENCH_SIZES=1k,32k BENCH_MODELS=gpt2,mistral-tekken ./benchmarks/run_cross_runtime_benchmarks.sh
```

Useful knobs for richer and faster custom sweeps:

```bash
BENCH_IMPLEMENTATIONS=classic,fast PY_BENCH_DURATION_SECS=1.0 PY_BENCH_REPEATS=2 ./benchmarks/run_cross_runtime_benchmarks.sh
```

Outputs:

- Java JMH JSON: `toknroll-core/target/jmh-model-tokenizers.json`
- Python CSV: `target/python-tokenizers.csv`
- Merged CSV: `target/benchmarks/merged-cross-runtime.csv`
- Charts: `target/benchmarks/charts/*.png`

Default chart set includes:

- grouped comparison bars for Tok'n'Roll (`classic` + `fast`) vs TikToken / HuggingFace / Mistral
- corpus x size speedup heatmaps for each model/backend/operation
- summary speedup overview chart (`summary-speedups-modern.png`)

Chart metrics are:

- encode: `chars/s`
- decode: `ops/s`

Using `ops/s` for decode avoids JMH `AuxCounters` aggregation pitfalls and keeps Java/Python units aligned.

## Quick benchmark flow

Run from the repository root.

1) Compile Java benchmark classes:

```bash
mvn -pl toknroll-core -DskipTests test-compile
```

2) Run Java JMH benchmarks:

```bash
mvn -pl toknroll-core -Dexec.mainClass=com.qxotic.toknroll.benchmarks.TokenizerBenchmarkRunner -Dexec.classpathScope=test exec:java
```

This produces JMH JSON under `toknroll-core/target/` (for example `jmh-model-tokenizers.json` and `jmh-openai-encodings.json`).

3) Convert JMH JSON to markdown (optional):

```bash
python benchmarks/jmh_to_markdown.py \
  --input toknroll-core/target/jmh-model-tokenizers.json \
  --output toknroll-core/target/jmh-model-tokenizers.md
```

4) Run Python benchmarks (using `.venv`):

```bash
.venv/bin/python benchmarks/benchmark_model_tokenizers.py --csv target/python-tokenizers.csv
```

5) Build cross-runtime comparison report:

```bash
.venv/bin/python benchmarks/openai_apples_report.py \
  --java-json toknroll-core/target/jmh-openai-encodings.json \
  --python-csv target/python-tokenizers.csv \
  --output target/openai-apples.md \
  --ops encode,decode
```

6) Generate cross-runtime charts:

```bash
.venv/bin/python benchmarks/plot_cross_runtime.py \
  --java-json toknroll-core/target/jmh-model-tokenizers.json \
  --python-csv target/python-tokenizers.csv \
  --output-dir target/benchmarks/charts \
  --merged-csv target/benchmarks/merged-cross-runtime.csv
```

## Regenerating ground truth fixtures

If tokenizer fixtures need to be refreshed:

```bash
python benchmarks/generate-ground-truth.py
```

This updates `toknroll-core/src/test/resources/ground_truth_model_families.json` and related fixture outputs.
