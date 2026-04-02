# Tok’n’Roll

[![Java](https://img.shields.io/badge/Java-17+-blue)](https://openjdk.org/projects/jdk/17/)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

> *The best tokenizer is no tokenizer at all; in the meantime, this library bridges the gap.*

Tok’n’Roll (`toknroll`) is a pure Java library for LLM tokenization with a focus on TikToken compatibility, predictable behavior, and good runtime performance.

This module is intentionally minimal: it provides the tokenizer API and core algorithms, but does
not ship model catalogs or downloaded tokenizer assets.

## Features

- TikToken-compatible BPE tokenization
- Optional [JTokkit](https://github.com/knuddelsgmbh/jtokkit) adapter module for compatibility paths
- Includes a classic GPT-2 style BPE implementation
- Strong test coverage against reference tokenizers
- `IntSequence` API to avoid unnecessary boxing/copying
- GraalVM Native Image support

## Supported families

Built-in splitter presets are available for common model families such as:

- Llama / Mistral / Mixtral / Phi / DBRX
- Qwen 2/3
- SmolLM
- Gemma
- Tekken / Refact / Granite

Splitters in core `toknroll` are range-preserving partitioners (no text transformation).

TikToken encoding fixtures used in tests include `r50k_base`, `p50k_base`, `cl100k_base`, and `o200k_base`.

## Design guarantees

- Fidelity-first API: `decode(encode(text)) == text` for regular text inputs
- Deterministic behavior: same text, same token IDs
- No hidden rewrites in core tokenizer behavior (no implicit trim/prefix/suffix injection)
- `decodeBytes(...)` is the authoritative lossless decode API for byte-level workflows

## Usage

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>toknroll</artifactId>
  <version>0.1.0</version>
</dependency>
```

Optional utility modules:

- `com.qxotic:toknroll-jtokkit` for `Tokenizers.fromTiktoken(...)` via JTokkit-backed compatibility
- `com.qxotic:toknroll-gguf` for building tokenizers from GGUF metadata (bring your own `.gguf`)
- `com.qxotic:toknroll-safetensors` for building tokenizers from HuggingFace tokenizer files (bring your own `tokenizer.json` and companion files)

Tip: you can fetch official TikToken files with `./download-tiktoken.sh`.

Ground-truth fixtures are generated with `./generate-ground-truth.py`:
- `src/test/resources/ground_truth_tokens.json` for OpenAI tokenizer families
- `src/test/resources/ground_truth_model_families.json` for modern model families (Gemma 3, Qwen 3.5, Phi-4, Mistral Tekken) when optional Python dependencies are installed

Python dependencies for benchmarks and model-family fixtures (`uv`, local `.venv`):

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

### Quick API examples

```java
Tokenizer tokenizer = Tokenizers.fromTiktoken(name, mergeableRanks, splitPatternRegex);

int[] ids = tokenizer.encodeToArray("Hello world");
String text = tokenizer.decode(ids);
byte[] raw = tokenizer.decodeBytes(ids);
```

```java
Tokenizer tokenizer = Tokenizers.fromTiktoken(name, mergeableRanks, splitPattern, specialTokens);
```

```java
Tokenizer classic = Tokenizers.classicBpe(mergeableRanks, specialTokens, splitter);
```

## Benchmarks (JMH)

JMH benchmarks are available under `src/test/java/com/qxotic/toknroll/benchmarks`.

- `ModelTokenizerBenchmark`: end-to-end model tokenizer throughput (`encode`, `encodeInto`,
  `decode`, `countTokens`) for:
  - implementations: `jtokkit` (baseline), `classic`, `fast`
  - `gpt2`
  - `llama3`
  - `qwen35`
  - `mistral-tekken`

Results capture template: `BENCHMARK_RESULTS_TEMPLATE.md`.

- `FastTikTokenMergeBenchmark`: merge-loop-only throughput on raw bytes (no regex splitting,
  normalization, or UTF-8 transcoding) to isolate small/large BPE path performance.

- `OpenAiEncodingBenchmark`: OpenAI encoding-only benchmark (`r50k_base`, `cl100k_base`,
  `o200k_base`) for tighter apples-to-apples encode comparisons.

Run from Maven:

```bash
mvn -pl toknroll -DskipTests test-compile
mvn -pl toknroll -Dexec.mainClass=com.qxotic.toknroll.benchmarks.TokenizerBenchmarkRunner -Dexec.classpathScope=test exec:java

# Optional: convert JMH JSON to markdown tables
python toknroll/scripts/jmh_to_markdown.py \
  --input toknroll/target/jmh-model-tokenizers.json \
  --output toknroll/target/jmh-model-tokenizers.md

# OpenAI encode-only apples run (Java + Python) and merged markdown report
mvn -pl toknroll -DskipTests exec:java \
  -Dexec.mainClass=com.qxotic.toknroll.benchmarks.TokenizerBenchmarkRunner \
  -Dexec.classpathScope=test \
  -Dexec.args='OpenAiEncodingBenchmark.encode -p implementation=jtokkit,fast -p encoding=r50k_base,cl100k_base,o200k_base -p corpus=chat,code,json -p size=1k,32k -wi 1 -i 2 -w 1s -r 1s -f 0 -rf json -rff toknroll/target/jmh-openai-encode.json'

python toknroll/benchmark_model_tokenizers.py \
  --backends tiktoken,tokie \
  --models gpt2,llama3,qwen35 \
  --corpora chat,code,json \
  --sizes 1k,32k \
  --warmup 0.2 \
  --duration 0.8 \
  --repeats 2 \
  --csv toknroll/target/python-bench-openai-apples.csv

python toknroll/scripts/openai_apples_report.py \
  --java-json toknroll/target/jmh-openai-encode.json \
  --python-csv toknroll/target/python-bench-openai-apples.csv \
  --output toknroll/target/openai-apples-report.md
```
