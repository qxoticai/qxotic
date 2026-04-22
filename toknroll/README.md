# Tok'n'Roll

[![Java](https://img.shields.io/badge/Java-17+-blue)](https://openjdk.org/projects/jdk/17/)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

> *The best tokenizer is no tokenizer at all; in the meantime, this library bridges the gap.*

Tok'n'Roll (`toknroll`) is a pure Java library for LLM tokenization with a focus on TikToken compatibility, predictable behavior, and good runtime performance.

This module is intentionally minimal: it provides the tokenizer API and core algorithms, but does
not ship model catalogs or downloaded tokenizer assets.

## Features

- Pure Java tokenizer core with predictable behavior
- Fast TikToken-compatible BPE implementation
- Optional adapter modules for JTokkit, GGUF, and Safetensors
- `IntSequence` API to reduce boxing and copying
- GraalVM Native Image support

## Supported families

Built-in splitter presets are available for common model families:

- Llama / Mistral / Mixtral / Phi / DBRX
- Qwen 2/3
- SmolLM
- Gemma
- Tekken / Refact / Granite

Core splitters are range-preserving partitioners (no text transformation).

## Design guarantees

- Round-trip integrity for regular text inputs: `decode(encode(text)) == text`
- Deterministic behavior: same text, same token IDs
- No hidden rewrites in core tokenizer behavior (no implicit trim/prefix/suffix injection)
- `decodeBytes(...)` is the authoritative byte-exact decode API for byte-level workflows
- Explicit opt-in transforms are supported via `Normalizer` and pipeline post-processors; these can be lossy and are generally discouraged unless you intentionally want mutated input/output token streams

## Dependency

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>toknroll</artifactId>
  <version>0.1.0</version>
</dependency>
```

Optional modules:

- `com.qxotic:toknroll-jtokkit` for `JTokkitTokenizers.fromTiktoken(...)`
- `com.qxotic:toknroll-gguf` for tokenizers from GGUF metadata
- `com.qxotic:toknroll-hf` for tokenizers from HuggingFace tokenizer files

Tip: fetch official TikToken files with `./download-tiktoken.sh`.

Python dependencies for benchmarks and model-family fixtures (`uv`, local `.venv`):

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

## Quick API examples

```java
Vocabulary vocab = Tokenizers.vocabulary(specialTokens, rankedTokens);
TokenizationModel model = Tokenizers.tikTokenModel(vocab, mergeRules);
Tokenizer tokenizer = Tokenizers.pipeline(model)
    .splitter(Splitter.regex(splitPatternRegex))
    .build();

int[] ids = tokenizer.encodeToArray("Hello world");
String text = tokenizer.decode(ids);
byte[] raw = tokenizer.decodeBytes(ids);
```

```java
TokenizationModel spModel = Tokenizers.sentencePieceBpeModel(vocab, mergeRules);
Tokenizer tokenizer = Tokenizers.pipeline(spModel).splitter(splitter).build();
```

```java
Tokenizer tokenizer = JTokkitTokenizers.fromTiktoken(name, mergeableRanks, splitPattern, specialTokens);
```

## Specials API (explicit control-token path)

For detailed behavior and design rationale, see `docs/SPECIALS.md`.

```java
Tokenizer tokenizer = Tokenizers.pipeline(model)
    .splitter(Splitter.regex(splitPatternRegex))
    .build();

// Compile once, reuse many times.
Specials specials = Specials.compile(tokenizer.vocabulary(), specialTokens.keySet());

IntSequence withSpecials = specials.encode(tokenizer, "hello <|endoftext|>");
```

## Benchmarks (JMH)

Primary benchmark: `ModelTokenizerBenchmark` under `src/test/java/com/qxotic/toknroll/benchmarks`.

Results template: `BENCHMARK_RESULTS_TEMPLATE.md`.

Run from Maven:

```bash
mvn -pl toknroll -DskipTests test-compile
mvn -pl toknroll -Dexec.mainClass=com.qxotic.toknroll.benchmarks.TokenizerBenchmarkRunner -Dexec.classpathScope=test exec:java

# Optional: convert JMH JSON to markdown
python toknroll/scripts/jmh_to_markdown.py \
  --input toknroll/target/jmh-model-tokenizers.json \
  --output toknroll/target/jmh-model-tokenizers.md
```
