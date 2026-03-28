# Quixotic Tokenizers

[![Java](https://img.shields.io/badge/Java-17+-blue)](https://openjdk.org/projects/jdk/17/)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

> The best tokenizer is no tokenizer at all; in the meantime, this library bridges the gap.

`tokenizers` is a pure Java library for LLM tokenization with a focus on TikToken compatibility, predictable behavior, and good runtime performance.

## Features

- TikToken-compatible BPE tokenization
- Backed by [JTokkit](https://github.com/knuddelsgmbh/jtokkit) for fast production paths
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

TikToken encoding fixtures used in tests include `r50k_base`, `p50k_base`, `cl100k_base`, and `o200k_base`.

## Usage

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>tokenizers</artifactId>
  <version>0.1.0</version>
</dependency>
```

Tip: you can fetch official TikToken files with `./download-tiktoken.sh`.

