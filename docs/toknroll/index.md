---
sidebar_position: 1
---

# Tok'n'Roll

Pure Java, fast, composable tokenization for LLMs with token-perfect parity, matching reference Python implementations.

Tok'n'Roll is designed to be simple and efficient. It runs on Java 11+, requires no native bindings, supports GraalVM Native Image.

Key properties:

- **Token-perfect parity.** Matches reference Python tokenizer implementations.
- **Compatible with industry standards.** Spawn Tok'n'Roll tokenizers from HuggingFace's `tokenizer.json` and llama.cpp's GGUF model files.
- **Zero-allocation, zero-copy APIs.** `encodeInto(...)`, `decodeBytesInto(...)`, and `IntSequence` support low-GC hot paths.
- **Fast and efficient.** Performant BPE implementations, with strong algorithmic guarantees and optimized fast-paths, competitive with native alternatives.
- **Portable.** Pure Java, 11+, GraalVM's Native Image compatible.

## Modules

| Module | Artifact | Purpose |
|--------|----------|---------|
| [Core](/toknroll/core) | `toknroll-core` | Extensible tokenizer API with efficient Tiktoken and SentencePiece BPE implementations |
| [HuggingFace](/toknroll/hf) | `toknroll-hf` | Load tokenizers from `tokenizer.json` (HuggingFace / ModelScope) |
| [GGUF](/toknroll/gguf) | `toknroll-gguf` | Load tokenizers from GGUF files (llama.cpp format) |

## Installation

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs>
  <TabItem value="maven" label="Maven">

```xml
<!-- Core API (zero dependencies) -->
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>toknroll-core</artifactId>
  <version>0.1.0</version>
</dependency>

<!-- HuggingFace tokenizer.json loading -->
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>toknroll-hf</artifactId>
  <version>0.1.0</version>
</dependency>

<!-- GGUF (llama.cpp) tokenizer loading -->
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>toknroll-gguf</artifactId>
  <version>0.1.0</version>
</dependency>
```

  </TabItem>
  <TabItem value="gradle" label="Gradle">

```groovy
implementation 'com.qxotic:toknroll-core:0.1.0'
implementation 'com.qxotic:toknroll-hf:0.1.0'
implementation 'com.qxotic:toknroll-gguf:0.1.0'
```

  </TabItem>
  <TabItem value="mill" label="Mill">

```scala
mvn"com.qxotic::toknroll-core:0.1.0"
mvn"com.qxotic::toknroll-hf:0.1.0"
mvn"com.qxotic::toknroll-gguf:0.1.0"
```

  </TabItem>
</Tabs>

## Quick Start

### Load and tokenize

```java
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.hf.HuggingFaceTokenizerLoader;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.nio.file.Path;

// HuggingFace tokenizer.json format
Tokenizer t = HuggingFaceTokenizerLoader
    .fromHuggingFace("google", "gemma-4-e2b-it");

// GGUF format (llama.cpp)
GGUFTokenizerLoader ggufLoader = GGUFTokenizerLoader
    .createBuilderWithBuiltins()
    .build();

Tokenizer t2 = ggufLoader.fromHuggingFace(
    "unsloth", "Llama-3.2-1B-Instruct-GGUF",
    "Llama-3.2-1B-Instruct-Q8_0.gguf");

// Local files
Tokenizer t3 = HuggingFaceTokenizerLoader
    .fromLocal(Path.of("/models/gemma-4-e2b-it"));

Tokenizer t4 = ggufLoader
    .fromLocal(Path.of("/models/model.gguf"));

// ModelScope
Tokenizer t5 = HuggingFaceTokenizerLoader
    .fromModelScope("deepseek-ai", "DeepSeek-V4-Pro");

// Encode / decode
int[] tokens = t.encodeToArray("Hello, world!");
String decoded = t.decode(tokens);
int count = t.countTokens("How many tokens is this?");
```

### Zero-allocation hot path

```java
import com.qxotic.toknroll.IntSequence;
import java.nio.ByteBuffer;

IntSequence.Builder out = IntSequence.newBuilder(128);
t.encodeInto("Hello, world!", out);

ByteBuffer bytes = ByteBuffer.allocate(256);
int consumed = t.decodeBytesInto(tokens, 0, bytes); // no partial-token writes
```

### Build from scratch

```java
import java.util.regex.Pattern;
import com.qxotic.toknroll.*;

// Build a vocabulary from your model's ranked tokens
Vocabulary vocab = Toknroll.vocabulary(specialTokens, rankedTokens);
TokenizationModel model = Toknroll.tiktokenModel(vocab, mergeRules);

// Choose a regex splitter (e.g. cl100k_base pattern for GPT-4)
Splitter splitter = Splitter.regex(Pattern.compile(
    "(?i:'(?:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}|"
    + " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+)"));

// Assemble the pipeline
Tokenizer tokenizer = Toknroll.pipeline(splitter, model);

// Use it
int[] tokens = tokenizer.encodeToArray("Hello world");
```

## toknroll@qxoticai CLI

The [toknroll CLI](https://github.com/qxoticai/qxotic/blob/main/toknroll/scripts/toknroll.java) allows to encode/decode/count by loading
tokenizers from HuggingFace/ModelScope, arbitrary URLs, GGUF files, local or remote.

```bash
# Pass input directly with --input
jbang toknroll@qxoticai --source google/gemma-4-e2b-it --input "Hello, Tok'n'Roll 🎸"
# 9259
# 236764
# 14873
# 236789
# 236749
# 236789
# 47201
# 236743
# 248290

# Or pipe text via stdin
echo "Hello, World\!" | jbang toknroll@qxoticai --source Qwen/Qwen3.6-35B-A3B
# 9259
# 236764
# 4109
# 236888

# Any source: HuggingFace, ModelScope, GGUF, local
jbang toknroll@qxoticai \
    --source modelscope:deepseek-ai/DeepSeek-V4-Pro \
    --input "Hello, DeepSeek-V4-Pro 🐋"
# 19923
# 14
# 22651
# 4374
# 1465
# 13582
# 22
# 59819
# 7351
# 241
# 236

# GGUF from huggingface.co
jbang toknroll@qxoticai --source unsloth/granite-4.1-3b-GGUF/Q8_0 --input "Hello, Granite 4.1"
# 9906
# 11
# 65594
# 220
# 19
# 13
# 16

# Decode: token IDs → text
echo "22177 1044 42301 2784 1033" | jbang toknroll@qxoticai --decode --source mistralai/Mistral-Medium-3.5-128B
# Hello, Mistral!

# Count tokens
echo "Hello, World\!" | jbang toknroll@qxoticai --count --source google/gemma-4-e2b-it
# 4
```

The `--source` flag accepts several formats:

| Source | Loads | Example |
|--------|-------|---------|
| `user/repo` | HF tokenizer.json | `google/gemma-4-e2b-it` |
| `user/repo/Q8_0` | GGUF metadata from HF | `unsloth/Llama-3.2-1B-Instruct-GGUF/Q8_0` |
| `ms:user/repo` | ModelScope | `ms:Qwen/Qwen3-4B` |
| `./path` or `/path` | Local .gguf or HF dir | `./model.gguf` |
| Full URL | hf.co / modelscope.cn | `https://huggingface.co/user/repo` |

First run caches the tokenizer on disk. Public models need no API key.

## Tested Models

Token-perfect parity tested against reference Python implementations:

- **OpenAI** tiktoken (GPT-2, GPT-3.5, GPT-4, GPT-4o)
- **Google** Gemma 3, Gemma 4
- **Alibaba** Qwen 3.5+
- **Moonshot AI** Kimi 2.5+
- **DeepSeek** DeepSeek 3.2, v4
- **Mistral AI** Tekken
- **IBM** Granite 4+
- **Meta** Llama 3+
- **Microsoft** Phi 4+
- **HuggingFace** SmolLM3

See the module pages for exact feature support and known limitations:

- [`/toknroll/core`](/toknroll/core): core API contracts and composition model
- [`/toknroll/hf`](/toknroll/hf): supported `tokenizer.json` features and strict-failure behavior
- [`/toknroll/gguf`](/toknroll/gguf): built-in GGUF model families and registry extension points

## Configuration

| Key | Purpose |
|-----|---------|
| `toknroll.cache.root` / `TOKNROLL_CACHE_ROOT` | Cache directory for downloaded artifacts |
| `toknroll.huggingface.token` / `HF_TOKEN` | HuggingFace authentication token |
| `toknroll.modelscope.token` / `MODELSCOPE_TOKEN` | ModelScope authentication token |
