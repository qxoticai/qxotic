---
sidebar_position: 3
---

# GGUF Loader

Loads Tok'n'Roll `Tokenizer` instances from GGUF files (llama.cpp format).

## Quick Start

```java
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.nio.file.Path;

GGUFTokenizerLoader loader = GGUFTokenizerLoader.createBuilderWithBuiltins().build();

// From a local .gguf file
Tokenizer t = loader.fromLocal(Path.of("/models/model.gguf"));

// From HuggingFace
Tokenizer t2 = loader.fromHuggingFace(
    "unsloth", "Llama-3.2-1B-Instruct-GGUF",
    "Llama-3.2-1B-Instruct-Q8_0.gguf");

// From ModelScope
Tokenizer t3 = loader.fromModelScope(
    "Qwen", "Qwen3-8B-GGUF",
    "qwen3-8b-Q8_0.gguf");

// From a pre-parsed GGUF instance
GGUF gguf = GGUF.read(Path.of("/models/model.gguf"));
Tokenizer t4 = loader.fromGGUF(gguf);

// Encode and decode
int[] tokens = t.encodeToArray("Hello, world!");
String decoded = t.decode(tokens);
```

## Examples by Model Family

```java
GGUFTokenizerLoader loader = GGUFTokenizerLoader.createBuilderWithBuiltins().build();

// Meta Llama 3
Tokenizer llama = loader.fromHuggingFace(
    "unsloth", "Llama-3.2-1B-Instruct-GGUF",
    "Llama-3.2-1B-Instruct-Q8_0.gguf");

// Google Gemma 4
Tokenizer gemma = loader.fromHuggingFace(
    "unsloth", "gemma-4-E2B-it-GGUF",
    "gemma-4-E2B-it-Q8_0.gguf");

// Alibaba Qwen 3
Tokenizer qwen = loader.fromHuggingFace(
    "unsloth", "Qwen3.6-35B-A3B-GGUF",
    "Qwen3.6-35B-A3B-Q8_0.gguf");

// Kimi 2.6 (from ModelScope)
Tokenizer kimi = loader.fromModelScope(
    "unsloth", "Kimi-K2.6-GGUF",
    "BF16/Kimi-K2.6-BF16-00001-of-00046.gguf");
```

## Extending

Register custom model factories, pre-tokenizers, and normalizers for unsupported model families:

```java
GGUFTokenizerLoader loader = GGUFTokenizerLoader.createBuilderWithBuiltins()
    // tokenizer.ggml.model
    .registerModelFactory("my-model", gguf -> myModel)
    // tokenizer.ggml.pre
    .registerPreTokenizer("my-pre", gguf -> mySplitter)
    .registerNormalizer("my-pre", gguf -> myNormalizer)
    .build();
```

Use `createEmptyBuilder()` to start with an empty registry and register only what you need.

### Factory Signatures

```java
// Model factory: GGUF metadata -> TokenizationModel
@FunctionalInterface
interface TokenizerModelFactory {
    TokenizationModel create(GGUF gguf);
}

// Pre-tokenizer factory: GGUF metadata -> Splitter
@FunctionalInterface
interface PreTokenizerFactory {
    Splitter create(GGUF gguf);
}

// Normalizer factory: GGUF metadata -> Normalizer
@FunctionalInterface
interface NormalizerFactory {
    Normalizer create(GGUF gguf);
}
```

## Built-in Support

| GGUF key | Model family |
|----------|-------------|
| `gpt2` | GPT-2, GPT-3.5, GPT-4 (tiktoken) |
| `llama` | Meta Llama 3+, Mistral, DeepSeek, Qwen, Gemma 4 (SentencePiece) |
| `gemma4` | Google Gemma 4 (SentencePiece) |

## Caching

Only GGUF header and metadata is downloaded (not tensor data). Cache location is configurable:

| Key | Purpose |
|-----|---------|
| `toknroll.cache.root` / `TOKNROLL_CACHE_ROOT` | Cache directory |
| `toknroll.huggingface.token` / `HF_TOKEN` | HuggingFace auth token |
| `toknroll.modelscope.token` / `MODELSCOPE_TOKEN` | ModelScope auth token |
| `toknroll.gguf.maxMetadataBytes` | Max metadata bytes to read (default 1 GiB) |
| `toknroll.gguf.connectTimeoutSeconds` | HTTP connect timeout (default 120) |
