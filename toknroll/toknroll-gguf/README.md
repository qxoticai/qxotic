# Tok'n'Roll GGUF Loader

Loads Tok'n'Roll `Tokenizer` instances from GGUF files (llama.cpp format).

## Maven

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>toknroll-gguf</artifactId>
  <version>0.1.0</version>
</dependency>
```

## Sources

Local `.gguf` files, [HuggingFace](https://huggingface.co), or [ModelScope](https://modelscope.cn).
Remote loading fetches only GGUF metadata (header + key-value pairs), not model weights.
GGUF metadata is cached on disk and reused across runs.

### Examples by model family

```java
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;

GGUFTokenizerLoader loader = GGUFTokenizerLoader
        .createBuilderWithBuiltins().build();

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

// Kimi 2.6 (ModelScope)
Tokenizer kimi = loader.fromModelScope(
        "unsloth", "Kimi-K2.6-GGUF",
        "BF16/Kimi-K2.6-BF16-00001-of-00046.gguf");
```

## Extending

Register custom model factories, pre-tokenizers, and normalizers:

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

## Configuration

| Key | Purpose |
|---|---|
| `toknroll.cache.root` / `TOKNROLL_CACHE_ROOT` | Cache directory for downloaded GGUF metadata |
| `toknroll.huggingface.token` / `HF_TOKEN` | HuggingFace authentication token |
| `toknroll.modelscope.token` / `MODELSCOPE_TOKEN` | ModelScope authentication token |

## Tested Models

Token-perfect tested against 12 model families:

- **OpenAI** - tiktoken (GPT-2, GPT-3.5, GPT-4, GPT-4o)
- **Google** - Gemma 3, Gemma 4
- **Alibaba** - Qwen 3.5+
- **Moonshot AI** - Kimi 2.5+
- **DeepSeek** - DeepSeek 3.2, v4
- **Mistral AI** - Tekken
- **IBM** - Granite 4+
- **Meta** - Llama 3+
- **Microsoft** - Phi 4+
- **HuggingFace** - SmolLM3

Other models are likely to work but are not tested against reference Python tokenizers.
