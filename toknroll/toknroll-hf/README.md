# Tok'n'Roll HuggingFace Loader

Loads a Tok'n'Roll `Tokenizer` from a HuggingFace `tokenizer.json`.

Unsupported features (e.g. `post_processor`, custom `decoders`) fail fast
instead of introducing silent incompatibilities.

## Maven

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>toknroll-hf</artifactId>
  <version>0.1.0</version>
</dependency>
```

## Quick Start

```java
// From local files (directory or tokenizer.json path)
Tokenizer t = HuggingFaceTokenizerLoader
                .fromLocal(Path.of("/models/gemma-4-e2b-it"));

// From HuggingFace
Tokenizer t = HuggingFaceTokenizerLoader
        .fromHuggingFace("google", "gemma-4-e2b-it");

// From ModelScope
Tokenizer t = HuggingFaceTokenizerLoader
        .fromModelScope("deepseek-ai", "DeepSeek-V4-Pro");

// Encode and decode
int[] tokens = t.encodeToArray("Hello, world!");
String decoded = t.decode(tokens);
```

`fromLocal(...)` accepts a model directory (containing `tokenizer.json`) or a direct path to a `tokenizer.json` file.

Authenticated repos require a HuggingFace token via `HF_TOKEN` env var or system property `toknroll.huggingface.token`.

## Sources

Local filesystem, [HuggingFace](https://huggingface.co), or [ModelScope](https://modelscope.cn).

Remote loading fetches the `tokenizer.json` (and optional `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`). If `tokenizer.json` is not found, the loader falls back to `tiktoken.model` reconstruction.

Artifacts are cached on disk and reused across runs.

## Configuration

| Key | Purpose |
|---|---|
| `toknroll.cache.root` / `TOKNROLL_CACHE_ROOT` | Cache directory for downloaded artifacts |
| `toknroll.huggingface.token` / `HF_TOKEN` | HuggingFace authentication token |
| `toknroll.modelscope.token` / `MODELSCOPE_TOKEN` | ModelScope authentication token |

## Tested Models

Token-perfect tested against 10 model families:

- **OpenAI** - tiktoken (GPT-2, GPT-3.5, GPT-4, GPT-4o)
- **Google** - Gemma 3, Gemma 4
- **Alibaba** - Qwen 3.5+
- **Moonshot AI** - Kimi 2.5+
- **DeepSeek** - DeepSeek 3.2, DeepSeek 4
- **Mistral AI** - Tekken
- **IBM** - Granite 4+
- **Meta** - Llama 3+
- **Microsoft** - Phi 4+
- **HuggingFace** - SmolLM3

Other models are likely to work but are not tested against reference Python tokenizers.

