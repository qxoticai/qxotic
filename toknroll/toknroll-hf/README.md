# Tok'n'Roll HuggingFace Loader

Loads a Tok'n'Roll `Tokenizer` from a HuggingFace `tokenizer.json`.

Unsupported features fail fast instead of introducing silent incompatibilities: an unknown model, normalizer or pre-tokenizer, an added token Hugging Face would match other than by its exact text (`lstrip`, `rstrip`, `single_word`, or `normalized` under a normalizer that changes it), a SentencePiece pre-tokenizer stage or decoder step toknroll cannot reproduce.
The `post_processor` is not read: encoding is pure tokenization, with no BOS or EOS added.

## Maven

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>toknroll-hf</artifactId>
  <version>0.3.1</version>
</dependency>
```

## Quick Start

```java
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.hf.HuggingFaceTokenizerLoader;
import java.nio.file.Path;

// From local files (directory or tokenizer.json path)
Tokenizer t = HuggingFaceTokenizerLoader
        .fromLocal(Path.of("/models/gemma-4-e2b-it"));

// From HuggingFace
Tokenizer hf = HuggingFaceTokenizerLoader
        .fromHuggingFace("google", "gemma-4-e2b-it");

// From ModelScope
Tokenizer ms = HuggingFaceTokenizerLoader
        .fromModelScope("deepseek-ai", "DeepSeek-V4-Pro");

// Encode and decode
int[] tokens = hf.encodeToArray("Hello, world!");
String decoded = hf.decode(tokens);
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

Token-perfect against each model's own tokenizer, with the [known divergence](#known-divergence) below, tested against tokenizers from 17 model vendors:

- **OpenAI** - tiktoken (GPT-2, GPT-3.5, GPT-4, GPT-4o), gpt-oss
- **Google** - Gemma 4
- **Alibaba** - Qwen 3.5+
- **Liquid AI** - LFM 2.5, LFM 3
- **OpenBMB** - MiniCPM5
- **Moonshot AI** - Kimi 2.5+, Kimi K3
- **DeepSeek** - DeepSeek 3.2, DeepSeek 4
- **Mistral AI** - Tekken
- **IBM** - Granite 4+
- **Meta** - Llama 3+
- **HuggingFace** - SmolLM3
- **NVIDIA** - Nemotron 3
- **Z.ai** - GLM 5.1
- **MiniMax** - M2.7
- **Xiaomi** - MiMo V2
- **Poolside** - Laguna XS 2.1, Laguna S 2.1
- **inclusionAI** - Ling 3.0, Ling 3.1

Other models are likely to work but are not tested against reference Python tokenizers.

## Known divergence

Plain `encode` treats non-special added tokens (such as `<think>`, or DeepSeek's `<｜User｜>`) as text, while Hugging Face matches them as tokens.
`Specials.of(tokenizer).encode` is the path that matches them as Hugging Face does.
