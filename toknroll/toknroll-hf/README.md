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

Token-perfect against each model's own tokenizer, in both [modes](#raw-text-and-special-tokens), tested against tokenizers from 17 model vendors:

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

## Raw text and special tokens

`encode` takes its input as raw text: no added token is matched, special or not, as with tiktoken's `encode_ordinary`.
`Specials.of(tokenizer).encode(tokenizer, text)` matches every added token in the text and substitutes its id, as Hugging Face's `encode` does.
Use it for text you wrote, such as a rendered chat template, never for a user's text.
Each mode is parity-tested against its own reference: `encode` against the reference with its added tokens switched off, `Specials` against the reference as it is.
