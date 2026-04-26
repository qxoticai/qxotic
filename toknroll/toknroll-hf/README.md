# Tok'n'Roll HuggingFace Loader (`toknroll-hf`)

`toknroll-hf` loads tokenizers in HuggingFace tokenizer format and builds a Tok'n'Roll `Tokenizer`.

Supported sources:

- local filesystem (`tokenizer.json`)
- remote HuggingFace repositories
- remote ModelScope repositories

The loader is strict by design: unsupported tokenizer features fail fast.

Spartan design note: raw tokenizer behavior only, with no automatic BOS/EOS injection, no
`post_processor` execution, and no implicit special-token insertion policy.

## Quick usage

```java
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.hf.HuggingFaceTokenizerLoader;
import java.nio.file.Path;

Tokenizer local = HuggingFaceTokenizerLoader.fromLocal(Path.of("/models/my-tokenizer"));

Tokenizer gemma = HuggingFaceTokenizerLoader.fromHuggingFace("google", "gemma-4-e2b-it");

Tokenizer qwen =
        HuggingFaceTokenizerLoader.fromHuggingFace("Qwen", "Qwen3.6-35B-A3B");

Tokenizer ms = HuggingFaceTokenizerLoader.fromModelScope("deepseek-ai", "DeepSeek-V4-Pro");
```

`fromLocal(...)` accepts either:

- a directory containing `tokenizer.json`
- a direct `tokenizer.json` file path

## Tested models

"Tested" here means token-perfect parity on enwik8 vs HuggingFace tokenizer outputs:

- encoded token IDs match golden HF outputs exactly (full-corpus chunk)
- decode/token lookup parity is also validated by the parity harness

### Token-perfect parity on enwik8 (with HF tokenizer output)

- `Qwen/Qwen3.5-0.8B`
- `unsloth/Llama-3.2-1B-Instruct`
- `deepseek-ai/DeepSeek-V3.2`
- `deepseek-ai/DeepSeek-V4-Pro`
- `moonshotai/Kimi-K2.6`
- `HuggingFaceTB/SmolLM3-3B`
- `ibm-granite/granite-4.1-8b`
- `mistralai/ministral-8b-instruct-2410`
- `zai-org/GLM-5.1`
- `MiniMaxAI/MiniMax-M2.7`
- `XiaomiMiMo/MiMo-V2-Flash`
- `microsoft/phi-4`
- `openai/gpt-oss-20b`
- `google/gemma-4-e2b-it`

## Cache

Downloaded artifacts are stored on disk.

- `useCacheOnly=true`: only cached artifacts are used; cache miss fails
- `forceRefresh=true`: re-downloads even if cached
- default (`useCacheOnly=false`, `forceRefresh=false`): reuses cache
- cache root configuration:
  - Java system property (highest priority): `toknroll.cache.root`
  - environment variable: `TOKNROLL_CACHE_ROOT`
  - resolution order: system property -> env var -> OS default cache directory

Examples:

```bash
export TOKNROLL_CACHE_ROOT="/data/toknroll-cache"
```

```bash
java -Dtoknroll.cache.root="/data/toknroll-cache" -jar app.jar
```

## Remote auth configuration

- HuggingFace token:
  - system property: `toknroll.huggingface.token`
  - env var: `HF_TOKEN`
- ModelScope token:
  - system property: `toknroll.modelscope.token`
  - env var: `MODELSCOPE_TOKEN`

## Remote artifact loading strategy

For both HuggingFace and ModelScope:

1. Try `tokenizer.json`.
2. If `tokenizer.json` returns HTTP 404, fallback to `tiktoken.model`.
3. Opportunistically fetch optional files if present:
   - `tokenizer_config.json`
   - `special_tokens_map.json`
   - `added_tokens.json`

When `tiktoken.model` fallback is used, the loader performs a best-effort lookup to find the
additional data needed to reconstruct the tokenizer accurately.

If required data cannot be resolved, fallback loading fails fast.
