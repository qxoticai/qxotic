# Tok'n'Roll HuggingFace Loader

Loads a Tok'n'Roll `Tokenizer` from a HuggingFace `tokenizer.json`.

Supports a useful subset of HuggingFace tokenizer features — enough to cover all modern
model families with token-perfect parity. Unsupported features (e.g. `post_processor`,
custom `decoders`) fail fast rather than silently degrading.

## Quick start

```java
Tokenizers t = HuggingFaceTokenizerLoader.fromLocal(Path.of("/models/my-tokenizer"));
Tokenizers t = HuggingFaceTokenizerLoader.fromHuggingFace("google", "gemma-4-e2b-it");
Tokenizers t = HuggingFaceTokenizerLoader.fromModelScope("deepseek-ai", "DeepSeek-V4-Pro");
```

`fromLocal(...)` accepts a directory or a `tokenizer.json` file path.

## Sources

Local filesystem, [HuggingFace](https://huggingface.co), or [ModelScope](https://modelscope.cn).

## Configuration

| Key | Purpose |
|---|---|
| `toknroll.cache.root` / `TOKNROLL_CACHE_ROOT` | Cache directory |
| `toknroll.huggingface.token` / `HF_TOKEN` | HuggingFace auth |
| `toknroll.modelscope.token` / `MODELSCOPE_TOKEN` | ModelScope auth |

14 model families tested token-perfect against reference implementations.
