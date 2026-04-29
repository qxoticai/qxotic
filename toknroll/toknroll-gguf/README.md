# Tok'n'Roll GGUF Loader

Loads Tok'n'Roll `Tokenizers` instances from GGUF files (llama.cpp format).

## Quick start

```java
GGUFTokenizerLoader loader = GGUFTokenizerLoader.createBuilderWithBuiltins().build();

Tokenizers t = loader.fromLocal(Path.of("/models/model.gguf"));
Tokenizers t = loader.fromHuggingFace("unsloth", "Llama-3.2-1B-Instruct-GGUF",
    "Llama-3.2-1B-Instruct-Q8_0.gguf");
```

## Sources

Local `.gguf` files, [HuggingFace](https://huggingface.co), or [ModelScope](https://modelscope.cn).
Remote loading fetches only GGUF metadata, not model weights.

## Extending

```java
GGUFTokenizerLoader loader = GGUFTokenizerLoader.createBuilderWithBuiltins()
    .registerModelFactory("my-model", gguf -> myModel)
    .registerPreTokenizer("my-pre", gguf -> mySplitter)
    .registerNormalizer("my-pre", gguf -> myNormalizer)
    .build();
```

Use `createEmptyBuilder()` to start with an empty registry.

## Configuration

| Key | Purpose |
|---|---|
| `toknroll.cache.root` / `TOKNROLL_CACHE_ROOT` | Cache directory |
| `toknroll.huggingface.token` / `HF_TOKEN` | HuggingFace auth |
| `toknroll.modelscope.token` / `MODELSCOPE_TOKEN` | ModelScope auth |

12 model families tested token-perfect against reference implementations.
