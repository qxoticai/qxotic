# toknroll-gguf — Architecture

`toknroll-gguf` (JPMS module `com.qxotic.toknroll.gguf`) builds a Tok'n'Roll `Tokenizer` from the
`tokenizer.ggml.*` metadata of a **GGUF** model file (llama.cpp's format), locally or remotely
from Hugging Face / ModelScope. Remotely, only the GGUF **header** is fetched (one bounded HTTP
Range request), never the tensors.

Dependencies: `toknroll-core` (transitive, plus its qualified `impl` export), `com.qxotic.format.gguf`
(GGUF reader, transitive) and `com.qxotic.jinfer.fetch` (HTTP `Transport`).

| Class | Visibility | Role |
|---|---|---|
| `GGUFTokenizerLoader` | public | Entry point: registries of model factories / normalizers / pre-tokenizers, `fromLocal`, `fromHuggingFace`, `fromModelScope`, `fromGGUF` |
| `GGUFTokenizerLoader.Builder` | public | Registers factories, aliases; `build()` applies `-Dtoknroll.gguf.pre.<name>=...` overrides |
| `GGUFTokenizerLoader.UnsupportedPreTokenizerException` | public | Unknown `tokenizer.ggml.pre`, with a fix hint |
| `GGUFTokenizerDefaults` | package | The builtins: model types and the llama.cpp pre-tokenizer names → regexes |
| `GGUFTokenizerModelFactory` | package | Builds the `TokenizationModel` (tiktoken or SentencePiece) from GGUF arrays |
| `GGUFMetadataKeys` | package | `tokenizer.ggml.model / pre / tokens / merges / scores / token_type` |
| `GGUFMetadataCache` | package | On-disk cache of remote GGUF headers, `TeeReadableByteChannel` |

## 1. Loader and its registries

```mermaid
classDiagram
    direction TB
    class GGUFTokenizerLoader {
        <<final>>
        +OVERRIDE_PREFIX = "toknroll.gguf.pre."$
        -Registries registries
        +createEmptyBuilder()$ Builder
        +createBuilderWithBuiltins()$ Builder
        +fromLocal(Path ggufFile) Tokenizer
        +fromHuggingFace(user, repo, ggufPath) Tokenizer
        +fromModelScope(user, repo, ggufPath) Tokenizer
        +fromGGUF(GGUF gguf) Tokenizer
    }
    class Builder {
        <<static nested>>
        +registerModelFactory(String key, Function~GGUF,TokenizationModel~) Builder
        +registerNormalizer(String key, Function~GGUF,Normalizer~) Builder
        +registerPreTokenizer(String key, Function~GGUF,Splitter~) Builder
        +aliasPreTokenizer(String key, String target) Builder
        +build() GGUFTokenizerLoader
    }
    class Registries {
        <<private, immutable>>
        -Map modelFactories
        -Map normalizers
        -Map splitters
        -Map aliases
        -Map preFallbackByModel
        -Map normalizerFallbackByModel
    }
    class UnsupportedPreTokenizerException
    class UnsupportedTokenizerException {
        <<core>>
    }
    class GGUFTokenizerDefaults {
        <<package>>
        ~METASPACE_NORMALIZER$ Normalizer
        ~applyTo(Builder)$
    }
    class GGUFTokenizerModelFactory {
        <<package>>
        ~IGNORE_MERGES$ Set~String~
        ~buildTiktokenModel(GGUF)$ TokenizationModel
        ~buildSentencePieceModel(GGUF)$ TokenizationModel
    }
    class GGUFMetadataKeys {
        <<package>>
        ~MODEL, PRE, TOKENS, MERGES, SCORES, TOKEN_TYPE
        ~requireModel(GGUF)$ String
        ~key(GGUF, String)$ String
    }

    GGUFTokenizerLoader *-- Builder
    GGUFTokenizerLoader *-- Registries
    GGUFTokenizerLoader *-- UnsupportedPreTokenizerException
    UnsupportedTokenizerException <|-- UnsupportedPreTokenizerException
    Builder ..> Registries : build()
    GGUFTokenizerDefaults ..> Builder : registers builtins
    GGUFTokenizerDefaults ..> GGUFTokenizerModelFactory : gpt2 / llama / gemma4
    GGUFTokenizerModelFactory ..> GGUFMetadataKeys
    GGUFTokenizerLoader ..> GGUFMetadataKeys
```

Resolution layers, each outranking the previous one:

1. builtins (`createBuilderWithBuiltins()` → `GGUFTokenizerDefaults.applyTo`),
2. programmatic `register*` / `alias*` calls on the `Builder`,
3. system properties `-Dtoknroll.gguf.pre.<name>=...`, applied in `build()` (fix a deployment
   without a rebuild; replacements are logged).

Builtin model factories: `gpt2` → tiktoken byte-level BPE, `llama` and `gemma4` → SentencePiece
BPE. Pre-tokenizer names are grouped as in llama.cpp's `llama-vocab.cpp` (`gpt-2`, `llama3`/`llama-bpe`,
`qwen2`, `qwen35`, …): names in a group share one splitter factory.

## 2. From GGUF metadata to a Tokenizer

```mermaid
classDiagram
    direction LR
    class GGUF {
        <<format.gguf>>
        metadata key-values
    }
    class GGUFTokenizerLoader
    class TokenizationModel {
        <<core>>
    }
    class Splitter {
        <<core>>
    }
    class Normalizer {
        <<core>>
    }
    class TokenizationPipeline {
        <<core>>
    }
    class MetaspaceTokenizer {
        <<core impl>>
    }
    class Toknroll {
        <<core>>
    }
    GGUFTokenizerLoader ..> GGUF : reads tokenizer.ggml.*
    GGUFTokenizerLoader ..> TokenizationModel : modelFactories[model]
    GGUFTokenizerLoader ..> Splitter : splitters[alias(pre)]
    GGUFTokenizerLoader ..> Normalizer : normalizers[...]
    GGUFTokenizerLoader ..> Toknroll : pipeline(n, s, m)
    Toknroll ..> TokenizationPipeline
    GGUFTokenizerLoader ..> MetaspaceTokenizer : SPM models (via ImplAccessor)
```

`fromGGUF`:

1. `model = tokenizer.ggml.model` → model factory (else `UnsupportedTokenizerException`),
2. `pre = tokenizer.ggml.pre` (or a per-model fallback) → alias chain → splitter factory (else
   `UnsupportedPreTokenizerException` naming the `-D` override),
3. normalizer key → normalizer factory,
4. `Toknroll.pipeline(normalizer, splitter, model)`, wrapped in a `MetaspaceTokenizer` for
   SentencePiece models that use metaspace normalization.

## 3. Remote headers and the cache

```mermaid
classDiagram
    direction LR
    class GGUFMetadataCache {
        <<package final>>
        -Path cacheRoot
        -Transport transport
        -Duration downloadTimeout
        -MAX_RANGE_BYTES = 32 MB$
        ~create(Transport)$
        ~create(Path cacheRoot)$
        ~create(Path, Duration)$
        ~create(Path, Transport)$
    }
    class TeeReadableByteChannel {
        <<static nested>>
        copies the bytes read into the cache file
    }
    class Transport {
        <<jinfer.fetch>>
    }
    GGUFMetadataCache *-- TeeReadableByteChannel
    GGUFMetadataCache --> Transport
    GGUFTokenizerLoader ..> GGUFMetadataCache : fromHuggingFace / fromModelScope
```

Configuration: `-Dtoknroll.cache.root` / `TOKNROLL_CACHE_ROOT`, `HF_TOKEN` /
`-Dtoknroll.huggingface.token`, `MODELSCOPE_TOKEN` / `-Dtoknroll.modelscope.token`,
`-Dtoknroll.gguf.maxMetadataBytes` (default 32 MB: one Range request, stopped where the header
ends), `-Dtoknroll.downloadTimeoutSeconds` (300).
