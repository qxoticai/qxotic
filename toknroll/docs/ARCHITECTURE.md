# Tok'n'Roll — Architecture (summary)

This page ties the modules together. Each module has its own detailed page:

| Module | JPMS module | Java | Page |
|---|---|---|---|
| `toknroll-core` | `com.qxotic.toknroll` | 11 | [ARCHITECTURE](../toknroll-core/docs/ARCHITECTURE.md) |
| `toknroll-gguf` | `com.qxotic.toknroll.gguf` | 11 | [ARCHITECTURE](../toknroll-gguf/docs/ARCHITECTURE.md) |
| `toknroll-hf` | `com.qxotic.toknroll.hf` | 11 | [ARCHITECTURE](../toknroll-hf/docs/ARCHITECTURE.md) |
| `toknroll-vector` | (class path, `jdk.incubator.vector`) | 25 | [ARCHITECTURE](../toknroll-vector/docs/ARCHITECTURE.md) |
| `toknroll-native` | (native image → `libtoknroll`) | 25 | [ARCHITECTURE](../toknroll-native/docs/ARCHITECTURE.md) |
| `toknroll-incubator` | — | 11 | [ARCHITECTURE](../toknroll-incubator/docs/ARCHITECTURE.md) |
| `toknroll-testkit` | `com.qxotic.toknroll.testkit` | 11 | [ARCHITECTURE](../toknroll-testkit/docs/ARCHITECTURE.md) |
| `toknroll-benchmarks` | — (JMH + Python scripts, test sources only) | 17 | — |

See also [ARCHITECTURE-overview.md](ARCHITECTURE-overview.md) for the big picture, and
[SPECIALS.md](SPECIALS.md) for special-token handling.

## 1. Module dependencies

```mermaid
flowchart BT
    core["toknroll-core<br/>API + models + splitters + SPI"]
    gguf["toknroll-gguf<br/>GGUF metadata loader"]
    hf["toknroll-hf<br/>tokenizer.json loader"]
    vector["toknroll-vector<br/>TiktokenEngine (Vector API)"]
    native["toknroll-native<br/>libtoknroll (C API) + Python"]
    incubator["toknroll-incubator<br/>Ollama, JTokkit"]
    testkit["toknroll-testkit<br/>goldens + harnesses"]
    bench["toknroll-benchmarks<br/>JMH, ground-truth generators"]

    fgguf[(format-gguf)]
    fjson[(format-json)]
    fetch[(jinfer-fetch)]
    hub[(jinfer-hub)]
    jtokkit[(JTokkit)]

    gguf --> core
    hf --> core
    vector -. "ServiceLoader (TiktokenEngine)" .-> core
    native --> core & gguf & hf & vector & hub
    incubator --> core & gguf & jtokkit
    gguf --> fgguf & fetch
    hf --> fjson & fetch
    testkit --> fjson
    bench --> core & gguf & hf & testkit
    core -. tests .-> testkit
    gguf -. tests .-> testkit
    hf -. tests .-> testkit
    vector -. tests .-> testkit
```

`toknroll-core`'s `impl` package is exported **only** to `toknroll-gguf` and `toknroll-hf`
(through `ImplAccessor`); `toknroll-vector` sees only the public `spi` package.

## 2. The central abstraction: `Tokenizer`

All entry points — loaders, the factory, the native library — end up producing a `Tokenizer`.

```mermaid
classDiagram
    direction TB
    class Tokenizer {
        <<interface>>
        +vocabulary() Vocabulary
        +encodeInto(text, start, end, Builder out)
        +countTokens(text, start, end) int
        +decodeBytesInto(IntSequence, from, ByteBuffer) int
        +encode / encodeToArray / encodeUtf8
        +decode / decodeBytes / countBytes
    }
    class TokenizationModel {
        <<interface>>
    }
    class TokenizationPipeline {
        normalizer → splitter → model
    }
    class MetaspaceTokenizer {
        <<impl>> SentencePiece ▁ handling
    }
    class TiktokenModel {
        <<impl>> byte-level BPE
    }
    class SentencePieceBpeModel {
        <<impl>> SPM BPE + byte fallback
    }
    class Adapter {
        <<incubator>> JTokkit encoding
    }
    class Native_Loaded {
        <<native>> handle payload
    }
    Tokenizer <|-- TokenizationModel
    Tokenizer <|.. TokenizationPipeline
    Tokenizer <|.. MetaspaceTokenizer
    Tokenizer <|.. Adapter
    TokenizationModel <|.. TiktokenModel
    TokenizationModel <|.. SentencePieceBpeModel
    TokenizationPipeline o-- TokenizationModel
    MetaspaceTokenizer o-- TokenizationPipeline
    Native_Loaded --> Tokenizer
```

## 3. Building blocks of a pipeline (core)

```mermaid
classDiagram
    direction LR
    class TokenizationPipeline
    class Normalizer {
        <<interface>>
        identity / lowercase / unicode / sequence
    }
    class Splitter {
        <<interface>>
        identity / regex / sequence
    }
    class RegexSplitter
    class TiktokenSplitter {
        hand-written scanner for known regexes
    }
    class SentencePieceSplitter
    class TokenizationModel {
        <<interface>>
    }
    class Vocabulary {
        <<interface>>
    }
    class TokenType {
        <<interface>>
    }
    class StandardTokenType {
        <<enum>> NORMAL UNKNOWN CONTROL USER_DEFINED UNUSED BYTE
    }
    class MergeRule
    class Specials {
        <<interface>>
        matches special tokens whole
    }
    class IntSequence {
        <<interface>>
    }
    class Builder {
        <<IntSequence.Builder>>
    }
    TokenizationPipeline o-- Normalizer
    TokenizationPipeline o-- Splitter
    TokenizationPipeline o-- TokenizationModel
    Splitter <|.. RegexSplitter
    Splitter <|.. TiktokenSplitter
    Splitter <|.. SentencePieceSplitter
    TokenizationModel --> Vocabulary
    Vocabulary ..> TokenType
    TokenType <|.. StandardTokenType
    TokenizationModel ..> MergeRule : built from
    Specials ..> Vocabulary : CONTROL / USER_DEFINED / UNKNOWN
    TokenizationModel ..> Builder : writes ids
    IntSequence *-- Builder
```

## 4. Loaders: from a file format to a pipeline

```mermaid
classDiagram
    direction LR
    class HuggingFaceTokenizerLoader {
        <<toknroll-hf>>
        +fromLocal(Path)$
        +fromHuggingFace(user, repo)$
        +fromModelScope(user, repo)$
    }
    class RepositoryArtifactCache {
        <<toknroll-hf>>
    }
    class GGUFTokenizerLoader {
        <<toknroll-gguf>>
        +createBuilderWithBuiltins()$ Builder
        +fromLocal(Path)
        +fromHuggingFace(user, repo, file)
        +fromGGUF(GGUF)
    }
    class GGUFMetadataCache {
        <<toknroll-gguf>>
    }
    class OllamaTokenizerLoader {
        <<toknroll-incubator>>
        +fromLocal(String name)
    }
    class Sources {
        <<toknroll-native>>
        +load(String source)$
    }
    class ImplAccessor {
        <<core impl bridge>>
    }
    class Toknroll {
        <<core public factory>>
    }
    class Transport {
        <<jinfer-fetch>>
    }
    HuggingFaceTokenizerLoader --> RepositoryArtifactCache
    GGUFTokenizerLoader --> GGUFMetadataCache
    RepositoryArtifactCache --> Transport
    GGUFMetadataCache --> Transport
    OllamaTokenizerLoader --> GGUFTokenizerLoader
    Sources ..> HuggingFaceTokenizerLoader
    Sources ..> GGUFTokenizerLoader
    HuggingFaceTokenizerLoader ..> ImplAccessor
    GGUFTokenizerLoader ..> ImplAccessor
    HuggingFaceTokenizerLoader ..> Toknroll : pipeline()
    GGUFTokenizerLoader ..> Toknroll : pipeline()
```

| Source | Model selection | Splitter selection |
|---|---|---|
| `tokenizer.json` (HF) | `model.type == BPE`; byte-level → `TiktokenModel`, metaspace → `SentencePieceBpeModel` + `MetaspaceTokenizer` | parsed `pre_tokenizer` (Split / ByteLevel / Sequence / Metaspace) |
| `tiktoken.model` (HF) | mergeable ranks → `TiktokenModel` | `pat_str` from `tokenizer_config.json` or the repo's Python module |
| GGUF | `tokenizer.ggml.model`: `gpt2` → tiktoken, `llama`/`gemma4` → SentencePiece | `tokenizer.ggml.pre` name → builtin regex (llama.cpp names), overridable with `-Dtoknroll.gguf.pre.<name>` |
| Ollama | the model layer blob is a GGUF | as GGUF |

## 5. Performance path: the engine SPI

```mermaid
classDiagram
    direction LR
    class TiktokenModel {
        <<core impl>>
        reference encoder + memo
    }
    class TiktokenSplitter {
        <<core impl>>
        rule set id
    }
    class TiktokenEngine {
        <<core spi>>
        +encoder(TiktokenTables, int rules) Encoder
        +words(TiktokenTables) WordEncoder
    }
    class TiktokenTables {
        <<core spi>>
    }
    class VectorEngine {
        <<toknroll-vector>>
    }
    class FastEncoder {
        <<toknroll-vector>>
    }
    class BitSplit {
        SIMD regex bitstreams
    }
    class WordCache {
        lock-free word → ids
    }
    class PairMap {
        merge table
    }
    TiktokenModel ..> TiktokenSplitter : scanned pipeline
    TiktokenModel ..> TiktokenEngine : ServiceLoader, verify on sample
    TiktokenEngine ..> TiktokenTables
    TiktokenEngine <|.. VectorEngine
    VectorEngine ..> FastEncoder
    FastEncoder --> BitSplit
    FastEncoder --> WordCache
    FastEncoder --> PairMap
```

Correctness is never delegated: core checks an engine's output against its own encoder on a
sample covering every kind of piece before using it, and falls back silently otherwise
(`-Dtoknroll.engine=none` disables engines).

## 6. Native face

```mermaid
classDiagram
    direction LR
    class toknroll_h {
        <<C API>>
        load / free / encode / encode_batch / decode
        vocabulary_size / token_flags / token / id
        last_error / log_set / configure / version
    }
    class python_toknroll {
        <<Python package>>
    }
    class Native {
        <<CEntryPoints>>
    }
    class Loaded {
        <<record>> tokenizer, specials, texts
    }
    class Sources
    class Errors
    class Escape
    class Workers
    class Tokenizer {
        <<core>>
    }
    python_toknroll ..> toknroll_h : ctypes
    toknroll_h ..> Native : C shim → toknroll__*
    Native *-- Loaded
    Loaded --> Tokenizer
    Native ..> Sources
    Native ..> Errors
    Native ..> Escape
    Native ..> Workers
```

## 7. Testing architecture

```mermaid
classDiagram
    direction LR
    class PythonReference {
        <<toknroll-benchmarks scripts>>
        generate_ground_truth.py
        generate_specials_golden.py
        generate_enwik8_ground_truth.py
    }
    class Goldens {
        <<JSON fixtures>>
    }
    class FamilyGoldenFixture
    class EdgeParityGolden
    class TokenizerParityHarness
    class TokenizerInvariantHarness
    class Enwik8Corpus
    class ModuleTests {
        <<core / gguf / hf / vector tests>>
    }
    PythonReference ..> Goldens : HF tokenizers / tiktoken / llama.cpp
    FamilyGoldenFixture ..> Goldens
    EdgeParityGolden ..> Goldens
    ModuleTests ..> FamilyGoldenFixture
    ModuleTests ..> EdgeParityGolden
    ModuleTests ..> TokenizerParityHarness
    ModuleTests ..> TokenizerInvariantHarness
    TokenizerParityHarness ..> Enwik8Corpus
```
