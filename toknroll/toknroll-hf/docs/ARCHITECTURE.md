# toknroll-hf — Architecture

`toknroll-hf` (JPMS module `com.qxotic.toknroll.hf`) builds a Tok'n'Roll `Tokenizer` from
**Hugging Face tokenizer assets** (`tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json`, `added_tokens.json`, or a `tiktoken.model` + pattern module), from a
local folder or fetched from Hugging Face / ModelScope and cached on disk. It only fetches
tokenizer metadata, never weights.

Dependencies: `toknroll-core` (transitive + qualified `impl` export), `com.qxotic.format.json`
(JSON parser), `com.qxotic.jinfer.fetch` (HTTP `Transport`).

| Class | Visibility | Role |
|---|---|---|
| `HuggingFaceTokenizerLoader` | public | Static entry points + the whole `tokenizer.json` → pipeline translation (≈1.5k lines) |
| `RepositoryArtifactCache` | package | Downloads and caches repository files (revision → commit SHA), HF / ModelScope endpoints, tokens |

## 1. Public API

```mermaid
classDiagram
    direction LR
    class HuggingFaceTokenizerLoader {
        <<final, static>>
        +fromLocal(Path modelDirOrTokenizerJson)$ Tokenizer
        +fromHuggingFace(String user, String repo)$ Tokenizer
        +fromHuggingFace(user, repo, revision, useCacheOnly, forceRefresh)$ Tokenizer
        +fromModelScope(String user, String repo)$ Tokenizer
        +fromModelScope(user, repo, revision, useCacheOnly, forceRefresh)$ Tokenizer
        ~loadFromTiktokenModel(...)$ Tokenizer
        ~parsePreTokenizer(Object json)$ Splitter
        ~parsePatStrFromPythonModule(String)$ String
    }
    class RepositoryArtifactCache {
        <<package final>>
        -Path cacheRoot
        -Transport transport
        -Duration downloadTimeout
        -String huggingFaceEndpoint
        ~create(Transport)$
        ~create(Path cacheRoot)$
        ~create(Path, Duration)$
        ~create(Path, String endpoint)$
    }
    class Transport {
        <<jinfer.fetch>>
    }
    class Tokenizer {
        <<core>>
    }
    HuggingFaceTokenizerLoader ..> RepositoryArtifactCache : remote sources
    RepositoryArtifactCache --> Transport
    HuggingFaceTokenizerLoader ..> Tokenizer : returns
```

Cache policy: a cached artifact is reused; `forceRefresh` re-downloads; `useCacheOnly` never
touches the network. Configuration: `-Dtoknroll.cache.root` / `TOKNROLL_CACHE_ROOT`, `HF_TOKEN`,
`MODELSCOPE_TOKEN`, `-Dtoknroll.downloadTimeoutSeconds`.

## 2. Translating `tokenizer.json`

The loader maps each section of `tokenizer.json` onto a core building block, and **fails fast**
(`UnsupportedTokenizerException`) on anything it does not reproduce exactly.

```mermaid
classDiagram
    direction TB
    class tokenizer_json {
        <<JSON>>
        normalizer
        pre_tokenizer
        model (type, vocab, merges)
        decoder
        added_tokens
    }
    class Normalizer {
        <<core>>
    }
    class Splitter {
        <<core>>
    }
    class Vocabulary {
        <<core>>
    }
    class TiktokenModel {
        <<core impl>>
        byte-level BPE
    }
    class SentencePieceBpeModel {
        <<core impl>>
        metaspace BPE
    }
    class MetaspaceTokenizer {
        <<core impl>>
    }
    class TokenizationPipeline {
        <<core>>
    }
    class TokenEntries {
        <<private nested>>
        String[] tokens
        int[] tokenTypes
    }
    class Span {
        <<private nested>>
        int start, end
    }
    tokenizer_json ..> Normalizer : parseNormalizer (Sequence, Replace, NFC/NFKC, Lowercase)
    tokenizer_json ..> Splitter : parsePreTokenizer (ByteLevel, Split, Sequence, Metaspace, Digits)
    tokenizer_json ..> TokenEntries : buildTokenEntries (vocab + added_tokens)
    TokenEntries ..> Vocabulary
    tokenizer_json ..> TiktokenModel : BPE + ByteLevel
    tokenizer_json ..> SentencePieceBpeModel : BPE + Metaspace / ▁ spelling
    SentencePieceBpeModel --o MetaspaceTokenizer : wrapped
    Normalizer --o TokenizationPipeline
    Splitter --o TokenizationPipeline
    TiktokenModel --o TokenizationPipeline
    Splitter ..> Span : Split behaviours (isolated, merged_with_previous/next...)
```

Decision points of `loadUnchecked`:

* `model.type` must be `BPE` (untyped files: merges ⇒ BPE, else Unigram / WordPiece ⇒ unsupported).
* Metaspace pre-tokenizer or `▁`-spelled vocabulary ⇒ **SentencePiece style**:
  `SentencePieceBpeModel` + `SentencePieceSplitter`, wrapped in `MetaspaceTokenizer` with the
  `prepend_scheme` and a decoder-derived `trimLeadingSpace`.
* Otherwise ⇒ **byte-level** (`ByteLevel` pre-tokenizer + decoder required): `TiktokenModel`
  behind the parsed splitter (regexes are recognised by core's `TiktokenSplitter`, so the fast scanned
  path and the vector engine apply).
* No `tokenizer.json` (HTTP 404) but a `tiktoken.model` ⇒ `loadFromTiktokenModel`: mergeable ranks
  + the `pat_str` read from `tokenizer_config.json` or from the repository's Python module
  (`auto_map`).
