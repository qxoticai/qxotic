# toknroll-incubator — Architecture

`toknroll-incubator` holds experimental loaders and adapters that are not part of the stable API.
It depends on `toknroll-core`, `toknroll-gguf`, a JSON parser and **JTokkit** (a third-party Java
tiktoken implementation, used for comparisons).

| Package | Class | Role |
|---|---|---|
| `com.qxotic.toknroll.incubator` | `OllamaTokenizerLoader` | Loads the tokenizer of a model pulled by Ollama, from its local store |
| `com.qxotic.toknroll.jtokkit` | `JTokkitTokenizers` | Exposes a JTokkit encoding as a Tok'n'Roll `Tokenizer` |

## 1. Ollama loader

```mermaid
classDiagram
    direction LR
    class OllamaTokenizerLoader {
        <<final>>
        -GGUFTokenizerLoader ggufLoader
        -Pattern NAME$
        -Pattern DIGEST$
        +OllamaTokenizerLoader()
        +OllamaTokenizerLoader(GGUFTokenizerLoader)
        +fromLocal(String name) Tokenizer
        +fromLocal(Path store, String name) Tokenizer
        ~modelBlob(Path store, String name)$ Path
    }
    class GGUFTokenizerLoader {
        <<toknroll-gguf>>
        +fromLocal(Path) Tokenizer
    }
    class OllamaStore {
        <<filesystem>>
        manifests/host/namespace/model/tag
        blobs/sha256-digest
    }
    OllamaTokenizerLoader --> GGUFTokenizerLoader : delegates
    OllamaTokenizerLoader ..> OllamaStore : manifest → model layer blob
```

A name as `ollama run` takes it (`[host/][namespace/]model[:tag]`, defaults
`registry.ollama.ai` / `library` / `latest`, case-insensitive) resolves through the manifest to the
blob of the layer `application/vnd.ollama.image.model`, which is a GGUF file handed to
`GGUFTokenizerLoader.fromLocal`. The store is `$OLLAMA_MODELS` or `~/.ollama/models`; nothing is
downloaded.

## 2. JTokkit adapter

```mermaid
classDiagram
    direction LR
    class JTokkitTokenizers {
        <<final, static>>
        +fromTiktoken(name, Map mergeableRanks, Pattern split)$ Tokenizer
        +fromTiktoken(name, Map mergeableRanks, Pattern split, Map specials)$ Tokenizer
        +fromTiktoken(name, Map mergeableRanks, String split)$ Tokenizer
        +fromTiktoken(name, Map mergeableRanks, String split, Map specials)$ Tokenizer
    }
    class Adapter {
        <<private static nested>>
        -Encoding encoding
        -Vocabulary vocabulary
    }
    class Tokenizer {
        <<core interface>>
    }
    class Encoding {
        <<jtokkit>>
    }
    JTokkitTokenizers *-- Adapter
    Tokenizer <|.. Adapter
    Adapter --> Encoding
    JTokkitTokenizers ..> Encoding : registers mergeable ranks + pattern
```

Useful as a reference/baseline in benchmarks and parity tests against Tok'n'Roll's own tiktoken
implementation.
