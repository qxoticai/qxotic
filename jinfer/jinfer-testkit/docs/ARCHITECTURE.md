# jinfer-testkit — class architecture

Package `com.qxotic.jinfer.testkit` (6 public classes, ~700 lines). Shared **test fixtures** for the Jinfer build:
checkpoint lookup by ref (never a download), a weightless `LanguageModel`, two reusable contract checkers
(last-row prefill, `MediaProjector`), and two small process utilities (resident set size, scoped system property).
It is packaged as a plain jar (not a test-jar) so its dependencies resolve transitively; every consumer declares it
with `<scope>test</scope>`, so it never reaches a shipped artifact. Not published to Maven Central.

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends;
classes marked `<<external>>` live outside this package. There is **no SPI registration** (`META-INF/services`)
in this module: every class is called statically or instantiated directly by tests.

1. [Position in the build](#1-position-in-the-build)
2. [Class diagrams](#2-class-diagrams) — 2a overview, 2b `TestLanguageModel`
3. [Model resolution — `TestModels`](#3-model-resolution--testmodels)
4. [Contract checker — `LastRowPrefill`](#4-contract-checker--lastrowprefill)
5. [Contract checker — `MediaProjectorContract`](#5-contract-checker--mediaprojectorcontract)
6. [Process utilities — `ProcessRss`, `SystemProperty`](#6-process-utilities)
7. [Consumers](#7-consumers)

| Class | Role |
|---|---|
| `TestModels` | The one place tests acquire checkpoints, by ref (`repo:QUANT` or `repo/file.gguf`). Lookup-only via `ModelStore`; abort/fail policy driven by system properties; per-ref path override; `fixture(name)` for media under `test-fixtures/` |
| `TestLanguageModel` | Weightless `LanguageModel<Configuration, Void, State>` that always emits `'x'`; byte-level 256-token `TOKENIZER`; nested `Configuration` record, `State` (extends `ContextState`), anonymous `CheckpointCodec` |
| `LastRowPrefill` | Contract check: a prefill built with `Batch.Outputs.LAST` must produce the same logits (within tolerance) as `Batch.Outputs.ALL`, over the prompt and 8 forced decode steps; `COUNTING` prompt constant |
| `MediaProjectorContract` | Contract check for any `MediaProjector`: row count = `positions`, dense FP32 `[rows, modelDim]` finite chunks, chunk lifetime scoped to `project`, chunk-size bound, determinism, rejection of `maxChunkSize=0` |
| `ProcessRss` | This JVM's resident set size in KB: `/proc/self/status` (Linux), `ps` (macOS), `tasklist` (Windows); empty elsewhere |
| `SystemProperty` | `AutoCloseable` override of one system property, restored (or cleared) on `close()` |

---

## 1. Position in the build

`pom.xml` dependencies: `toknroll-core`, `jinfer-core`, `jota-core`, `jota-memory`, `jinfer-hub`,
`junit-jupiter-api`, `opentest4j` (JUnit assertions and `TestAbortedException` are used from *main* sources).

```mermaid
flowchart TB
    subgraph consumers["Consumer test suites (scope test)"]
        Models["jinfer-llama / lfm2 / lfm3 / gemma4 / qwen3 / qwen35 / ..."]
        Integr["jinfer-chat / cli / server / langchain4j / spring-ai / ..."]
    end
    subgraph tk["jinfer-testkit"]
        TM["TestModels"]
        TLM["TestLanguageModel"]
        LRP["LastRowPrefill"]
        MPC["MediaProjectorContract"]
        RSS["ProcessRss"]
        SP["SystemProperty"]
    end
    subgraph hub["jinfer-hub"]
        MS["ModelStore"]
        SRC["HuggingFaceSource / ModelScopeSource"]
    end
    subgraph core["jinfer-core"]
        LM["LanguageModel / ContextState / CheckpointCodec / Batch / Views / Arenas"]
        MP["media.MediaProjector / media.Media"]
    end
    subgraph jota["jota-core / jota-memory"]
        MV["MemoryView / MemoryArena / DataType"]
    end
    TOK["toknroll-core: Tokenizer / Vocabulary"]
    JU["junit-jupiter-api / opentest4j"]

    Models --> TM
    Models --> LRP
    Models --> MPC
    Integr --> TM
    Integr --> TLM
    TM --> MS --> SRC
    TLM --> LM
    TLM --> TOK
    LRP --> LM
    MPC --> MP
    TLM --> MV
    MPC --> MV
    TM --> JU
    LRP --> JU
    MPC --> JU
```

---

## 2. Class diagrams

### 2a. Overview

All classes are `final`; the static-utility ones (`TestModels`, `LastRowPrefill`, `MediaProjectorContract`,
`ProcessRss`) have a private constructor.

```mermaid
classDiagram
    direction LR
    class LanguageModel~C,W,S~ { <<interface, external>> }
    class MediaProjector~R~ { <<interface, external>> }
    class ModelStore { <<external>> }
    class AutoCloseable { <<interface, external>> }
    class TestAbortedException { <<external>> }

    class TestModels {
        <<final, utility>>
        -OVERRIDE_PREFIX "jinfer.testModel."
        +require(ref)$ Path
        +find(ref)$ Optional
        +fixture(name)$ Path
        ~resolve(ref, store, props)$ Optional
        ~checkoutModels(start)$ Optional
        -lookup(ref)$ Optional
        -fixtureStore(root)$ ModelStore
        -checkout(start)$ Optional
        -requirePinned(ref)$
    }
    class TestLanguageModel {
        <<final>>
        +TOKENIZER Tokenizer$
    }
    class LastRowPrefill {
        <<final, utility>>
        -STEPS = 8
        +COUNTING String$
        +assertMatchesFullPass(model, ids, tolerance)$
        +assertMatchesFullPass(model, prompt, rows, tolerance)$
        -run(model, prompt, rows)$ List
        -maxDrift(a, b)$ double
        -tokens(ids, at)$ Function
    }
    class MediaProjectorContract {
        <<final, utility>>
        +assertContract(projector, media, modelDim)$
        -projectAndCheck(projector, media, positions, modelDim, checkExpiry)$
    }
    class ProcessRss {
        <<final, utility>>
        +supported()$ boolean
        +unsupportedReason()$ String
        +kilobytes()$ OptionalLong
        -fromProc()$
        -fromPs()$
        -fromTasklist()$
        ~run(timeout, command)$ String
    }
    class SystemProperty {
        <<final>>
        -key
        -previous
        +override(key, value)$ SystemProperty
        +close()
    }

    LanguageModel <|.. TestLanguageModel
    AutoCloseable <|.. SystemProperty
    TestModels ..> ModelStore : standard() / of(root, sources)
    TestModels ..> TestAbortedException : missing model
    LastRowPrefill ..> LanguageModel : drives
    MediaProjectorContract ..> MediaProjector : checks
```

### 2b. `TestLanguageModel` and its nested types

```mermaid
classDiagram
    direction TB
    class LanguageModel~C,W,S~ { <<interface, external>> }
    class ContextConfiguration { <<interface, external>> }
    class ContextState { <<abstract, external>> }
    class CheckpointCodec~S~ { <<abstract, external>> }
    class Tokenizer { <<interface, external>> }
    class Vocabulary { <<interface, external>> }
    class MemoryArena~T~ { <<interface, external>> }

    class TestLanguageModel {
        <<final>>
        +TOKENIZER Tokenizer$
        +configuration() Configuration
        +weights() Void
        +newState(context, batch) State
        +newState(context, batch, memory) State
        +ingest(State, Batch)
        +logits(State, output) MemoryView
        +checkpointCodec() Optional
    }
    class Configuration {
        <<record>>
        vocabularySize = 256
        maxContextLength = 16384
    }
    class State {
        <<static final>>
        -MemorySegment tokens
        -MemoryView logits
        #clearHistory()
        ~ingest(Batch)
    }
    class AnonCodec {
        <<anonymous CheckpointCodec>>
        #sizeOf(positions) long
        #transfer(state, from, to, memory, capture)
    }
    class AnonTokenizer {
        <<anonymous Tokenizer>>
        vocabulary()
        encodeInto() UTF-8 bytes
        countTokens()
        decodeBytesInto()
    }
    class AnonVocabulary {
        <<anonymous Vocabulary>>
        size() 256
        token(id) id(text)
        isTokenOfType() NORMAL only
    }

    LanguageModel <|.. TestLanguageModel
    ContextConfiguration <|.. Configuration
    ContextState <|-- State
    CheckpointCodec <|-- AnonCodec
    Tokenizer <|.. AnonTokenizer
    Vocabulary <|.. AnonVocabulary
    TestLanguageModel ..> Configuration : configuration()
    TestLanguageModel ..> State : newState
    TestLanguageModel ..> AnonCodec : checkpointCodec()
    TestLanguageModel *-- AnonTokenizer : TOKENIZER
    AnonTokenizer *-- AnonVocabulary
    State --> MemoryArena : allocates tokens + logits
    AnonCodec ..> State : copies tokens rows
```

Behaviour verified in the code:

- `newState(context, batch)` uses `Arenas.newCrossThreadMemoryArena()` and passes `owns = true`; the arena overload
  passes `owns = false`.
- `State` allocates `context * 4` bytes of token ids and a 256-float logits view (`Views.allocateF32`) whose index
  `'x'` is set to `1.0` once: the model always prefers `'x'`.
- `ingest` runs `state.ingest(batch)` under `state.exclusively(...)`: it writes the `Batch.Input.Tokens` ids at
  `position()` and calls `advanceContext(batch.count(), batch.outputs())`. `logits(state, output)` returns the same
  fixed view for any output.
- `clearHistory()` zero-fills the token buffer. The codec stores `positions * Integer.BYTES` bytes and copies the
  `[from, to)` slice of `tokens` to/from the checkpoint segment, so checkpoint/restore protocols can be tested.
- `TOKENIZER` is byte-level: encoding emits the UTF-8 bytes (`b & 255`), decoding writes token ids back as bytes;
  every id is `StandardTokenType.NORMAL`.

---

## 3. Model resolution — `TestModels`

A ref **must be pinned**: its last `/` segment contains `:` (quant form, e.g. `unsloth/gemma-4-E2B-it-GGUF:Q8_0`) or
ends with `.gguf` (exact companion file, e.g. `.../mmproj-F32.gguf`); otherwise `IllegalArgumentException`.

```mermaid
flowchart TD
    A["require ref / find ref"] --> NM{"jinfer.test.noModels?"}
    NM -- true --> F1["AssertionError: model lookup in the model-free suite"]
    NM -- false --> PIN{"ref pinned? ':' or .gguf"}
    PIN -- no --> F2["IllegalArgumentException: specify the quant"]
    PIN -- yes --> OV{"-Djinfer.testModel.LAST-SEGMENT set?"}
    OV -- yes --> EX{"path exists?"}
    EX -- no --> F3["IllegalArgumentException: stale override"]
    EX -- yes --> P1["print 'served by' to stderr, return path"]
    OV -- no --> L1["ModelStore.standard.find ref<br/>jinfer cache then HF hub cache"]
    L1 -- found --> OK["Path"]
    L1 -- empty --> L2["checkoutModels: parent-of-git-checkout/models<br/>ModelStore.of root, HuggingFaceSource, ModelScopeSource"]
    L2 -- found --> OK
    L2 -- empty --> MISS{"caller"}
    MISS -- "find" --> E["Optional.empty"]
    MISS -- "require" --> RM{"jinfer.test.requireModels?"}
    RM -- true --> F4["AssertionError with the fix"]
    RM -- false --> AB["TestAbortedException: test skipped, with the fix"]
```

Notes:

- `find(ref)` under `jinfer.test.requireModels=true` delegates to `require`, so a probe also fails on absence.
- The git checkout is found by walking up from the working directory to the first directory containing `.git`
  (file or directory, so worktrees count).
- `fixture(name)` resolves `<checkout>/test-fixtures/<name>` (or relative to the cwd if no checkout); a missing file
  raises `TestAbortedException`, or `AssertionError` under `jinfer.test.requireModels`.
- Package-private overloads `require/resolve/fixture(..., store, props)` inject the store and property source so
  `TestModelsTest` runs in memory without touching global state.

---

## 4. Contract checker — `LastRowPrefill`

Checks that a model which skips work for non-final rows when `Batch.Outputs.LAST` is requested still yields the same
logits, KV caches and histories as a pass with `Batch.Outputs.ALL`.

```mermaid
sequenceDiagram
    participant T as Model test
    participant L as LastRowPrefill
    participant M as LanguageModel
    participant S as ContextState

    T->>L: assertMatchesFullPass(model, ids, tolerance)
    L->>L: warm run, split at n/2, ALL
    L->>L: split drift = maxDrift(run whole ALL, run split ALL)
    loop at in n, n/2
        L->>L: assertMatchesFullPass(model, tokens(ids, at), n, max(tolerance, split))
        Note over L: warm ALL, warm LAST, then full=ALL vs LAST
    end
    rect rgba(128,128,128,0.1)
    Note over L,S: run(model, batches, rows)
    L->>M: newState(rows + 8, rows)
    M-->>L: S (try-with-resources)
    loop each prompt batch
        L->>M: ingest(state, batch)
    end
    loop step 0..8
        L->>M: logits(state)
        L->>L: copy vocab floats via Views
        L->>M: ingest(state, Batch.step(100 + step))
    end
    end
    L->>T: assertTrue(drift <= tolerance)
```

- `tokens(ids, at)` yields a `Function<Batch.Outputs, List<Batch>>`: `[0, at)` and `[at, n)` as
  `Batch.Input.Tokens` batches, empty parts dropped. With `at = n/2` the second batch starts mid-context and reads
  what a skipping first batch wrote.
- Each comparison runs both modes once untimed first ("a cold JIT tier alone moves logits by up to 0.3").
- `COUNTING` is the string `"1, 2, ..., 400"`, a prompt any length can be cut from.

---

## 5. Contract checker — `MediaProjectorContract`

```mermaid
sequenceDiagram
    participant T as Vision/audio test
    participant C as MediaProjectorContract
    participant P as MediaProjector
    T->>C: assertContract(projector, media, modelDim)
    C->>P: positions(media)
    Note over C: assert >= 1, planId() stable
    loop two runs, first checks expiry
        C->>P: project(media, positions, sink)
        loop each chunk
            P-->>C: sink.accept(MemoryView)
            Note over C: segment-backed, scope alive, dense FP32,<br/>rank 2, dim1 == modelDim, 1 <= rows <= positions,<br/>all finite, copy to float array
        end
        Note over C: sum rows == positions
        Note over C: run 1 only: every borrowed view now NOT alive
    end
    Note over C: both runs bitwise identical
    C->>P: project(media, 0, noop)
    P--xC: IllegalArgumentException expected
```

The checker validates the *shape* of the contract only; per-port numerics stay in each port's own tests. Its
Javadoc advises keeping toy dimensions asymmetric (a projector-dim == vision-dim toy hid a Qwen3.5 merger bug).

---

## 6. Process utilities

```mermaid
flowchart LR
    K["ProcessRss.kilobytes"] --> SUP{"supported? linux / mac / windows"}
    SUP -- no --> E["OptionalLong.empty"]
    SUP -- linux --> PR["fromProc: VmRSS line of /proc/self/status"]
    SUP -- mac --> PS["fromPs: ps -o rss= -p PID"]
    SUP -- windows --> TL["fromTasklist: tasklist CSV, digits of last field"]
    PS --> RUN["run timeout 10s, command"]
    TL --> RUN
    RUN --> FILE["output redirected to a temp file<br/>waitFor timeout, destroyForcibly on overrun<br/>null on timeout or non-zero exit"]
```

- Any `IOException`/`RuntimeException` yields empty; `InterruptedException` re-sets the interrupt flag and yields
  empty. Gates use `unsupportedReason()` to `assume` the check away on other platforms.
- Output goes to a file rather than a pipe so the wait stays bounded even if a child keeps stdout open.

`SystemProperty.override(key, value)` sets the property (or clears it when `value` is null), remembers the previous
value, and `close()` restores it or clears it if there was none. Nested overrides unwind in order through
try-with-resources.

---

## 7. Consumers

Modules whose test sources import each class (grep of `com.qxotic.jinfer.testkit.<Class>`):

| Class | Used by |
|---|---|
| `TestModels` | almost every model and integration module: llama, lfm2, lfm3, gemma4, gemma-embedding2, qwen3, qwen35, mellum, maple, nemotronh, inflect2, kokoro, parakeet, decision, native, cli, server, langchain4j, spring-ai (+ autoconfigure, starter), examples |
| `TestLanguageModel` | jinfer-chat, jinfer-cli, jinfer-server, jinfer-langchain4j, jinfer-spring-ai |
| `LastRowPrefill` | jinfer-gemma4, jinfer-lfm2, jinfer-lfm3 |
| `MediaProjectorContract` | jinfer-gemma4, jinfer-lfm2, jinfer-qwen35 |
| `SystemProperty` | jinfer-chat, jinfer-parakeet, jinfer-qwen35 |
| `ProcessRss` | jinfer-langchain4j |

The module's own tests (`TestModelsTest`, `SystemPropertyTest`, `ProcessRssTest`) cover ref pinning, overrides,
abort-vs-fail policy, checkout fixture lookup, property restoration and the bounded process wait.
