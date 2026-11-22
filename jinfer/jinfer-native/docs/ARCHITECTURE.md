# jinfer-native — class architecture

Package `com.qxotic.jinfer.libjinfer` (8 classes, ~1740 lines of Java) plus one C file
(`src/main/c/jinfer_shim.c`, 77 lines) and the public header `include/jinfer.h`. It builds **libjinfer**: jinfer as a
GraalVM native-image shared library with a C API. The C shim forwards each `jinfer_*` header function to a
`jinfer__*` `@CEntryPoint` in `Native`, which resolves `ObjectHandle`s to Java objects (`Model`, `State`, `Stream`)
and turns every exception into a thread-local last error plus a NULL / negative return. The module has no
`META-INF/services` registrations of its own; model kinds are discovered at run time through jinfer-chat's `Models`
(ServiceLoader over `jinfer-models-all`, a runtime dependency). Its only native-image resource is
`native-image.properties`: `--initialize-at-run-time=com.qxotic.jinfer.libjinfer`.

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends;
classes marked `<<external>>` live outside this package.

1. [Position in the stack](#1-position-in-the-jinfer-stack)
2. [Class diagrams](#2-class-diagrams) — 2a handles and ownership, 2b errors and options
3. [C to Java call path](#3-c-to-java-call-path)
4. [Model loading](#4-model-loading)
5. [Streams: chat, speech, transcription](#5-streams)
6. [Raw state door](#6-raw-state-door)
7. [Embed and rerank](#7-embed-and-rerank)
8. [Lifetime, reference counting and memory](#8-lifetime-reference-counting-and-memory)
9. [Error mapping](#9-error-mapping)

| Class | Role |
|---|---|
| `Native` | `public final`; every `@CEntryPoint` (`jinfer__last_error`, `jinfer__model_*`, `jinfer__state_*`, `jinfer__stream_*`, `jinfer__chat`, `jinfer__speak`, `jinfer__transcribe`, `jinfer__embed`, `jinfer__rerank`, `jinfer__prepare`, `jinfer__error_set`); handle lookup via global `ObjectHandles`; `collect()` forces a GC in the image only |
| `Model` | A loaded model of one `Kind` (`LANGUAGE`, `EMBEDDING`, `RERANKING`, `SPEECH`, `TRANSCRIPTION`); owns a `ChatEngine` or a weights `Arena` + face; reference-counted; embed/rerank over one shared `ContextState`; properties and `info` JSON |
| `State` | The raw door: one `ContextState` over `Model.contextModel()`; `ingest`, `logits`, `embedding`, `reset`, `position` |
| `Stream` | Every generation: producer thread pushes text / PCM chunks into a `LinkedBlockingQueue`, the caller steps; factories `chat`, `speech`, `transcription`, plus `prepare` |
| `Options` | `@CContext` + `@CStruct("jinfer_model_options")` view read through `jinfer.h`; `Values` record (companions, prompt cache, context capacity) |
| `Errors` | Result codes (`OK`, `DONE`, `ERROR`, `EINVAL`, `ENOTFOUND`, `EUNSUPPORTED`, `ECANCELLED`, `EBUSY`), `Failure` exception, thread-local last code, message in a C TLS buffer via `@CFunction jinfer__error_buffer` |
| `Exclusive` | `AtomicBoolean` guard: a concurrent second call on one state/stream is `EBUSY` |
| `Resample` | Lanczos (a = 3) sample-rate conversion for `jinfer_transcribe` input |

---

## 1. Position in the jinfer stack

```mermaid
flowchart TB
    Host["C / C++ / Python host<br/>include/jinfer.h, python/jinfer"]
    subgraph lib["libjinfer shared library"]
        Shim["jinfer_shim.c<br/>jinfer_* to jinfer__*"]
        Iso["toknroll_isolate.c<br/>shared isolate, configure, log"]
        subgraph pkg["com.qxotic.jinfer.libjinfer"]
            N["Native"]
            M["Model"]
            S["State"]
            St["Stream"]
        end
        TN["toknroll libtoknroll.Native<br/>toknroll_* entry points"]
    end
    subgraph chat["jinfer-chat"]
        CE["ChatEngine"]
        Ms["Models / ModelProvider"]
        LE["LoadedEmbedder / LoadedReranker"]
    end
    subgraph server["jinfer-server"]
        OAI["OpenAiChat / ServerConfig"]
    end
    subgraph core["jinfer-core"]
        CM["ContextModel / ContextState / LanguageModel / EmbeddingModel"]
        SP["SpeechSynthesisModel / TranscriptionModel"]
        AR["Arenas / RuntimeFlags"]
    end
    Hub["jinfer-hub ModelStore"]
    Cache["jinfer-cache PromptCache"]
    GGUF["gguf GGUF"]
    Models["jinfer-models-all (runtime)"]
    JAM["jam-native / jam-vector / jam-scalar (runtime)"]

    Host --> Shim --> N
    Shim --> Iso
    N --> M & S & St
    N --> TN
    M --> CE & Ms & LE & Hub & Cache & GGUF & AR
    St --> OAI & CE & SP
    S --> CM
    Ms -.->|ServiceLoader| Models
    CM -.-> JAM
```

`pom.xml` compile dependencies: `jinfer-core`, `jinfer-chat`, `jinfer-hub`, `jinfer-codecs`, `jinfer-server`
(for `OpenAiChat`), `jinfer-cache`, `gguf`, `toknroll-core`, `toknroll-native` (libjinfer also carries the
`toknroll_*` API), `json`; `nativeimage` is `provided`; `jinfer-models-all`, `jam-native`, `jam-vector`,
`jam-scalar` are `runtime`; `jinfer-testkit` is `test`.

---

## 2. Class diagrams

### 2a. Handles and ownership

```mermaid
classDiagram
    class Native {
        <<final>>
        -ObjectHandles HANDLES$
        -get(ObjectHandle, Class, String) T$
        ~collect()$
        ~modelLoad() ObjectHandle$
        ~modelFree()$
        ~stateNew() ObjectHandle$
        ~chat() ObjectHandle$
        ~speak() ObjectHandle$
        ~transcribe() ObjectHandle$
        ~embed() int$
        ~rerank() int$
        ~prepare() int$
        ~streamStep() int$
    }
    class Model {
        <<final>>
        ~Kind kind
        ~String name
        ~Path path
        ~Map companions
        ~ChatEngine engine
        ~LoadedEmbedder embedder
        ~LoadedReranker reranker
        ~SpeechSynthesisModel speech
        ~TranscriptionModel transcription
        ~int capacity
        -Arena arena
        -AtomicInteger references
        -ExecutorService driver
        -ContextState shared
        ~load(String, Values) Model$
        ~retain()
        ~release() boolean
        ~driver() ExecutorService
        ~engine() ChatEngine
        ~contextModel() ContextModel
        ~tokenizer() Tokenizer
        ~embed(List, int) float[]
        ~rerank(String, List, String) float[]
        ~property(int) int
        ~info() CCharPointerHolder
    }
    class Kind {
        <<enumeration>>
        LANGUAGE
        EMBEDDING
        RERANKING
        SPEECH
        TRANSCRIPTION
    }
    class State {
        <<final>>
        -Model model
        -ContextModel cm
        -ContextState state
        ~Exclusive exclusive
        ~ingest(int[]) int
        ~logits() float[]
        ~embedding() float[]
        ~reset()
        ~position() int
        ~close()
    }
    class Stream {
        <<final>>
        -Model model
        -LinkedBlockingQueue chunks
        ~Exclusive exclusive
        -boolean cancelled
        -boolean finished
        ~String text
        ~float[] samples
        ~String result
        ~chat(Model, String) Stream$
        ~speech(Model, String) Stream$
        ~transcription(Model, float[], int) Stream$
        ~prepare(Model, String)$
        ~step() int
        ~cancel()
        ~close()
    }
    class Exclusive {
        <<final>>
        -AtomicBoolean busy
        ~enter()
        ~leave()
    }
    class Resample {
        <<final>>
        ~to(float[], int, int) float[]$
    }
    class ChatEngine
    <<external>> ChatEngine
    class LoadedEmbedder
    <<external>> LoadedEmbedder
    class LoadedReranker
    <<external>> LoadedReranker
    class SpeechSynthesisModel
    <<external>> SpeechSynthesisModel
    class TranscriptionModel
    <<external>> TranscriptionModel
    class ContextModel
    <<external>> ContextModel
    class ContextState
    <<external>> ContextState
    class ObjectHandles
    <<external>> ObjectHandles
    class OpenAiChat
    <<external>> OpenAiChat

    Model *-- Kind : nested
    Native --> ObjectHandles : HANDLES
    Native ..> Model : load / handle
    Native ..> State : new / handle
    Native ..> Stream : chat, speak, transcribe
    State --> Model : retain / release
    Stream --> Model : retain until finish
    State *-- Exclusive
    Stream *-- Exclusive
    State --> ContextState
    State --> ContextModel
    Model --> ChatEngine : LANGUAGE
    Model --> LoadedEmbedder : EMBEDDING
    Model --> LoadedReranker : RERANKING
    Model --> SpeechSynthesisModel : SPEECH
    Model --> TranscriptionModel : TRANSCRIPTION
    Model --> ContextState : shared, embed/rerank
    Stream ..> OpenAiChat : lower / completion
    Stream ..> Resample : transcription
```

Handles given to C are `ObjectHandles.getGlobal()` words. `Native.get` checks the handle is non-zero and of the
expected class, so passing a state handle where a model is expected is `EINVAL`, not a crash.

### 2b. Errors and options

```mermaid
classDiagram
    class Errors {
        <<final>>
        ~int OK$
        ~int DONE$
        ~int ERROR$
        ~int EINVAL$
        ~int ENOTFOUND$
        ~int EUNSUPPORTED$
        ~int ECANCELLED$
        ~int EBUSY$
        -ThreadLocal LAST$
        -messageBuffer(int) CCharPointer$
        ~clear()$
        ~code(Throwable) int$
        ~set(Throwable) int$
        ~last() CCharPointer$
        ~lastCode() int$
        ~empty() CCharPointer$
        ~invalid(String) Failure$
        ~unsupported(String) Failure$
    }
    class Failure {
        ~int code
    }
    class RuntimeException
    <<external>> RuntimeException
    class Options {
        <<final>>
        ~read(Struct) Values$
    }
    class Directives {
        +getHeaderFiles() List
        +getOptions() List
    }
    class Struct {
        <<interface>>
        structSize() long
        companions() CCharPointerPointer
        companionCount() int
        promptCache() CCharPointer
        contextCapacity() int
    }
    class Values {
        <<record>>
        companions Map
        promptCache String
        contextCapacity int
    }
    class CContext_Directives
    <<external>> CContext_Directives
    class PointerBase
    <<external>> PointerBase

    Errors *-- Failure : nested
    Failure --|> RuntimeException
    Options *-- Directives : nested
    Options *-- Struct : nested
    Options *-- Values : nested
    Directives ..|> CContext_Directives
    Struct --|> PointerBase
    Options ..> Errors : invalid
```

`Options.Directives` reads `-Djinfer.native.include=<dir of jinfer.h>:<dir of toknroll.h>` at image build time, so
the struct layout is taken from `jinfer.h` itself. `Options.read` returns `Values.DEFAULTS` for a NULL struct,
refuses a `struct_size` smaller than the compiled struct, and parses each companion as `"<capability>=<ref>"`.

---

## 3. C to Java call path

```mermaid
sequenceDiagram
    participant H as C host
    participant Sh as jinfer_shim.c
    participant I as toknroll_isolate.c
    participant N as Native
    participant E as Errors
    participant O as Model / State / Stream

    H->>Sh: jinfer_state_ingest(s, ids, n)
    Sh->>I: toknroll__thread()
    I-->>Sh: graal_thread* or NULL
    alt no runtime
        Sh-->>H: JINFER_ERROR
    else
        Sh->>N: jinfer__state_ingest(t, s, ids, n)
        N->>E: clear()
        N->>N: get(handle, State.class)
        N->>O: exclusive.enter()
        N->>O: ingest(copy)
        N->>O: exclusive.leave()
        alt throws
            N->>E: set(t) - code + message into C TLS buffer
            N-->>H: negative code
        else
            N-->>H: position
        end
    end
```

The shim (`jinfer_shim.c`) knows nothing about models: every function attaches through `toknroll__thread()`
(the isolate is shared with the `toknroll_*` symbols in the same library), returns NULL / `""` / `JINFER_ERROR`
when there is no runtime, and never starts the runtime for a NULL free. `jinfer_last_error`, `jinfer_log_set` and
`jinfer_configure` delegate to `toknroll__last_error_of`, `toknroll__log` and `toknroll__configure`, passing
`jinfer__last_error` / `jinfer__error_set` as callbacks. `jinfer_model_options_init` is pure C (zero + `struct_size`).

Float outputs (`logits`, `embedding`, `embed`, `rerank`) follow an snprintf convention (`Native.floats`): write
`min(n, capacity)`, return `n`; `embed` and `rerank` compute the size first (`embedSize`, `rerankSize`) and return it
without running the model when the buffer is too small.

---

## 4. Model loading

```mermaid
flowchart TD
    A["Native.modelLoad(source, options)"] --> B["Options.read(struct)"]
    B --> C["Model.load / load0"]
    C --> D{"path-shaped ref<br/>missing?"}
    D -->|yes| X["NoSuchFileException -> ENOTFOUND"]
    D -->|no| E["ModelStore.standard().resolveAll(source + companions)"]
    E --> F["header(path): GGUF general.* metadata"]
    F --> G["PromptCache.Options: sessions, budget, catalog, contextCapacity"]
    G --> H["new ChatEngine(path, companions, cache)"]
    H -->|ok| L["Model LANGUAGE"]
    H -->|IncompatibleModelException| I["Arenas.newCrossThread()"]
    I --> J{"Models.retrieval(path)"}
    J -->|EMBEDDING| K1["Models.loadEmbedder -> Model EMBEDDING"]
    J -->|RERANKING| K2["Models.loadReranker -> Model RERANKING"]
    J -->|empty| T["Models.loadTranscription"]
    T -->|ok| K3["Model TRANSCRIPTION"]
    T -->|incompatible| S["Models.loadSpeech"]
    S -->|ok| K4["Model SPEECH"]
    S -->|incompatible| R["rethrow the language refusal -> EUNSUPPORTED"]
    L & K1 & K2 & K3 & K4 --> Z["HANDLES.create(model)"]
```

The model file selects the face, as the CLI server does. A language model's `capacity` is the engine's resolved
`contextCapacity()`; embedding / reranking models resolve it with `PromptCache.Options.contextCapacity`; speech and
transcription have `capacity = 0` (no token state). On a failure in `loadOther` the arena is closed best-effort.

---

## 5. Streams

```mermaid
sequenceDiagram
    participant H as C host
    participant N as Native
    participant S as Stream
    participant P as producer thread
    participant CE as ChatEngine / speech / transcription

    H->>N: jinfer_chat(model, request_json)
    N->>S: Stream.chat(model, json)
    S->>S: parse + nulls, OpenAiChat.lower(wire, name, CONFIG, defaults)
    S->>S: new Stream(model) - model.retain()
    S->>P: produce(body, engine::stream)
    N-->>H: stream handle
    P->>CE: engine.complete(request, ReplySink)
    CE-->>P: Delta (CONTENT channel only)
    P->>S: chunks.add(text)
    loop until DONE
        H->>N: jinfer_stream_step
        N->>S: step() - chunks.take()
        N-->>H: JINFER_OK
        H->>N: jinfer_stream_text / samples
    end
    P->>S: result = OpenAiChat.completion(...), chunks.add(END)
    H->>N: jinfer_stream_step
    S->>S: finish() - model.release()
    N-->>H: JINFER_DONE
    H->>N: jinfer_stream_result / free
```

- **chat**: the request JSON is the OpenAI `chat.completions` dialect, lowered by jinfer-server's `OpenAiChat`
  with `Model.CONFIG` (`ServerConfig.local(0)` with custom `Limits`, no request deadline). Requests asking for
  `Modality.AUDIO` output are refused (`EUNSUPPORTED`). The producer runs on `ChatEngine::stream`; the result is
  the `chat.completion` JSON.
- **speak**: reads `input` (required) and optional positive `speed`; `SpeechSynthesisModel.speak` pushes
  `float[]` PCM chunks; the result is `{sample_rate, samples, seconds}`. Runs on `Model.driver()`, a single
  daemon thread per model.
- **transcribe**: copies PCM, `Resample.to` the model's rate, `TranscriptionModel.transcribe(Media.Audio)`, one text
  chunk; the result is `{text, words[{text,start,end,confidence}]}`. Also on `Model.driver()`.
- **prepare** (`jinfer_prepare`): lowers the request and calls `ChatEngine.definePrompt` to compute the prompt
  into the cache, no stream.

```mermaid
stateDiagram-v2
    [*] --> Running : factory, model.retain
    Running --> Running : step = OK, text or samples
    Running --> Cancelled : cancel from any thread
    Cancelled --> Finished : step drains to END, ECANCELLED once
    Running --> Finished : END - DONE, or Throwable - error code
    Finished --> Finished : step = DONE
    Finished --> [*] : free
    Running --> [*] : free = cancel + drain + collect
```

`step()` is `synchronized` and also guarded by `Exclusive`. `cText`, `cResult` and `cSamples` (malloc'd via
`UnmanagedMemory`) are borrowed C copies owned by the stream and freed on the next call or in `close()`.

---

## 6. Raw state door

`jinfer_state_new(model, capacity)` builds a `State` over `model.contextModel()` (language, embedding or reranking;
speech/transcription are `EUNSUPPORTED`). Capacity follows `PromptCache.Options.contextCapacity` (0 = model's,
-1 = maximum, larger refused).

```mermaid
flowchart LR
    I["State.ingest(tokens)"] --> C{"position + n<br/>> capacity?"}
    C -->|yes| E["EINVAL"]
    C -->|no| K{"model.kind"}
    K -->|EMBEDDING| P["one Batch.pack, at most batchCapacity tokens"]
    K -->|other| F["Batch.prefill chunks of batchCapacity"]
    P & F --> CM["ContextModel.ingest(state, batch)"]
    L["logits()"] --> LM["LanguageModel.logits(state)"]
    EM["embedding()"] --> PE["EmbeddingModel.projectEmbedding"]
    LM & PE --> V["Views: castToSegmentBacked, requireF32, requireContiguous, copyToArray"]
```

`jinfer_model_tokenizer` hands out a toknroll handle via `com.qxotic.toknroll.libtoknroll.Native.handle(tokenizer)`,
freed with `toknroll_tokenizer_free`; it is available for language and embedding models.

---

## 7. Embed and rerank

```mermaid
sequenceDiagram
    participant N as Native
    participant M as Model
    participant CS as shared ContextState
    participant F as LoadedEmbedder / LoadedReranker

    N->>M: embedSize(count, dim) or rerankSize(count)
    alt size > floatCapacity
        M-->>N: size, nothing run
    else
        N->>M: embed(texts, dim) / rerank(q, docs, instruction)
        Note over M: synchronized on the model
        M->>M: sharedState(longest tokens)
        M->>CS: grow if batchCapacity too small, reset()
        M->>F: embedAll(state, texts, dim, sink) / scoreAll(state, task, q, docs, sink)
        F-->>M: vectors / scores
        M-->>N: float[]
    end
```

The shared state grows to the longest text seen and never shrinks: in a native image a dropped state is freed only
by a GC (~100 MB each), so one reused state is kept per model. `dimension(0)` means the native dimension;
otherwise it must lie in `minimumDimension()..dimension()`. A null rerank instruction uses
`reranker().defaultInstruction()`.

---

## 8. Lifetime, reference counting and memory

```mermaid
flowchart TD
    L["jinfer_model_load: references = 1"] --> R1["State / Stream ctor: retain()"]
    R1 --> R2["State.close / Stream.finish: release()"]
    F["jinfer_model_free: HANDLES.destroy + release()"] --> Q{"last reference?"}
    R2 --> Q
    Q -->|no| W["weights stay alive for remaining children"]
    Q -->|yes| T["driver.shutdownNow, info.close, shared.close<br/>engine.savePrompts + close, Arenas.close"]
    T --> G["Native.collect(): System.gc() only if ImageInfo.inImageRuntimeCode()"]
```

In the native image `Arenas.newCrossThread()` is `Arena.ofAuto()` (GraalVM cannot combine shared arenas with the
Vector API), so off-heap memory is released only by a collection; `Native.collect()` forces one after a last
release, a state free, or a stream freed early/cancelled. On the JVM (tests) it is a no-op. A stream releases its
model at the step that ends it, so a finished-but-unfreed stream keeps no weights alive.

---

## 9. Error mapping

`Errors.set` unwraps `UncheckedIOException` / `ExceptionInInitializerError`, maps the throwable via `Errors.code`,
formats a one-line message bounded to 1024 UTF-8 bytes (renaming `contextCapacity` to `context_capacity`), logs the
full throwable at DEBUG, and writes the message into a C TLS buffer from `jinfer__error_buffer`. An
`OutOfMemoryError` copies a pre-built message without allocating.

| Throwable | Code |
|---|---|
| `Errors.Failure` | its own `code` |
| `ModelProvider.IncompatibleModelException`, `UnsupportedOperationException` | `EUNSUPPORTED` (-4) |
| `ModelNotFoundException` | `ENOTFOUND` (-3) |
| `IllegalArgumentException`, `NullPointerException`, `Json.ParseException`, `EOFException` | `EINVAL` (-2) |
| `NoSuchFileException`, `FileNotFoundException` | `ENOTFOUND` (-3) |
| `CancellationException` | `ECANCELLED` (-5) |
| anything else | `ERROR` (-1) |

`EBUSY` (-6) comes from `Exclusive.enter()`; `DONE` (1) from `Stream.step()`. String-returning entry points return
`Errors.empty()` (`""`, never NULL) on failure.
