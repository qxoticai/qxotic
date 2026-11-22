# jinfer-core — class architecture

Three packages, 44 top-level types (+ `package-info`), ~4200 lines: `com.qxotic.jinfer` (27 types),
`com.qxotic.jinfer.media` (4) and `com.qxotic.jinfer.telemetry` (13). It is the **foundation of the jinfer stack**:
the model contracts every `jinfer-<model>` port implements (`Model` → `ContextModel` → `LanguageModel` /
`EmbeddingModel`, plus `Reranker`, `SpeechSynthesisModel`, `TranscriptionModel`), the state/lifetime rules
(`RuntimeState`, `ContextState`, `CheckpointCodec`), the unit of work (`Batch`), the memory and threading runtime
(`Arenas`, `Workspace`, `Views`, `Segments`, `Parallel`, `RuntimeFlags`) and JFR telemetry. It has no model code, no
tokenizer and no `META-INF/services` SPI; its only compile dependencies are `jota-core`, `jota-memory` and the
GraalVM `nativeimage` SDK (for the `VectorWidth` build-time `Feature`).

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends;
`*--` owns (nested/composed); classes marked `<<external>>` live outside this module.

1. [Position in the stack](#1-position-in-the-jinfer-stack)
2. [Model contracts](#2-model-contracts) — 2a hierarchy, 2b speech & transcription
3. [State lifetime and exclusive access](#3-state-lifetime-and-exclusive-access) — 3a classes, 3b `exclusively`/`close`
4. [Batch — the unit of work](#4-batch--the-unit-of-work)
5. [Embedding and checkpoint flows](#5-embedding-and-checkpoint-flows)
6. [Media and multimodal input](#6-media-and-multimodal-input)
7. [Memory runtime: Arenas, Workspace, Views](#7-memory-runtime)
8. [Compute runtime: Segments, Parallel, RuntimeFlags, VectorWidth](#8-compute-runtime)
9. [Telemetry (JFR)](#9-telemetry-jfr)
10. [Native-image configuration](#10-native-image-configuration)

| Class | Role |
|---|---|
| `Model<C,W,S>` | Root contract: immutable `configuration()` + `weights()`, typed by its `RuntimeState` |
| `ContextModel<C,W,S>` | Bounded incremental context: `newState(...)` (owned or borrowed arena), `ingest(S, Batch)`, optional `checkpointCodec()` |
| `ContextConfiguration` | `vocabularySize()`, `maxContextLength()` |
| `LanguageModel<C,W,S>` | `ContextModel` + vocabulary-logits head `logits(S, output)` |
| `EmbeddingModel<C,W,S>` | `ContextModel` + pooled head `projectEmbedding`; defaults `embedAll` (packed sequences), `embed` (multimodal parts) |
| `Reranker<S>` | Task recipe over a `ContextModel`: `scoreAll(state, instruction, query, documents, sink)` |
| `SpeechSynthesisModel<C,W,S>` | Phonemes → `Media.Audio` (`synthesize`), text → clips (`speak`) |
| `TranscriptionModel<C,W,S>` | PCM → `Transcription`; optional `stream(S)` |
| `TranscriptionStream` | Live audio: `feed` / `partial` / `finish` |
| `Transcription` (+`Word`, `Token`) | Timed transcript record; `words()` groups tokens |
| `Phonemizer` / `RunPhonemizer` | Text → phoneme ids; `Phonemizer.ipa(symbols, g2p)` builds the package-private `RunPhonemizer` |
| `SpeechOptions` | Narrow per-utterance knobs (`speed()`), `NONE`, `speed(double)` |
| `AudioFormat` | Package-private `requireMono(audio, rate)` shared by transcription APIs |
| `RuntimeState` | `AutoCloseable` base: reentrant exclusive access, fail-fast concurrency, exactly-once `releaseResources` |
| `ContextState` | `RuntimeState` + capacity, position, outputCount, tokensRead; arena ownership via `Cleaner` |
| `CheckpointCodec<S>` | Template: validated `capture` / `restore` of `[from,to)` to opaque memory; ports supply `sizeOf` / `transfer` |
| `Batch` (+`Input`, `Outputs`, `Positions`) | One forward's input (`Tokens` / `Embeddings` / `Sequences`) and retained rows (`LAST`/`ALL`); `prepare` normalises lists |
| `ContentKey` | Printable content identity (`sha256:<hex>`), `digestBytes()` |
| `Arenas` | Cross-thread arena policy: `requireCrossThread`, `newCrossThread`, best-effort `close` |
| `LeakWatch` | Opt-in (`-Djinfer.leakDetection`) CloseGuard-style unclosed-resource reporter |
| `Workspace` | Grow-then-reuse scratch `MemoryAllocator` with stack `Scope`s; zero steady-state allocation |
| `Views` | `MemoryView` factories and kernel entry checks (dtype, contiguity, liveness), `sliceLeadingAxis` |
| `Segments` | Vector API species, `FAST_VECTOR_JIT`, absolute-address `GLOBAL_SEGMENT`, scalar read/write helpers |
| `Parallel` | Process-wide fork/join pool (`shared()`, `forLoop`, `run`) with spinning daemon workers |
| `RuntimeFlags` | `BATCH_CAPACITY`, `THREADS` (physical/fast cores), `DECODE_BLOCK_SIZE`, `cacheSessions()`, `cacheBudget()` |
| `VectorWidth` | Package-private GraalVM `Feature`: pins `jinfer.vector.width` to the image's target ISA |
| `media.Media` (+`Image`, `Audio`, `Video`) | Sealed decoded non-text input |
| `media.MediaProjector<R>` | Media → model context rows in chunks; optional `decoderPositions`, `planId` |
| `media.Multimodal` | Capability: `projector(Class)`, `supports(Class)` |
| `media.FrameRef` | Frame cache identity: video `ContentKey` + timestamp |
| `telemetry.Telemetry` (+`CacheGauge`) | Idempotent JFR periodic-event install; weakly-held cache gauges |
| `telemetry.PerformanceCliff` | Enum of once-per-run degraded-path warnings (log + `CliffEvent`) |
| `telemetry.*Event` | JFR events: `Inference`, `ModelLoad`, `Runtime`, `PromptCache`, `MediaCache`, `MediaProjection`, `Speculation`, `Decode`, `Cliff` |
| `telemetry.CacheSample` / `MediaCacheSample` | Telemetry-owned records the cache gauge reads |

---

## 1. Position in the jinfer stack

```mermaid
flowchart TB
    Callers["cli / server / langchain4j / spring-ai / bench"]
    subgraph chat["jinfer-chat / jinfer-cache"]
        CE["ChatEngine · Generator · Models · ModelProvider"]
        PC["PromptCache · BlockTree · CachedSession"]
    end
    subgraph ports["jinfer-&lt;model&gt; ports"]
        P["Qwen3 · Gemma4 · Lfm2 · GptOss · Laguna · BailingMoe3 · Parakeet · KokoroTTS · ..."]
    end
    subgraph kern["jinfer-kernels"]
        K["MatMul · FlashAttention · Norms · ..."]
    end
    subgraph core["jinfer-core"]
        C1["Model contracts<br/>ContextModel · LanguageModel · EmbeddingModel<br/>Reranker · Speech · Transcription"]
        C2["State<br/>RuntimeState · ContextState · CheckpointCodec"]
        C3["Batch · ContentKey · media.*"]
        C4["Runtime<br/>Arenas · Workspace · Views · Segments · Parallel · RuntimeFlags"]
        C5["telemetry (JFR)"]
    end
    Jota["jota-core / jota-memory<br/>MemoryView · MemoryArena · MemoryAllocator · DataType"]

    Callers --> chat --> core
    chat --> ports
    ports -.->|implement| C1
    ports -.->|extend| C2
    ports --> kern --> C4
    PC -->|capture/restore| C2
    core --> Jota
```

The upstream implementers above were confirmed by grep (e.g. `Qwen3`, `Gemma4CheckpointCodec`, `Lfm2`, `Parakeet`,
`KokoroTTS`; `ChatEngine` builds a `Telemetry.CacheGauge`; `Generator` emits `DecodeEvent`;
`BailingMoe3Speculative` emits `SpeculationEvent`).

---

## 2. Model contracts

### 2a. Hierarchy

```mermaid
classDiagram
    direction TB
    class Model~C,W,S~ {
        <<interface>>
        configuration() C
        weights() W
    }
    class ContextConfiguration {
        <<interface>>
        vocabularySize() int
        maxContextLength() int
    }
    class ContextModel~C,W,S~ {
        <<interface>>
        newState(contextCapacity, batchCapacity) S
        newState(contextCapacity, batchCapacity, arena) S
        ingest(S, Batch)
        checkpointCodec() Optional
    }
    class LanguageModel~C,W,S~ {
        <<interface>>
        logits(S, output) MemoryView
        logits(S) MemoryView
    }
    class EmbeddingModel~C,W,S~ {
        <<interface>>
        projectEmbedding(S, outputIndex, consumer)
        projectEmbedding(S, consumer)
        embedAll(S, Sequences, sink)
        embed(S, parts, consumer) int
        requireCovers(Sequences, end)$
    }
    class Reranker~S~ {
        <<interface>>
        model() ContextModel
        defaultInstruction() String
        hasInstructionSlot() boolean
        scoreAll(S, instruction, query, documents, sink) int
    }
    class CheckpointCodec~S~ { <<abstract>> }
    class ContextState { <<abstract>> }
    class RuntimeState { <<abstract>> }
    class Batch { <<record>> }
    class MemoryView { <<external>> }
    class MemoryArena { <<external>> }

    Model <|-- ContextModel
    ContextModel <|-- LanguageModel
    ContextModel <|-- EmbeddingModel
    ContextModel ..> ContextConfiguration : C bound
    Model ..> RuntimeState : S bound
    ContextModel ..> ContextState : S bound
    ContextModel ..> Batch : ingest
    ContextModel ..> CheckpointCodec : optional
    ContextModel ..> MemoryArena : borrowed state memory
    LanguageModel ..> MemoryView : borrowed logits
    EmbeddingModel ..> MemoryView : borrowed embedding
    Reranker --> ContextModel : model()
```

`ContextModel.newState(capacity)` defaults the batch capacity to `RuntimeFlags.BATCH_CAPACITY`. The interface
Javadoc pins the implementation contract: validate/compute/mutate under `RuntimeState.exclusively`, advance the
context only after a successful forward, and `Reference.reachabilityFence` the model. Returned views are
**borrowed** from the state.

### 2b. Speech synthesis and transcription

```mermaid
classDiagram
    direction LR
    class Model~C,W,S~ { <<interface>> }
    class SpeechSynthesisModel~C,W,S~ {
        <<interface>>
        sampleRate() int
        phonemizer() Phonemizer
        newState() S
        newState(arena) S
        synthesize(S, phonemes, SpeechOptions) Audio
        speak(S, text, SpeechOptions, sink)
        speak(text, options) Audio
    }
    class TranscriptionModel~C,W,S~ {
        <<interface>>
        sampleRate() int
        newState() S
        newState(arena) S
        transcribe(S, pcm) Transcription
        transcribe(S, Audio) Transcription
        stream(S) TranscriptionStream
    }
    class TranscriptionStream {
        <<interface>>
        feed(pcm, offset, length) Transcription
        partial() Transcription
        finish() Transcription
        close()
    }
    class Transcription {
        <<record>>
        text String
        tokens List
        words() List
        empty()$
    }
    class Word { <<record>> }
    class Token { <<record>> }
    class Phonemizer {
        <<functional interface>>
        phonemize(text) int[]
        ipa(symbols, ipa)$ Phonemizer
    }
    class RunPhonemizer { <<package-private final>> }
    class SpeechOptions {
        <<interface>>
        speed() Double
        NONE$
        speed(double)$ SpeechOptions
    }
    class AudioFormat {
        <<package-private final>>
        requireMono(Audio, sampleRate)$ float[]
    }
    class Audio { <<Media.Audio>> }

    Model <|-- SpeechSynthesisModel
    Model <|-- TranscriptionModel
    SpeechSynthesisModel --> Phonemizer
    SpeechSynthesisModel ..> SpeechOptions
    SpeechSynthesisModel ..> Audio : produces / concat
    Phonemizer <|.. RunPhonemizer
    Phonemizer ..> RunPhonemizer : ipa() creates
    TranscriptionModel ..> TranscriptionStream : stream()
    TranscriptionModel ..> Transcription
    TranscriptionModel ..> AudioFormat
    TranscriptionStream ..> AudioFormat
    Transcription *-- Word
    Transcription *-- Token
```

These two are bound to plain `RuntimeState` (no context/positions). `stream` throws
`UnsupportedOperationException` by default; the `Audio` overloads route through `AudioFormat.requireMono`, which
refuses (never resamples) non-mono or wrong-rate audio. `RunPhonemizer` splits text into punctuation-free runs, calls
the G2P function once per run, re-emits marks, and maps code points via a direct-indexed table (unknown IPA dropped).

---

## 3. State lifetime and exclusive access

### 3a. Classes

```mermaid
classDiagram
    direction TB
    class AutoCloseable { <<interface, external>> }
    class RuntimeState {
        <<abstract>>
        -ReentrantLock lock
        -AtomicBoolean closing
        +exclusively(Runnable)
        +exclusively(Supplier) T
        +isAlive() boolean
        +close()
        #checkResourcesAlive()
        #releaseResources()*
    }
    class ContextState {
        <<abstract>>
        -MemoryArena memoryArena
        -Cleanable cleanup
        -int contextCapacity, batchCapacity
        -int position, outputCount, lastBatchSize
        -long tokensRead
        +position() int
        +tokensRead() long
        +resumeAt(position)
        +reset()
        #advanceContext(batchSize, Outputs)
        #clearHistory()*
        #memoryArena() MemoryArena
    }
    class CheckpointCodec~S~ {
        <<abstract>>
        +byteSize(positions) long
        +capture(S, from, to, MemorySegment)
        +restore(S, from, to, MemorySegment)
        #sizeOf(positions)* long
        #transfer(S, from, to, memory, capture)*
    }
    class Arenas { <<final>> }
    class LeakWatch { <<final>> }
    class Cleaner { <<external>> }
    class PortState["Qwen3.State, Gemma4 State, ..."] { <<external>> }
    class PortCodec["Gemma4CheckpointCodec, Lfm2CheckpointCodec, ..."] { <<external>> }

    AutoCloseable <|.. RuntimeState
    RuntimeState <|-- ContextState
    ContextState <|-- PortState
    CheckpointCodec <|-- PortCodec
    CheckpointCodec ..> ContextState : exclusively + span checks
    ContextState ..> Arenas : requireCrossThread / close
    ContextState ..> LeakWatch : site / report
    ContextState --> Cleaner : owned arena
```

Ownership rule (from `package-info`): **whoever supplies an arena owns it**. A `ContextState` built with
`ownsArena=true` registers a `Cleaner` action that closes the arena (via `Arenas.close`) and, if `LeakWatch` is
armed and `close()` never ran, reports the creation site. A borrowed arena is only liveness-checked
(`checkResourcesAlive` throws `FREED_MESSAGE` when the caller closed it too early).

### 3b. `exclusively` and `close`

```mermaid
sequenceDiagram
    participant T1 as Thread A
    participant T2 as Thread B
    participant S as RuntimeState
    T1->>S: exclusively(op)
    S->>S: enter(): tryLock, closing? checkResourcesAlive()
    T2->>S: exclusively(op2)
    S-->>T2: ConcurrentModificationException (single serial pipeline)
    T1->>S: nested exclusively(...) same thread: allowed
    T2->>S: close()
    S->>S: closing=true, lock.lock() waits for A
    S-->>T1: op returns, unlock
    S->>S: releaseResources() exactly once
    S-->>T2: close returns (or rethrows the shared failure)
    T1->>S: exclusively(op3)
    S-->>T1: IllegalStateException "state is closed"
```

`close()` from within the state's own operation is refused. `ContextState.advanceContext` validates
`batchSize ≤ batchCapacity` and `position + batchSize ≤ contextCapacity`, then sets `outputCount` to the batch size
(`Outputs.ALL`) or 1 (`LAST`). `CheckpointCodec.capture/restore` take exclusive access, check the span against
`contextCapacity`, require `memory.byteSize() == byteSize(to-from)` and, for capture, `position == to`, then call
the port's `transfer`.

---

## 4. Batch — the unit of work

```mermaid
classDiagram
    direction LR
    class Batch {
        <<record>>
        input Input
        outputs Outputs
        count() int
        prefill(ids)$ Batch
        step(id)$ Batch
        score(ids)$ Batch
        embeddings(rows, count, ...)$ Batch
        pack(seqs)$ Batch
        prepare(batches, batchCapacity)$ List
        tokenIds(batches)$ int[]
        positions(batches)$ int
    }
    class Outputs { <<enum>> LAST ALL }
    class Input { <<sealed interface>> }
    class Tokens { <<record>> ids int[] }
    class Embeddings {
        <<record>>
        rows MemoryView
        count int
        bidirectional boolean
        contentKey ContentKey
        positions Positions
    }
    class Sequences { <<record>> tokens Tokens; seqLen int[] }
    class Positions {
        <<final>>
        dimensions() int
        count() int
        value(row, dim) int
        advance() int
        slice(from, count, last) Positions
    }
    class ContentKey {
        <<record>>
        value String
        sha256(bytes)$ ContentKey
        digestBytes() byte[]
    }
    class MemoryView { <<external>> }

    Batch *-- Input
    Batch *-- Outputs
    Batch *-- Positions
    Input <|.. Tokens
    Input <|.. Embeddings
    Input <|.. Sequences
    Sequences --> Tokens
    Embeddings --> MemoryView : FP32 [count, dim] row-major
    Embeddings --> ContentKey
    Embeddings --> Positions
```

`Batch.prepare` normalises a list for a given batch capacity:

```mermaid
flowchart LR
    In["List of Batch"] --> Loop{"Tokens + LAST?"}
    Loop -->|yes| Run["accumulate run"]
    Loop -->|no| Flush["flushRun()"]
    Flush --> Bidi{"bidirectional Embeddings<br/>count &gt; capacity?"}
    Bidi -->|yes| Err["IllegalArgumentException"]
    Bidi -->|no| Pass["pass through unchanged"]
    Run --> FR["flushRun at end / next non-token"]
    FR --> One{"run size 1 and<br/>fits capacity?"}
    One -->|yes| Alias["reuse same Batch, no copy"]
    One -->|no| Slice["concat if needed, slice into<br/>prefill chunks of batchCapacity"]
```

---

## 5. Embedding and checkpoint flows

`EmbeddingModel.embedAll` (default method) packs ragged sequences through the port's `ingest`:

```mermaid
sequenceDiagram
    participant C as Caller (LoadedEmbedder)
    participant E as EmbeddingModel (default)
    participant S as ContextState
    participant P as Port ingest / projectEmbedding
    C->>E: embedAll(state, Sequences, sink)
    E->>E: requireComplete(sum seqLen == ids.length)
    E->>S: exclusively(embedAll0)
    E->>S: reset()
    loop chunks of batchCapacity
        E->>P: ingest(state, Batch(Sequences(chunk ids, full seqLen), ALL))
        P->>S: advanceContext(n, ALL)
        loop every sequence whose last token is in this chunk
            E->>P: projectEmbedding(state, lastRow - from, sink)
            P-->>C: sink.accept(borrowed MemoryView)
        end
    end
```

Each chunk carries the full `seqLen` layout, so ports validate with `EmbeddingModel.requireCovers` rather than
equality. Prompt caching (in `jinfer-cache`) uses `ContextModel.checkpointCodec()` →
`CheckpointCodec.byteSize/capture/restore` together with `ContextState.resumeAt(position)`; a port without a codec
triggers `PerformanceCliff.CACHE_SESSIONS_ONLY`.

---

## 6. Media and multimodal input

```mermaid
classDiagram
    direction LR
    class Media { <<sealed interface>> }
    class Image {
        <<record>>
        values float[] HWC in 0..1
        height, width, channels
    }
    class Audio {
        <<record>>
        pcm float[] in -1..1
        sampleRate, channels
        concat(clips)$ Audio
    }
    class Video { <<record>> frames List }
    class Frame { <<record>> image Image; timestamp Duration }
    class MediaProjector~R~ {
        <<interface>>
        positions(R) int
        project(R, maxChunkSize, sink)
        decoderPositions(R) Positions
        planId() String
    }
    class Multimodal {
        <<interface>>
        projector(Class) Optional
        supports(Class) boolean
    }
    class FrameRef { <<record>> video ContentKey; timestamp Duration }
    class ContentKey { <<record>> }
    class Positions { <<Batch.Positions>> }
    class EmbeddingModel { <<interface>> }

    Media <|.. Image
    Media <|.. Audio
    Media <|.. Video
    Video *-- Frame
    Frame --> Image
    Multimodal ..> MediaProjector : per modality
    MediaProjector ..> Media : R extends Media
    MediaProjector ..> Positions
    FrameRef --> ContentKey
    EmbeddingModel ..> Media : embed(parts)
```

Records validate eagerly (dimensions, channel count 1/3/4, value ranges, ascending frame timestamps). A projector's
chunks are borrowed views, typically wrapped by callers as `Batch.embeddings(rows, count, bidirectional, contentKey,
positions)`; `Positions.slice` rebases coordinates per chunk.

---

## 7. Memory runtime

```mermaid
classDiagram
    direction LR
    class Arenas {
        <<final>>
        requireCrossThread(Arena)$
        requireCrossThread(MemoryArena)$
        sharedArenas()$ boolean
        newCrossThread()$ Arena
        newCrossThreadMemoryArena()$ MemoryArena
        close(Arena)$
        close(MemoryArena)$
    }
    class Workspace {
        <<final>>
        rewind()
        floats(size) float[]
        floatsAtLeast(size) float[]
        ints(size) int[]
        matrix(rows, cols)
        allocateMemory(bytes, align) Memory
        takeFloats(allocator, size)$
        scope(allocator)$ Scope
        backingAllocations() int
        heapAllocations() int
    }
    class Scope { <<interface>> close() }
    class StackScope { <<private>> }
    class Views {
        <<final>>
        allocateF32 / allocateF16(allocator, dims)$
        fromFloatArray(allocator, values)$
        requireDatatype / requireContiguous / requireDense$
        checkAlive(view, name)$
        castToSegmentBacked(view, name)$
        sliceLeadingAxis(stacked)$
    }
    class LeakWatch {
        <<final>>
        arm(owner, what)$ Runnable
    }
    class MemoryAllocator~MemorySegment~ { <<interface, external>> }
    class MemoryArena~MemorySegment~ { <<interface, external>> }
    class MemoryAllocators { <<external>> }
    class MemoryView { <<external>> }

    MemoryAllocator <|.. Workspace
    Workspace --> MemoryAllocator : backing arena
    Workspace *-- StackScope
    Scope <|.. StackScope
    Arenas ..> MemoryAllocators : ofArena / adoptArena
    Arenas ..> MemoryArena
    Views ..> MemoryView : creates / checks
    Views ..> MemoryAllocator
```

`Arenas` probes a zero-byte allocation against an unstarted local thread to refuse confined arenas (kernels may run
on pool or native threads and bypass JDK confinement). `newCrossThread()` is `Arena.ofShared()` on the JVM and
`Arena.ofAuto()` in a native image (unless `-Djinfer.sharedArenas`), decided at call time. `Workspace` hands buffers
out by allocation order and reuses them on `rewind()` (contents not zeroed); the static `take*`/`scope` helpers fall
back to fresh arrays when the allocator is not a `Workspace`.

---

## 8. Compute runtime

```mermaid
classDiagram
    direction LR
    class RuntimeFlags {
        <<final>>
        BATCH_CAPACITY$ int
        THREADS$ int
        DECODE_BLOCK_SIZE$ int
        cacheSessions()$ int
        cacheBudget()$ long
    }
    class Parallel {
        <<final>>
        shared()$ Parallel
        of(width)$ Parallel
        threads()$ int
        forLoop(...)$
        inRegion()$ boolean
        run(jobs, Job)
        loop(start, end, body)
        close()
    }
    class Job { <<functional interface>> run(index, slot) }
    class Region { <<private>> next, finished, failure }
    class Worker { <<private Thread>> slot, parked }
    class Shared { <<private holder>> POOL }
    class Segments {
        <<final>>
        F_SPECIES, I_SPECIES, S_SPECIES_HALF, B_SPECIES$
        USE_VECTOR_API$, FAST_VECTOR_JIT$
        vectorBits()$ int
        vectorSegment(seg)$
        vectorBase(seg)$ long
        readFloat / writeFloat / readFloat16 ...$
    }
    class VectorWidth { <<package-private Feature>> beforeAnalysis() }
    class Feature { <<interface, external>> }
    class PerformanceCliff { <<enum>> }

    Parallel *-- Region
    Parallel *-- Worker
    Parallel *-- Shared
    Parallel ..> Job
    Shared ..> RuntimeFlags : THREADS
    Parallel ..> PerformanceCliff : NESTED_REGION, THREAD_OVERSUBSCRIPTION
    Segments ..> PerformanceCliff : SLOW_JIT, NATIVE_ACCESS_RESTRICTED
    Feature <|.. VectorWidth
    VectorWidth ..> Segments : sets jinfer.vector.width before init
```

```mermaid
sequenceDiagram
    participant K as Kernel (caller, slot 0)
    participant P as Parallel.shared()
    participant W as Worker 1..width-1
    K->>P: forLoop(start, end, body)
    alt one index, width 1 or nested in same pool
        P->>K: run inline (nested: NESTED_REGION cliff)
    else
        P->>P: pool() lazily starts daemon workers
        P->>P: lock, publish new Region (2 x width bands)
        P->>W: unpark parked workers
        par
            K->>K: region.work(0)
        and
            W->>W: region.work(slot), claim bands via NEXT
        end
        K->>K: spin until finished == jobs
        P-->>K: rethrow first failure, if any
    end
    W->>W: spin SPIN_NANOS (jinfer.spinNanos, 1 ms) then park
```

`RuntimeFlags.THREADS` defaults to the physical/fast-core count (Apple `sysctlbyname("hw.perflevel0.physicalcpu")`,
Linux sysfs topology incl. hybrid / big.LITTLE `cpu_capacity`, capped by `availableProcessors`). `Segments` fails
fast with a fix-it message when `jdk.incubator.vector` is missing; `-Djinfer.vector.width` accepts only
0/128/256/512 and `-Djinfer.vector.jit=auto|fast|slow` controls `FAST_VECTOR_JIT`. The absolute-address
`GLOBAL_SEGMENT` (`MemorySegment.NULL.reinterpret(Long.MAX_VALUE)`) needs native access.

---

## 9. Telemetry (JFR)

```mermaid
classDiagram
    direction TB
    class Event { <<abstract, external jdk.jfr>> }
    class Telemetry {
        <<final>>
        install()$
        register(CacheGauge)$
        unregister(CacheGauge)$
        queueWait(nanos)$
        takeQueueWait()$ long
    }
    class CacheGauge {
        <<static final>>
        CacheGauge(model, Supplier CacheSample, Supplier MediaCacheSample)
        emitPrompt()
        emitMedia()
    }
    class CacheSample { <<record>> }
    class MediaCacheSample { <<record>> }
    class PerformanceCliff {
        <<enum>>
        SLOW_JIT
        NATIVE_ACCESS_RESTRICTED
        THREAD_OVERSUBSCRIPTION
        NESTED_REGION
        JAM_ABSENT
        JAM_DECLINE
        MAMBA2_SCALAR
        GDN_SCALAR
        CACHE_SESSIONS_ONLY
        report()
    }
    class InferenceEvent { started(model, op, outputType)$ }
    class MediaProjectionEvent { started(Media)$ }
    class CliffEvent { emit(cliff, detail)$ }
    class RuntimeEvent
    class PromptCacheEvent
    class MediaCacheEvent
    class ModelLoadEvent
    class SpeculationEvent
    class DecodeEvent
    class ChatEngine { <<external jinfer-chat>> }

    Event <|-- InferenceEvent
    Event <|-- MediaProjectionEvent
    Event <|-- CliffEvent
    Event <|-- RuntimeEvent
    Event <|-- PromptCacheEvent
    Event <|-- MediaCacheEvent
    Event <|-- ModelLoadEvent
    Event <|-- SpeculationEvent
    Event <|-- DecodeEvent
    Telemetry *-- CacheGauge
    Telemetry o-- CacheGauge : WeakReference list
    CacheGauge --> CacheSample
    CacheGauge --> MediaCacheSample
    CacheGauge ..> PromptCacheEvent : emitPrompt
    CacheGauge ..> MediaCacheEvent : emitMedia
    Telemetry ..> RuntimeEvent : periodic
    PerformanceCliff ..> CliffEvent : once per run
    ChatEngine ..> CacheGauge : new + register
```

```mermaid
flowchart LR
    Install["Telemetry.install()<br/>idempotent, locked once"] --> FR["FlightRecorder.addPeriodicEvent"]
    FR --> RE["RuntimeEvent: Segments.vectorBits, RuntimeFlags.THREADS<br/>everyChunk"]
    FR --> PCE["PromptCacheEvent: each live CacheGauge.emitPrompt<br/>1 s, deltas vs last sample"]
    FR --> MCE["MediaCacheEvent: each live CacheGauge.emitMedia<br/>1 s"]
    Reg["register(gauge)"] --> Install
    Reg --> Prune["prune dead WeakReferences, add"]
```

JFR is the seam: no listener interface. Gauges are held weakly so an unclosed engine stays collectable and
`LeakWatch` can still report it. `src/main/resources/jinfer.jfc` is a JFR settings profile enabling these events
(`jinfer.Decode` off by default). `PerformanceCliff.report()` is a plain-read once-filter; the first report is
`@NeverInline` and emits one `CliffEvent` plus one `jinfer.perf` log line. Within this repository no emitter of
`ModelLoadEvent` was found outside its schema test.

---

## 10. Native-image configuration

`META-INF/native-image/com.qxotic/jinfer-core/native-image.properties` enables `VectorAPISupport` and
`ForeignAPISupport`, registers `--features=com.qxotic.jinfer.VectorWidth`, adds `jdk.incubator.vector`, enables
native access, initialises `com.qxotic.` at build time and opts back to run time: `RuntimeFlags`, `Parallel`,
`Parallel$Shared`, `LeakWatch`, `LeakWatch$Holder`, `ContextState` (flags and threads read on the host, cleaners
and pools not baked into the heap). `reachability-metadata.json` sits alongside. There is no
`META-INF/services` registration in this module; model discovery (`ModelProvider` via `ServiceLoader`) lives in
`jinfer-chat`.
