# jinfer-bench — class architecture

Package `com.qxotic.jinfer.bench` (7 top-level classes, ~1960 lines, plus 1 test class). It is the
**benchmark harness** of the jinfer stack: a set of independent `main` programs, none of which is a library API.
The headline program, `JinferBench`, is a llama-bench-parity harness (pp/tg throughput) that drives *any*
model on the classpath through the generic loader `Models.load`; the others benchmark embedding, speech
recognition (WER / RTF), unbounded streaming transcription, and two kernels (`conv1dRows`, `Parallel.forLoop`).
The module registers **no SPI** (`META-INF/services` is absent); it only *consumes* ServiceLoader-discovered
model ports (`jinfer-models-all`, runtime scope) and JAM backends (`jam-native`, `jam-vector`, `jam-scalar`, runtime scope).
The `JinferBench` main class is the executable (shaded jar / GraalVM native image `jinfer-bench`, via `jinfer.image.mainClass` in `pom.xml`).

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends;
`*--` composition (nested type / owned field); classes marked `<<external>>` live outside this module.

1. [Position in the stack](#1-position-in-the-jinfer-stack)
2. [Class diagrams](#2-class-diagrams) — 2a `JinferBench`, 2b `EmbedBench`, 2c speech benches, 2d kernel microbenches
3. [JinferBench: parity tests](#3-jinferbench-parity-tests) — 3a main loop, 3b loading, 3c `measure`/`runOnce`
4. [JinferBench: capabilities pass](#4-jinferbench-capabilities-pass)
5. [EmbedBench flow](#5-embedbench-flow)
6. [TranscriptionBench and StreamSoak flows](#6-transcriptionbench-and-streamsoak-flows)
7. [Kernel microbenches](#7-kernel-microbenches)
8. [Design notes](#8-design-notes)

| Class | Role |
|---|---|
| `JinferBench` | Main executable. llama-bench-parity `pp<N>` / `tg<N>` tests (optionally at depth `@d<N>`) per model, then a *capabilities* pass through `ChatEngine` (state alloc, TTFT, prompt-cache hit, MTP acceptance, media cold/warm, peak RSS). Nested: `BenchModel<S>`, records `Row`, `CapRow` |
| `JinferBench.BenchModel<S extends ContextState>` | The one model wrapper: holds a `LoadedModel<S>`, one reusable state, an optional depth snapshot (`CheckpointCodec` + `MemorySegment`); falls back to opening *decision checkpoints* (non-causal `lfm3` / `qwen35` / `qwen35moe`) directly via `Lfm3.loadModel` / `Qwen35.loadModel` |
| `EmbedBench` | Ragged/packed batched-embedding throughput through `EmbeddingModel.embedAll`, any embedder from `Models.loadEmbedder`. Nested: `EmbedModel<S>` |
| `TranscriptionBench` | ASR quality (WER over a LibriSpeech-layout corpus, optional `--gate` pass/fail and `--dump` TSV) and speed (RTF over one file) via `Models.loadTranscription`. Nested records `Utterance`, `Scored`; package-visible `normalize`, `editDistance` (unit-tested) |
| `StreamSoak` | Hours-long soak of one `TranscriptionStream` over a seeded synthetic program with injected pathologies; checks ordering/mute/recovery invariants, RSS slope, partial-latency percentiles. Nested: `Program`, record `Event` |
| `ConvPeak` | Microbench of `Convolutions.conv1dRows` per register-tile shape (`-Djinfer.convTile`), on a census parsed from a `-Djinfer.convProfile` dump or a fallback vocoder ladder; prints a FLOP-weighted predicted total |
| `ConvParity` | Parity check: SHA-256 digests of `conv1dRows` output for fixed shapes, to diff across tile shapes run in separate JVMs |
| `SpinProbe` | Microbench of the empty `Parallel.forLoop` region (dispatch + barrier latency) |
| `TranscriptionBenchTest` (test) | JUnit 5 tests of `TranscriptionBench.normalize` and `editDistance` |

---

## 1. Position in the jinfer stack

Dependencies are those declared in `jinfer-bench/pom.xml`. The bench sits at the very top: it depends on
the public loaders of `jinfer-chat` and the contracts of `jinfer-core`, and pulls every model port in at
runtime so that `Models.load` can resolve any GGUF architecture.

```mermaid
flowchart TB
    subgraph bench["jinfer-bench (this module)"]
        JB["JinferBench"]
        EB["EmbedBench"]
        TB["TranscriptionBench"]
        SS["StreamSoak"]
        CP["ConvPeak / ConvParity"]
        SP["SpinProbe"]
    end
    subgraph chat["jinfer-chat"]
        Models["Models<br/>load / loadEmbedder / loadTranscription"]
        CE["ChatEngine"]
        LM["LoadedModel / LoadedEmbedder"]
    end
    subgraph cache["jinfer-cache"]
        PC["PromptCache.Options"]
    end
    subgraph core["jinfer-core"]
        Contracts["LanguageModel / EmbeddingModel / TranscriptionModel<br/>ContextState / CheckpointCodec / Batch"]
        Par["Parallel / RuntimeFlags / Segments / Views"]
    end
    subgraph kernels["jinfer-kernels"]
        Conv["Convolutions.conv1dRows"]
    end
    subgraph codecs["jinfer-codecs"]
        Codec["ImageCodec / AudioCodec"]
    end
    subgraph ports["model ports (runtime, ServiceLoader)"]
        All["jinfer-models-all"]
        Direct["jinfer-lfm3 / jinfer-qwen35<br/>(compile: decision checkpoints)"]
    end
    subgraph jam["JAM backends (runtime, ServiceLoader)"]
        Jam["jam-native / jam-vector / jam-scalar"]
    end
    Mem["jota-memory<br/>MemoryAllocators / MemoryView"]
    Tok["toknroll-core<br/>Tokenizer"]
    Gguf["gguf<br/>GGUF"]

    JB --> Models
    JB --> CE
    JB --> PC
    JB --> Contracts
    JB --> Direct
    JB --> Gguf
    JB --> Codec
    JB -. "header: JAM.providers()" .-> Jam
    EB --> Models
    TB --> Models
    SS --> Models
    TB --> Codec
    SS --> Codec
    CP --> Conv
    CP --> Mem
    SP --> Par
    Models -. ServiceLoader .-> All
    CE --> Contracts
    Direct --> Tok
```

---

## 2. Class diagrams

### 2a. `JinferBench`

```mermaid
classDiagram
    direction LR
    class JinferBench {
        <<final>>
        -int UBATCH = 512$
        -volatile float sink$
        +main(String[])$
        -measure(BenchModel, name, threads, test, count, prefill, depth, ctx, minWarmup, reps, noWarmup)$ Row
        -runOnce(BenchModel, prefix, prompt, count, prefill, vocab)$ double
        -capabilities(BenchModel, name, promptLen, gen, mediaPath)$ CapRow
        -complete(ChatEngine, int[], Sampler, gen)$ Completion
        -complete(ChatEngine, Request)$ Completion
        -followUp(prompt, reply, vocab)$ int[]
        -textRequest(question, gen)$ Request
        -mediaRequest(Image, ContentKey, gen, question)$ Request
        -promptNanos(ChatEngine, Request)$ long
        -peakRssBytes()$ long
        -nextToken(previous, vocab)$ int
        -fillerTokens(vocab, count)$ int[]
        ~header(threadsRequested)$ String
        -printTable(List~Row~)$
        -printCaps(List~CapRow~)$
    }
    class BenchModel~S~ {
        <<static final>>
        +LoadedModel~S~ loaded
        +boolean decision
        -S state
        -CheckpointCodec~S~ depthCodec
        -MemorySegment depthSnapshot
        +open(Path, Map companions)$ BenchModel
        -openDecision(Path)$ BenchModel
        -network(LanguageModel, Tokenizer, Path)$ BenchModel
        +vocab() int
        +newState(ctx, batch) long
        +reset()
        +prepareDepth(int[] prefix)
        +resetToDepth(int[] prefix)
        +ingestPrefill(int[] prompt)
        +ingestStep(int tok)
        +logits0() float
    }
    class Row {
        <<record>>
        model, threads, test, mean, stddev
    }
    class CapRow {
        <<record>>
        model, stateAllocMs, ttftMs, cacheHit
        mtp, media, peakRssMb
    }
    class LoadedModel~S~ {
        <<external>>
        <<record>>
        model() LanguageModel
        tokenizer() Tokenizer
    }
    class Models {
        <<external>>
        +load(Path, Arena, Map)$
        +modelSeed(Path)$
    }
    class ChatEngine {
        <<external>>
        +complete(Prepared, ReplySink) Completion
        +prepare(Request) Prepared
        +speculationReady() boolean
        +speculationDepth(int)
    }
    class LanguageModel~C,W,S~ {
        <<external>>
        <<interface>>
        newState(ctx, batch) S
        ingest(S, Batch)
        logits(S) MemoryView
        checkpointCodec() Optional
    }
    class CheckpointCodec~S~ {
        <<external>>
        <<abstract>>
        byteSize(positions) long
        capture(S, from, to, dst)
        restore(S, from, to, src)
    }
    class Lfm3 {
        <<external>>
    }
    class Qwen35 {
        <<external>>
    }
    JinferBench *-- BenchModel
    JinferBench *-- Row
    JinferBench *-- CapRow
    JinferBench ..> ChatEngine : creates 2 per model
    BenchModel --> LoadedModel
    BenchModel --> CheckpointCodec : depth snapshot
    BenchModel ..> Models : load
    BenchModel ..> Lfm3 : loadModel (decision)
    BenchModel ..> Qwen35 : loadModel (decision)
    LoadedModel --> LanguageModel
    Lfm3 ..|> LanguageModel
    Qwen35 ..|> LanguageModel
    ChatEngine --> LoadedModel
```

### 2b. `EmbedBench`

```mermaid
classDiagram
    direction LR
    class EmbedBench {
        <<final>>
        -volatile double sink$
        +main(String[])$
        -runOnce(EmbedModel, ids, seqLen, total, nSeq)$ double
        -mean(double[])$ double
        -stddev(double[])$ double
    }
    class EmbedModel~S~ {
        <<static final>>
        -LoadedEmbedder~S~ loaded
        -S state
        +open(Path)$ EmbedModel
        +vocab() int
        +newState(ctx, batchCap)
        +embed(int[] ids, int[] seqLen) int
    }
    class LoadedEmbedder~S~ {
        <<external>>
        <<record>>
        model() EmbeddingModel
    }
    class EmbeddingModel~C,W,S~ {
        <<external>>
        <<interface>>
        embedAll(S, Batch.Input.Sequences, Consumer)
    }
    class Models {
        <<external>>
        +loadEmbedder(Path, Arena)$
    }
    EmbedBench *-- EmbedModel
    EmbedModel --> LoadedEmbedder
    EmbedModel ..> Models : loadEmbedder
    LoadedEmbedder --> EmbeddingModel
```

### 2c. Speech benches: `TranscriptionBench`, `StreamSoak`

```mermaid
classDiagram
    direction LR
    class TranscriptionBench {
        <<final>>
        +main(String[])$
        -wer(TranscriptionModel, corpus, limit, dump)$ double
        -rtf(TranscriptionModel, audio, reps)$
        -corpus(Path, limit)$ List
        ~normalize(String)$ String[]
        ~editDistance(String[], String[])$ int
    }
    class Utterance {
        <<record>>
        id, flac, reference
    }
    class Scored {
        <<record>>
        utterance, errors, words
        hypothesis, audioSeconds, decodeSeconds
    }
    class StreamSoak {
        <<final>>
        -int RATE = 16000$
        -int FEED_CHUNK = RATE/2$
        +main(String[])$
        -soak(TranscriptionModel, Program, StringBuilder)$
        ~program(clips, collapse, spanish, seed, seconds)$ Program
        ~loadClips(Path)$ List
        -streamWhole(TranscriptionModel, float[])$ String
        -rssMb()$ double
        -slopePerHour(List)$ double
    }
    class Program {
        <<static final>>
        blocks : List~float[]~
        events : List~Event~
        seconds : double
        add(float[] pcm, String eventType)
    }
    class Event {
        <<record>>
        type, startSec, endSec
    }
    class TranscriptionModel~C,W,S~ {
        <<external>>
        <<interface>>
        sampleRate() int
        newState() S
        transcribe(S, float[]) Transcription
        transcribe(float[]) Transcription
        stream(S) TranscriptionStream
    }
    class TranscriptionStream {
        <<external>>
        <<interface>>
        feed(pcm, offset, length) Transcription
        partial() Transcription
        finish() Transcription
    }
    class AudioCodec {
        <<external>>
        load(Path)$ Media.Audio
    }
    TranscriptionBench *-- Utterance
    TranscriptionBench *-- Scored
    Scored --> Utterance
    TranscriptionBench ..> TranscriptionModel : Models.loadTranscription
    TranscriptionBench ..> AudioCodec
    StreamSoak *-- Program
    StreamSoak *-- Event
    Program --> Event
    StreamSoak ..> TranscriptionModel : Models.loadTranscription
    StreamSoak ..> TranscriptionStream : one per soak
    StreamSoak ..> AudioCodec
    TranscriptionModel ..> TranscriptionStream : stream
```

### 2d. Kernel microbenches: `ConvPeak`, `ConvParity`, `SpinProbe`

```mermaid
classDiagram
    direction LR
    class ConvPeak {
        <<final>>
        -Shape[] LADDER$
        -WINDOW=3, TOL=0.03, REPS=5$
        -MIN_WARMUP_NANOS=1.5s, MAX_WARMUP_NANOS=6s$
        -SHARE_FLOOR=0.002$
        +main(String[])$
        -census(Path)$ Shape[]
        -measure(Shape)$ double[]
        -pass(Shape, in, out, taps)$ double
    }
    class ConvPeak_Shape {
        <<record>>
        channels, time, kernel, dilation, gflops
        gflopsPerCall() double
    }
    class ConvParity {
        <<final>>
        -Shape[] SHAPES$
        +main(String[])$
        -digest(Shape)$ String
    }
    class ConvParity_Shape {
        <<record>>
        inChannels, outChannels, time, kernel, dilation
    }
    class SpinProbe {
        <<final>>
        +main(String[])$
    }
    class Convolutions {
        <<external>>
        conv1dRows(in, inCh, out, outCh, time, kernel, dilation, taps, bias)$
    }
    class Parallel {
        <<external>>
        threads()$ int
        forLoop(start, end, body)$
    }
    class Views {
        <<external>>
        allocateF32()$
        copyFromArray()$
        toFloatArray()$
    }
    ConvPeak *-- ConvPeak_Shape
    ConvParity *-- ConvParity_Shape
    ConvPeak ..> Convolutions
    ConvParity ..> Convolutions
    ConvPeak ..> Views
    ConvParity ..> Views
    SpinProbe ..> Parallel
```

(`ConvPeak_Shape` / `ConvParity_Shape` stand for the private nested records both named `Shape`.)

---

## 3. JinferBench: parity tests

### 3a. Main loop

`main` parses llama-bench-style flags (`-m` repeatable, `-p`, `-n`, `-d`, `-r`, `-w`, `--no-warmup`,
`--no-capabilities`, `--ctx`, `-t`, `--media`, `--with cap=path`). `-t` is written to the system property
`jinfer.threads` **before** `RuntimeFlags` is first touched (by `header`). Per model: one `pp` row, one `tg`
row, one capabilities row; decision models get only `pp` at depth 0.

```mermaid
flowchart TB
    A["parse args"] --> T{"-t given?"}
    T -- yes --> P["System.setProperty jinfer.threads"]
    T -- no --> H
    P --> H["print header: runtime, JAM.providers, RuntimeFlags.THREADS"]
    H --> L["for each -m path"]
    L --> O["BenchModel.open path, companions"]
    O --> PP{"p > 0 and not decision-at-depth?"}
    PP -- yes --> M1["measure pp: ctx = --ctx or depth+p"]
    PP -- no --> TG
    M1 --> TG{"n > 0 and not decision?"}
    TG -- yes --> M2["measure tg: ctx = --ctx or depth+n"]
    TG -- no --> C
    M2 --> C{"capabilities enabled and not decision?"}
    C -- yes --> CAP["capabilities bench, name, p or 512, max n,16, media"]
    C -- no --> L
    CAP --> L
    L -- done --> OUT["printTable rows, printCaps caps"]
```

### 3b. Loading (`BenchModel.open`)

```mermaid
sequenceDiagram
    participant JB as JinferBench
    participant BM as BenchModel
    participant M as Models (jinfer-chat)
    participant G as GGUF
    participant D as Lfm3 / Qwen35

    JB->>BM: open(path, companions)
    BM->>M: load(path, Arena.ofAuto(), companions)
    alt provider accepts
        M-->>BM: LoadedModel
        BM-->>JB: BenchModel(loaded, decision=false)
    else IncompatibleModelException
        BM->>G: GGUF.read(path)
        BM->>BM: arch = general.architecture
        alt arch.attention.causal == true (default)
            BM--xJB: rethrow
        else non-causal lfm3 / qwen35 / qwen35moe
            BM->>D: loadModel(path, Arena.ofAuto())
            BM->>BM: network(model, model.tokenizer(), path)
            Note over BM: new LoadedModel(model, tokenizer, "", Set.of(),<br/>Models.modelSeed(path), Optional.empty(), SamplingDefaults.NONE)
            BM-->>JB: BenchModel(loaded, decision=true)
        end
    end
```

### 3c. `measure` / `runOnce`

One state per test, sized to the work (`newState(ctx, batch)` with `batch = UBATCH = 512`, or `count`
for a decision model which must read the prompt in one batch). The depth prefix is prepared once and
snapshotted through the model's `CheckpointCodec` if it has one; otherwise it is re-ingested before every
pass — always outside the timed region.

```mermaid
sequenceDiagram
    participant JB as JinferBench.measure
    participant BM as BenchModel
    participant LM as LanguageModel
    participant S as ContextState
    participant CC as CheckpointCodec

    JB->>BM: newState(ctx, batch)
    BM->>LM: newState(ctx, batch)
    JB->>BM: prepareDepth(prefix)
    opt prefix non-empty
        BM->>S: reset()
        BM->>LM: ingest(state, Batch) per chunk of Batch.prepare
        opt checkpointCodec() present
            BM->>CC: capture(state, d, d, heap snapshot)
        end
    end
    loop adaptive warmup until 3 passes agree within 3% (min -w, max 30), then reps
        JB->>BM: resetToDepth(prefix) [untimed]
        BM->>S: reset()
        alt snapshot present
            BM->>CC: restore(state, d, d, snapshot)
            BM->>S: resumeAt(d)
        else no codec
            BM->>LM: ingest prefix again
        end
        alt pp (prefill)
            Note over JB: t0
            BM->>LM: ingest chunks of filler prompt
            BM->>LM: logits(state) → read element 0 into sink
        else tg
            Note over JB: t0
            loop count times
                BM->>LM: ingest(state, Batch.step(tok))
                BM->>LM: logits(state) → sink
                JB->>JB: tok = LCG nextToken(tok, vocab)
            end
        end
        JB->>JB: tokens/second for this pass
    end
    JB-->>JB: Row(mean, stddev of reps)
```

Synthetic tokens: `fillerTokens` gives `(i*17+1) % vocab`; `nextToken` is an LCG standing in for
llama-bench's `rand() % n_vocab`. The `volatile sink` keeps the logit projection from being eliminated.

---

## 4. JinferBench: capabilities pass

`capabilities` times a fresh state allocation, then builds **two** `ChatEngine`s over the same
`LoadedModel`: one with `PromptCache.Options.DEFAULTS` and a no-cache control
(`withRetainedSessions(0).withBlockBudget(0)`).

```mermaid
sequenceDiagram
    participant JB as JinferBench.capabilities
    participant E as ChatEngine (cached)
    participant C as ChatEngine (control, no cache)
    participant MM as Multimodal model

    JB->>JB: allocMs = bench.newState(promptLen+gen+16, 512)
    JB->>E: complete(Prepared.raw(filler prompt, greedy, gen))
    E-->>JB: cold Completion → TTFT = result.promptTime
    JB->>JB: followUp = prompt + reply tokens + 8 new tokens
    JB->>E: complete(followUp)
    E-->>JB: hit (restoredTokens)
    opt restoredTokens > 0
        JB->>C: complete(followUp)
        C-->>JB: full re-prefill → "hit vs full (x, n restored)"
    end
    opt E.speculationReady()
        JB->>E: speculationDepth(0), then complete(textRequest)
        JB->>E: speculationDepth(4), then complete(textRequest)
        E-->>JB: speculated(): accepted/drafted, decode t/s ratio
    end
    opt --media and model instanceof Multimodal with an Image projector
        JB->>JB: ImageCodec.decode, ContentKey.sha256
        JB->>E: prepare+complete(mediaRequest "Describe this image.") → cold promptTime
        JB->>E: prepare+complete(mediaRequest "What is in this picture?") → warm promptTime
    end
    JB->>JB: peakRssBytes() from /proc/self/status VmHWM
    JB-->>JB: CapRow
```

The MTP check uses a real chat-templated prompt (`textRequest`) rather than filler tokens, because a
draft head only accepts predictable text. The media warm pass asks a *different* question so only the
media cache, not the prompt cache, can serve it.

---

## 5. EmbedBench flow

```mermaid
flowchart LR
    A["parse args<br/>-s -minlen -maxlen -b -r -w -t"] --> B["ragged lengths by<br/>multiplicative hash"]
    B --> C["ctx = total + 64"]
    C --> D["EmbedModel.open → Models.loadEmbedder"]
    D --> E["newState ctx, batchCap"]
    E --> F["adaptive warmup<br/>3 passes within 3%"]
    F --> G["reps × runOnce"]
    G --> H["embedAll state,<br/>Batch.Input.Sequences Tokens ids, seqLen<br/>consumer reads 1 float into sink"]
    H --> I["table: tok/s ± sd, seq/s"]
```

`runOnce` checks that exactly `nSeq` pooled vectors were streamed out, else throws.

---

## 6. TranscriptionBench and StreamSoak flows

### 6a. `TranscriptionBench`

Exactly one of `--librispeech` or `--audio` is required; `--gate` needs `--librispeech`. The model is loaded
into a shared `Arena` via `Models.loadTranscription`.

```mermaid
flowchart TB
    A["Models.loadTranscription model, Arena.ofShared"] --> B{"corpus or audio?"}
    B -- corpus --> W1["corpus: walk *.trans.txt, keep id with sibling .flac, up to --limit"]
    W1 --> W2["one state = transcriber.newState"]
    W2 --> W3["per utterance: AudioCodec.load → transcribe state, pcm timed"]
    W3 --> W4["normalize ref & hyp → word editDistance → errors/words"]
    W4 --> W5["print WER, speed, 5 worst; optional --dump TSV"]
    W5 --> G{"--gate?"}
    G -- "WER > gate" --> X["exit 1"]
    B -- audio --> R1["AudioCodec.load, one state"]
    R1 --> R2["1 warmup transcribe, then --reps timed runs → RTFx"]
```

### 6b. `StreamSoak`

```mermaid
sequenceDiagram
    participant SS as StreamSoak
    participant P as Program
    participant TM as TranscriptionModel
    participant TS as TranscriptionStream

    SS->>SS: loadClips(corpus *.flac), en.wav, es.wav via AudioCodec
    SS->>TM: Models.loadTranscription(model, Arena.ofShared())
    Note over SS: phase 0
    SS->>P: program(seed, 30 min)
    SS->>TM: transcribe(concat pcm) offline
    SS->>TM: newState + stream(state), feed 0.5s chunks, finish
    SS->>SS: EQUAL / MISMATCH
    Note over SS: soak
    SS->>P: program(seed, hours*3600): slot%13 silence, %17 collapse-clip,<br/>%19 language-switch, %23 tone-sweep, %11 gain-step, else clip
    SS->>TM: newState() then stream(state)
    loop every 0.5 s chunk of every block
        SS->>SS: markSpeech (RMS > 0.02)
        SS->>TS: feed(block, from, n) → final tokens, check order
        opt every 10 s fed
            SS->>TS: partial() timed → per-hour latency, ahead-of-audio, order, mute alarm
        end
        opt every 5 simulated min
            SS->>SS: rssMb() via ps
        end
    end
    SS->>TS: finish()
    SS->>SS: holes ≥45 s, recovery ≤60 s per Event, RSS slope, p50/p99, violations
```

`newState` / `stream` are unchecked generic helpers that rebind the wildcard `TranscriptionModel<?, ?, ?>`
to a type variable `S extends RuntimeState`.

---

## 7. Kernel microbenches

All three run on `jinfer-core`/`jinfer-kernels` primitives with no model loaded.

```mermaid
flowchart LR
    subgraph ConvPeak
        A1{"args[0] census file?"} -- yes --> A2["census: regex rows → buckets by channels,kernel,dilation<br/>FLOP-weighted mean time; skip share below 0.2%"]
        A1 -- no --> A3["LADDER: 4 vocoder shapes"]
        A2 --> A4["measure: warmup ≥1.5 s and stable 3%, ≤6 s; 5 reps"]
        A3 --> A4
        A4 --> A5["median rate G/s; predicted = gflops / rate; TOTAL"]
    end
    subgraph ConvParity
        B1["6 fixed shapes incl. ragged channels/time"] --> B2["Random 1234 inputs, taps, bias"]
        B2 --> B3["conv1dRows"] --> B4["SHA-256 of raw float bits, 32 hex"]
    end
    subgraph SpinProbe
        C1["5000 warm empty forLoop"] --> C2["iters × Parallel.forLoop 0..threads, no-op"] --> C3["µs per region"]
    end
```

The conv tile shape is a static final read once per JVM (`-Djinfer.convTile=auto|4x2|4x4`), so both
conv tools are meant to be run once per tile value and their outputs compared.

---

## 8. Design notes

- **Generic, no per-model code** — `JinferBench`, `EmbedBench`, `TranscriptionBench` and `StreamSoak` all
  resolve models through `Models` (ServiceLoader over `jinfer-models-all`). The only architecture-specific
  code is the decision-checkpoint fallback naming `Lfm3` and `Qwen35`, which is why those two ports are
  compile-scope dependencies.
- **Allocation outside timing** — one state per test, `reset()` / `resetToDepth()` before every pass;
  `newState` returns its own elapsed time so it is reported separately (capabilities `state ms`).
- **Pinned compute width** — `UBATCH = 512` is a constant rather than `RuntimeFlags.BATCH_CAPACITY`, so
  `-Djinfer.batchCapacity` cannot change what is measured.
- **Adaptive warmup** — throughput must settle within 3% over a window of 3 (JIT needs more than
  llama-bench's single pass); `ConvPeak` additionally enforces a minimum warmup wall time.
- **Dead-code guards** — `volatile` `sink` fields in `JinferBench` and `EmbedBench`.
- **Self-describing output** — `header()` reports native image vs GraalVM JIT vs C2 (`Segments.FAST_VECTOR_JIT`),
  the loaded `JAM.providers()` ids, and the thread count source.
- **Memory measurement** — `JinferBench` reads Linux `VmHWM`; `StreamSoak` shells out to `ps -o rss=`.
