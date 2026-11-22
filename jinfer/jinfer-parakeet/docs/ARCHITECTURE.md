# jinfer-parakeet — class architecture

Package `com.qxotic.jinfer.models.parakeet` (5 top-level classes, ~1530 lines). It is the `jinfer-<model>` provider for
GGUF architecture `parakeet` — NVIDIA Parakeet **speech recognition** (speech-to-text): a FastConformer encoder plus a
TDT (token-and-duration transducer) decoder, ported from parakeet.cpp. It is transcription-only: the provider overrides
only `loadTranscription`; CTC-only checkpoints (no `parakeet.tdt.durations`) are refused at load.

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends; `*--` owns
(nested/composed); classes marked `<<external>>` live outside this package.

1. [Position in the stack](#1-position-in-the-jinfer-stack)
2. [Class diagrams](#2-class-diagrams) — 2a contracts, 2b encoder internals, 2c decoder internals
3. [Loading sequence](#3-loading-sequence)
4. [Transcription: offline and streaming](#4-transcription-offline-and-streaming)
5. [Encoder forward pass](#5-encoder-forward-pass) — 5a subsampling, 5b Conformer block
6. [TDT greedy decoding](#6-tdt-greedy-decoding)
7. [State and memory](#7-state-and-memory)

| Class | Role |
|---|---|
| `Parakeet` | The model: `TranscriptionModel<Configuration, Weights, State>` with nested records `Configuration`, `Weights`, class `State` (extends `RuntimeState`, owns a `Workspace`) and private inner class `Stream` (`TranscriptionStream`, windowed chunked decoding) |
| `ParakeetProvider` | `ModelProvider` SPI: `architectures() = {"parakeet"}`, `loadTranscription(...)` → `Parakeet.load` |
| `ParakeetEncoder` | FastConformer encoder: NeMo log-mel front end (`LogMel`), 8x depthwise-separable conv subsampling, macaron Conformer blocks with Transformer-XL relative-position attention. Nested records `Config`, `Linear`, `Norm`, `Block`, `Scratch` |
| `ParakeetTdt` | TDT decoder: stacked-LSTM prediction network, ReLU joint network, greedy token+duration loop with a blank watchdog. Nested `Config`, `Emission`, `State` (LSTM h/c), `Cursor` (resumable decode position), `Scratch` |
| `Tensors` | Package-private record over `Map<String, MemoryView>`: `require`, `optional`, `weight` (checks leading extent), `vector` (checks size), `floats` (copies small F32 to heap) |

SPI registration: `src/main/resources/META-INF/services/com.qxotic.jinfer.chat.ModelProvider` contains
`com.qxotic.jinfer.models.parakeet.ParakeetProvider`.

Module dependencies (`pom.xml`): `jota-core`, `jota-memory`, `jinfer-core`, `jinfer-kernels`, `jinfer-chat`, `gguf`;
test-only `jinfer-codecs` (decodes/resamples test clips) and `jinfer-testkit`. No tokenizer dependency: the
SentencePiece pieces come from GGUF key `parakeet.tokenizer.pieces` and are mapped to text directly.

---

## 1. Position in the jinfer stack

```mermaid
flowchart TB
    Callers["Callers: cli / server / integrations"]
    subgraph chat["jinfer-chat"]
        Models["Models.loadTranscription - ServiceLoader"]
        MP["ModelProvider"]
    end
    subgraph core["jinfer-core"]
        TM["TranscriptionModel / Model"]
        TS["TranscriptionStream"]
        TR["Transcription + Token"]
        RS["RuntimeState"]
        WS["Workspace · Arenas · Views · LeakWatch"]
        EV["telemetry.InferenceEvent"]
    end
    subgraph pk["jinfer-parakeet"]
        P["ParakeetProvider"]
        M["Parakeet"]
        E["ParakeetEncoder"]
        D["ParakeetTdt"]
        T["Tensors"]
    end
    subgraph kern["jinfer-kernels"]
        K["ModelLoader · LogMel · MatMul · Norms<br/>Ops · Activations · Convert"]
    end
    G["gguf: GGUF metadata"]
    J["jota-memory: MemoryView / MemoryArena"]

    Callers --> Models --> P
    P -.->|implements| MP
    P -->|"load"| M
    M -.->|implements| TM
    M --> E
    M --> D
    E --> T
    D --> T
    M --> TS
    M --> TR
    M --> RS
    M --> WS
    M --> EV
    E --> K
    D --> K
    M --> G
    T --> J
```

---

## 2. Class diagrams

### 2a. Contracts and top-level relations

```mermaid
classDiagram
    direction LR
    class ModelProvider { <<interface, external>> }
    class Model~C,W,S~ { <<interface, external>> }
    class TranscriptionModel~C,W,S~ {
        <<interface, external>>
        sampleRate() int
        newState() S
        newState(arena) S
        transcribe(S, pcm) Transcription
        stream(S) TranscriptionStream
    }
    class TranscriptionStream {
        <<interface, external>>
        sampleRate() int
        feed(pcm, offset, length) Transcription
        partial() Transcription
        finish() Transcription
        close()
    }
    class RuntimeState { <<abstract, external>> }
    class ParakeetProvider {
        <<final>>
        architectures() Set
        loadTranscription(channel, gguf, path, arena, companions)
    }
    class Parakeet {
        <<final>>
        +ARCHITECTURE = "parakeet"$
        -Configuration configuration
        -Weights weights
        -String name
        +load(channel, gguf, arena)$ Parakeet
        +transcribe(State, pcm) Transcription
        +stream(State) TranscriptionStream
        ~chunkFrames(value, sampleRate, frameSamples)$ int
    }
    class Configuration {
        <<record>>
        sampleRate, dModel, layers
        vocabSize, subsamplingFactor, hop
        frame() Duration
    }
    class Weights {
        <<record>>
        ParakeetEncoder encoder
        ParakeetTdt decoder
    }
    class State {
        <<Parakeet.State>>
        -MemoryArena owned
        -MemoryArena allocator
        -Workspace workspace
        -Runnable disarm
    }
    class Stream {
        <<private inner>>
        -ParakeetTdt.Cursor cursor
        -float[] buffer
        -long bufferStart, chunkEnd
        -InferenceEvent event
    }
    class Tensors {
        <<record, package-private>>
        require(name)
        optional(name)
        weight(name, out)
        vector(name, length)
        floats(name, length) float[]
    }
    class ParakeetEncoder { <<final>> }
    class ParakeetTdt { <<final>> }

    ModelProvider <|.. ParakeetProvider
    Model <|-- TranscriptionModel
    TranscriptionModel <|.. Parakeet
    TranscriptionStream <|.. Stream
    RuntimeState <|-- State
    ParakeetProvider ..> Parakeet : load
    Parakeet *-- Configuration
    Parakeet *-- Weights
    Parakeet ..> State : newState
    Parakeet ..> Stream : stream / transcribe
    Weights --> ParakeetEncoder
    Weights --> ParakeetTdt
    Stream --> State
    Parakeet ..> Tensors : new from ModelLoader.loadTensors
    ParakeetEncoder ..> Tensors : load
    ParakeetTdt ..> Tensors : load
```

### 2b. Encoder internals — `ParakeetEncoder`

```mermaid
classDiagram
    direction TB
    class ParakeetEncoder {
        <<final>>
        -Config config
        -LogMel logMel
        -float[] conv0Taps, conv0Bias
        -float[] dw1Taps, dw1Bias, dw2Taps, dw2Bias
        -Linear pw1, pw2, preOut
        -Block[] blocks
        ~load(gguf, tensors)$
        ~mel(pcm, from, length) float[]
        ~frames(samples) int
        ~encode(pcm, from, length, layerTap, workspace) MemoryView
        ~preEncode(mel, melFrames, frames, valid, tile, workspace)
        ~relativeAttention(work, frames, valid, dim, heads)$
        ~relativePositions(frames, dModel, table)$
        -halfFfn() attention() convolution()
        -conv0() separable() positionTable()
    }
    class Config {
        <<record>>
        dModel, layers, heads, ffDim, convKernel
        subsamplingChannels, nMels, nFft, hop
        preemphasis, magPower, logZeroGuard, xscaling
    }
    class Linear {
        <<record>>
        weight, bias nullable, in, out
        apply(input, output, rows)
    }
    class Norm {
        <<record>>
        weight, bias
        apply(output, input, rows)
    }
    class Block {
        <<record>>
        ff1Norm ff1Up ff1Down
        attnNorm query key value position output
        posBiasU posBiasV
        convNorm pointwise1 depthwise bnScale bnShift pointwise2
        ff2Norm ff2Up ff2Down outNorm
    }
    class Scratch {
        <<record>>
        norm ff queryU queryV key value
        attention position pointwise glu
        scores positionScores valueT attentionT
        gated mixed
        allocate(workspace, frames, dim, ffDim)$
    }
    class LogMel { <<external, jinfer-kernels>> }
    class MatMul { <<external>> }
    class Norms { <<external>> }
    class Workspace { <<external>> }

    ParakeetEncoder *-- Config
    ParakeetEncoder *-- "layers" Block
    ParakeetEncoder *-- Linear : pw1 pw2 preOut
    ParakeetEncoder --> LogMel : featurizer
    Block *-- Linear
    Block *-- Norm
    ParakeetEncoder ..> Scratch : per encode
    Scratch ..> Workspace : drawn from
    Linear ..> MatMul : mm
    Norm ..> Norms : layerNormRows
```

### 2c. Decoder internals — `ParakeetTdt`

```mermaid
classDiagram
    direction TB
    class ParakeetTdt {
        <<final>>
        -Config config
        -MemoryView embed
        -MemoryView[] weightInput, weightHidden
        -float[][] gateBias
        -MemoryView encWeight, predWeight, outWeight
        -float[] encBias, predBias, outBias
        -WATCHDOG_FRAMES = 60$
        ~load(gguf, tensors)$
        ~encProjection(encoded, frames, workspace) float[]
        ~decode(encProj, first, until, cursor, workspace) List~Emission~
        ~predStep(token, in, out, scratch) float[]
        ~jointLogits(encProj, frame, prediction, logits, scratch)
        ~isSpecial(piece)$ boolean
        ~probeSos() probeJointLogits()
    }
    class Config {
        <<record>>
        vocabSize, blankId
        predHidden, predLayers
        jointHidden, encHidden
        int[] durations, maxSymbols
        String[] pieces
        tokenCount() int
        vPlus() int
    }
    class Emission {
        <<record>>
        token, frame, duration, confidence
    }
    class TdtState {
        <<ParakeetTdt.State>>
        float[][] hidden
        float[][] cell
        copyFrom(other)
    }
    class Cursor {
        -State committed
        -int lastToken
        -int frame
        -int lastEmissionFrame, quietSince, rescans
        frame() int
        copy() Cursor
    }
    class TdtScratch {
        <<ParakeetTdt.Scratch>>
        x h zInput zHidden fused logits
        zInputArr zHiddenArr fusedArr
    }

    ParakeetTdt *-- Config
    ParakeetTdt ..> Emission : emits
    ParakeetTdt ..> Cursor : advances
    Cursor *-- TdtState : committed
    ParakeetTdt ..> TdtState : predStep in/out
    ParakeetTdt ..> TdtScratch : per decode
```

`ParakeetTdt.State` and `ParakeetTdt.Scratch` are shown as `TdtState`/`TdtScratch` to avoid clashing with
`Parakeet.State` and `ParakeetEncoder.Scratch`.

---

## 3. Loading sequence

```mermaid
sequenceDiagram
    participant C as caller
    participant Ms as Models (jinfer-chat)
    participant P as ParakeetProvider
    participant Pk as Parakeet
    participant ML as ModelLoader
    participant E as ParakeetEncoder
    participant D as ParakeetTdt

    C->>Ms: loadTranscription(path, arena[, companions])
    Ms->>Ms: ServiceLoader: provider for general.architecture
    Ms->>P: loadTranscription(channel, gguf, path, arena, companions)
    P->>Pk: load(channel, gguf, arena)
    Pk->>Pk: require general.architecture == "parakeet"
    Pk->>ML: loadTensors(channel, gguf, arena)
    ML-->>Pk: Map name -> MemoryView
    Pk->>Pk: new Tensors(map)
    Pk->>E: load(gguf, tensors)
    E->>E: Config.from(gguf), check magPower in {1, 2}
    E->>E: per layer: fold BatchNorm (+ optional depthwise bias) into bnScale/bnShift
    E->>E: featurizer: centered window + filterbank -> LogMel
    Pk->>D: load(gguf, tensors)
    D->>D: refuse if parakeet.tdt.durations absent (CTC-only)
    D->>D: fold bias_ih + bias_hh into gateBias per LSTM layer
    Pk->>Pk: Configuration(sample_rate, dModel, layers, vocabSize, subsampling_factor, hop)
    Pk-->>C: Parakeet (name = general.name or "parakeet")
```

Notable load-time folds: BatchNorm running stats become a per-channel affine (`bnScale`, `bnShift`), and the depthwise
conv bias (absent on v2/v3) is folded into `bnShift`; the LSTM's two biases are summed once. Conformer `Linear` biases
are optional (bias-free checkpoints such as tdt-0.6b-v3); subsampling projections always have one.

---

## 4. Transcription: offline and streaming

Both paths go through the private `Stream`. Windows follow NeMo's chunked streaming: left context + chunk + right
context, in encoder frames (80 ms each). Only the chunk's tokens become final; the decoder `Cursor` carries over so words
continue across chunks.

| Mode | chunk | right | left |
|---|---|---|---|
| `stream(state)` live | 25 frames (2 s) | 25 (2 s) | 125 (10 s) |
| `transcribe(state, pcm)` offline | 575 (46 s) | 50 (4 s) | 125 (10 s) |
| `partial()` | up to fed audio minus 3 edge frames | none | 38 (~3 s) |

`-Djinfer.parakeet.chunkSeconds` (integer 1..3600) overrides the chunk size in both modes (`chunkFrames`).

```mermaid
sequenceDiagram
    participant C as caller
    participant Pk as Parakeet
    participant St as Stream
    participant S as Parakeet.State
    participant E as ParakeetEncoder
    participant D as ParakeetTdt

    C->>Pk: stream(state) or transcribe(state, pcm)
    Pk->>St: new Stream(state, chunkFrames, rightFrames)
    St->>St: cursor = new Cursor(config), InferenceEvent.started
    loop feed(pcm, off, len)
        St->>S: exclusively
        St->>St: append to buffer (grow x2)
        loop while buffered end >= chunkEnd + right
            St->>St: decode(bufferStart, chunkEnd+right, until=chunkEnd, cursor)
            St->>St: chunkEnd += chunk, then drop samples before chunkEnd - chunk - left
        end
        St-->>C: final piece (Transcription)
    end
    opt partial()
        St->>St: decode(from, end, end - 3 frames, cursor.copy())
        St-->>C: provisional Transcription
    end
    C->>St: finish()
    St->>St: decode(bufferStart, end, MAX, cursor), close()
    St->>St: event: inputTokens = frames fed, outputTokens, decodeTime = compute ns, commit
    St-->>C: last piece
```

`transcribe` = one `feed(pcm)` + `finish()`, concatenating text and tokens.

### `Stream.decode(from, to, until, cursor)`

```mermaid
flowchart TB
    R["state.workspace.rewind()"] --> EN["ParakeetEncoder.encode(buffer, offset, length) → [frames × dModel]"]
    EN --> PR["ParakeetTdt.encProjection → float[frames × jointHidden]"]
    PR --> DE["ParakeetTdt.decode(projected, first, last, cursor) → List of Emission"]
    DE --> MAP["per emission: pieces[token]; skip isSpecial; '▁' → ' '"]
    MAP --> TOK["Transcription.Token(text, frame·dt, min(end, audioEnd), confidence)"]
```

Token times are absolute from stream start (`first = from / frameSamples`); `frameSamples = hop × subsamplingFactor`.
Compute time is accumulated in `computeNanos` so waiting for live audio is not billed. The first token of a transcript
drops its leading space.

---

## 5. Encoder forward pass

### 5a. `encode` and subsampling

```mermaid
flowchart TB
    PCM["pcm from..from+length"] --> MEL["mel: LogMel.frames + normalizePerFeature → [melFrames × nMels]"]
    MEL --> PE
    subgraph PE["preEncode, tiles of 64 encoder frames, Workspace.scope per tile"]
        direction TB
        C0["conv0: 3x3 stride-2 conv 1→C, bias, ReLU"] --> S1["separable dw1 3x3 stride-2 + pw1 + ReLU"]
        S1 --> S2["separable dw2 3x3 stride-2 + pw2 + ReLU"]
        S2 --> FL["flatten channel-major C·F3, zero frames past valid"]
        FL --> PO["preOut Linear → x rows j0..j1"]
    end
    PE --> XS{"xscaling?"}
    XS -->|yes| MUL["x *= sqrt dModel"]
    XS -->|no| POS
    MUL --> POS["positionTable: sinusoids for offsets frames-1 .. 1-frames"]
    POS --> BL["for each Block: halfFfn → attention → convolution → halfFfn → outNorm"]
    BL --> OUT["x: [frames × dModel], valid until next rewind"]
```

`frames(samples)` = three halvings `(n-1)/2+1` of `1 + samples/hop`. Tiling keeps subsampling intermediates near 40 MB
for any window length; `valid` excludes the trailing center-pad mel frame. An optional `layerTap` receives per-layer
outputs (used for parity checks).

### 5b. Conformer block (macaron)

```mermaid
flowchart TB
    X["x"] --> F1["halfFfn: LayerNorm → up → SiLU → down; x += 0.5·out"]
    F1 --> AT
    subgraph AT["attention"]
        direction TB
        AN["LayerNorm"] --> QKV["query, key, value, position Linear"]
        QKV --> UV["queryU = q + posBiasU; queryV = q + posBiasV"]
        UV --> RA["relativeAttention per head:<br/>scores = queryU·keyᵀ + shift of queryV·posᵀ<br/>scale, softmax over valid keys<br/>attentionᵀ = valueᵀ·scoresᵀ"]
        RA --> AO["output Linear; x += out"]
    end
    AT --> CV
    subgraph CV["convolution"]
        direction TB
        CN["LayerNorm"] --> P1["pointwise1 → 2·dim; GLU → dim; zero past valid"]
        P1 --> DW["depthwise conv kernel K, folded BatchNorm affine, SiLU"]
        DW --> P2["pointwise2; x += out"]
    end
    CV --> F2["halfFfn ff2; x += 0.5·out"]
    F2 --> ON["outNorm LayerNorm"]
```

Only the `valid × valid` block of attention is computed; padded query rows are zeroed. Heads run sequentially (MatMul
`mm` refuses nested parallel regions); per-row work uses `Parallel.forLoop`.

---

## 6. TDT greedy decoding

`decode(encProjection, first, until, cursor, workspace)` runs from `cursor.frame` up to `until`; `vPlus = (vocab+1) +
durations.length` logits per step: the token slice and the duration slice.

```mermaid
flowchart TB
    START["t = cursor.frame"] --> WD{"t - quietSince >= 60 frames?"}
    WD -->|yes| RES["rescan: t = lastEmissionFrame+1 if rescans < 2;<br/>from 2nd rescan reset committed state, lastToken = -1"]
    WD -->|no| INNER
    RES --> INNER
    subgraph INNER["inner loop while skip == 0 and symbols < maxSymbols"]
        direction TB
        PS["predStep(lastToken, committed → stepped)<br/>zero input when lastToken = -1"] --> JL["jointLogits: outW·ReLU(encProj[t] + predW·pred + b) + b"]
        JL --> AM["token = argmax token slice; skip = durations argmax duration slice"]
        AM --> EM{"token != blank?"}
        EM -->|yes| COMMIT["emit Emission token, t, skip, confidence;<br/>committed ← stepped; reset watchdog"]
        EM -->|no| ADV
        COMMIT --> ADV["t += skip"]
    end
    INNER --> BUD{"symbols == maxSymbols?"}
    BUD -->|yes| PLUS["t += 1"]
    BUD -->|no| LOOP
    PLUS --> LOOP{"t < until?"}
    LOOP -->|yes| WD
    LOOP -->|no| END["cursor.frame = t; return emissions"]
```

Details worth knowing:

- Matches parakeet.cpp `tdt.cpp`/`prediction.cpp`: the LSTM state commits only on emission; PyTorch `i,f,g,o` gate order.
- The blank watchdog addresses greedy TDT getting stuck on blank (state only advances on emission): first rescan with
  the same state, then with a fresh one; if both stay silent the span is treated as silence.
- `confidence` = token-slice max softmax probability rescaled as `(N·p − 1)/(N − 1)`, clamped to [0, 1].
- `encProjection` does one `MatMul.gemm` for all frames of the window; per-step projections are single-row gemms.
- `probeSos` / `probeJointLogits` are parity probes used by tests.

---

## 7. State and memory

```mermaid
flowchart TB
    subgraph PS["Parakeet.State extends RuntimeState"]
        A["allocator: MemoryArena (cross-thread required)"]
        O["owned: same arena if newState(), null if borrowed"]
        W["Workspace over allocator: rewound every decode window"]
        L["LeakWatch.arm disarm hook"]
    end
    subgraph SS["Stream (per stream, heap)"]
        BUF["float[] buffer: starts at 8 s, doubles"]
        CUR["ParakeetTdt.Cursor: committed LSTM h/c, lastToken, frame, watchdog"]
    end
    W --> ENC["encoder Scratch, subsampling tiles, position table"]
    W --> DEC["encProjection, TDT Scratch, logits"]
    SS -->|"exclusively"| PS
```

- `newState()` creates an owned cross-thread arena (closed on failure and in `releaseResources`); `newState(arena)`
  borrows one (`Arenas.requireCrossThread`).
- `checkResourcesAlive` fails if the arena was closed. `scratchAllocations()` (backing + heap) lets tests check that a
  warm state transcribes without allocating.
- Weights are the GGUF `MemoryView`s from `ModelLoader.loadTensors`, plus small heap copies (conv taps, folded biases,
  BatchNorm affine).
