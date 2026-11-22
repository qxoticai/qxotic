# jinfer-lfm3 — class architecture

Package `com.qxotic.jinfer.models.lfm3` (11 classes, ~4,300 lines). It is the `jinfer-<model>` provider for GGUF
architecture `lfm3` (Liquid AI): the **LFM2-style hybrid** of gated short convolutions and GQA attention, where
attention layers alternate sliding-window (ring KV cache, plain RoPE) and full (partial RoPE via `rope_freqs`), every
attention output is gated per head by a sigmoid, and both sublayers carry sandwich post-norms. One backbone serves five
faces: LFM3 (text), LFM3-VL (+ vision tower), LFM3-Omni (+ audio tower + speech head), and the **d1 / d1-Omni decision
checkpoints** (bidirectional full attention, no text generation). Towers come from the `mmproj` companion GGUF.
There is no speculative-decoding class in this module.

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends;
classes marked `<<external>>` live outside this package.

1. [Position in the stack](#1-position-in-the-jinfer-stack)
2. [Class diagrams](#2-class-diagrams) — 2a contracts, 2b model internals, 2c media towers and speech, 2d decision model
3. [Loading sequences](#3-loading-sequences)
4. [Forward pass](#4-forward-pass) — 4.0 global, 4a `ingest`, 4b `decoderBlock` (+ 4b-i short conv, 4b-ii attention)
5. [Heads: logits, targeted logits, hidden state](#5-heads)
6. [State and checkpoint layout](#6-state-and-checkpoint-layout)
7. [Vision tower (LFM3-VL)](#7-vision-tower)
8. [Audio encoder (LFM3-Omni)](#8-audio-encoder)
9. [Speech head (LFM3-Omni output)](#9-speech-head)
10. [Chat template](#10-chat-template)
11. [Decision flow (d1)](#11-decision-flow-d1)

| Class | Role |
|---|---|
| `Lfm3` | The model: `LanguageModel` + `Multimodal` + `SpeechOutput` impl with nested `Configuration`, `Weights`, `LayerWeights`, `AttentionWeights`, `ShortConvWeights`, `State`, `Placed`; the whole forward pass, loading, `withMedia` |
| `Lfm3Provider` | `ModelProvider` SPI: `loadLanguage` (refuses decision checkpoints), `loadDecisionModel` (refuses chat checkpoints), `companionFiles` = `media` -> `mmproj` |
| `Lfm3ChatTemplate` | `ChatTemplate`: token-exact ChatML-like framing, tools as XML, `think` handling, media framing, spoken-turn replay, `tts` instruction helper |
| `Lfm3CheckpointCodec` | `CheckpointCodec<State>`: full-layer KV rows, SWA rings walked by position, short-conv taps |
| `Lfm3Decision` | `DecisionModel` + `Multimodal` for d1 / d1-Omni: packed prompts, label-token scoring, softmax with per-kind temperature |
| `Lfm3Vision` | `MediaProjector<Image>`: patchifier, conv / 2D-RoPE attention blocks, 2x2 merge, GELU MLP; also video frame pairs |
| `Lfm3VisionPreprocess` | `smart_resize` target sizes (image, video) and normalised patch vectors |
| `Lfm3Audio` | `MediaProjector<Audio>`: log-mel, 8x conv downsampler, causal LFM stack, MLP into model width |
| `Lfm3Speech` | Omni speech head: depthformer (MaskGIT), feedback rows, FSQ codec, streaming ISTFT; inner `Stream`, `Speaker`, `Feedback`, `Istft` |
| `Towers` | Shared helpers: per-head norm + NeoX RoPE, SwiGLU FFN, F16 roundings, 3x3 tap layout |
| `GeluErf` | Exact (erf) GELU for the two projectors |

---

## 1. Position in the jinfer stack

```mermaid
flowchart TB
    Callers["Callers: cli / server / langchain4j / spring-ai"]
    subgraph chat["jinfer-chat"]
        Models["Models (ServiceLoader)"]
        MP["ModelProvider"]
        LM["LoadedModel (model, tokenizer, template, stops, seed)"]
        DM["DecisionModel entry point"]
    end
    subgraph core["jinfer-core"]
        LMI["LanguageModel / Multimodal / SpeechOutput"]
        CS["ContextState"]
        CC["CheckpointCodec"]
        CT["ChatTemplate"]
        B["Batch (Tokens | Embeddings)"]
    end
    subgraph l3["jinfer-lfm3"]
        P["Lfm3Provider"]
        M["Lfm3"]
        T["Lfm3ChatTemplate"]
        C["Lfm3CheckpointCodec"]
        D["Lfm3Decision"]
        V["Lfm3Vision"]
        A["Lfm3Audio"]
        S["Lfm3Speech"]
    end
    subgraph kern["jinfer-kernels"]
        K["MatMul · Norms · RoPE · FlashAttention · Convolutions<br/>Activations · Convert · Ops · Resample · LogMel · ModelLoader"]
    end
    Tok["toknroll Tokenizer"]
    Files["model.gguf + mmproj-*.gguf"]

    Callers --> Models --> P
    P -.->|implements| MP
    P -->|produces| LM
    P -->|produces| DM
    M -.->|implements| LMI
    T -.->|implements| CT
    C -.->|extends| CC
    D -.->|implements| DM
    M --> V
    M --> A
    M --> S
    D --> M
    M --> K
    V --> K
    A --> K
    S --> K
    M --> Tok
    P --> Files
```

---

## 2. Class diagrams

### 2a. Contracts and relations

```mermaid
classDiagram
    direction LR
    class ModelProvider { <<interface, external>> }
    class LanguageModel~C,W,S~ { <<interface, external>> }
    class Multimodal { <<interface, external>> }
    class SpeechOutput~S~ { <<interface, external>> }
    class DecisionModel { <<interface, external>> }
    class ChatTemplate { <<interface, external>> }
    class MediaProjector~R~ { <<interface, external>> }
    class ContextState { <<abstract, external>> }
    class CheckpointCodec~S~ { <<abstract, external>> }
    class Lfm3Provider {
        <<final>>
        architectures() "lfm3"
        companionFiles() media to mmproj
        loadLanguage(...) LoadedModel
        loadDecisionModel(...) DecisionModel
    }
    class Lfm3 { <<final>> }
    class Lfm3ChatTemplate { <<package-private>> }
    class Lfm3CheckpointCodec { <<package-private>> }
    class Lfm3Decision { <<package-private>> }
    class State { <<Lfm3.State>> }

    ModelProvider <|.. Lfm3Provider
    LanguageModel <|.. Lfm3
    Multimodal <|.. Lfm3
    SpeechOutput <|.. Lfm3
    DecisionModel <|.. Lfm3Decision
    Multimodal <|.. Lfm3Decision
    ChatTemplate <|.. Lfm3ChatTemplate
    CheckpointCodec <|-- Lfm3CheckpointCodec
    ContextState <|-- State
    Lfm3Provider ..> Lfm3 : loadModel / withMedia
    Lfm3Provider ..> Lfm3ChatTemplate : new(model)
    Lfm3Provider ..> Lfm3Decision : new(model, gguf)
    Lfm3ChatTemplate --> Lfm3 : Multimodal media
    Lfm3Decision --> Lfm3 : model
    Lfm3 ..> Lfm3CheckpointCodec : checkpointCodec()
    Lfm3CheckpointCodec ..> State : reads/writes
    Lfm3 *-- State : newState
    Lfm3 o-- MediaProjector : projector(Image or Audio)
```

### 2b. Model internals — `Lfm3` and its nested types

```mermaid
classDiagram
    direction TB
    class Lfm3 {
        <<final>>
        -Configuration configuration
        -Tokenizer tokenizer
        -Weights weights
        -float[][] ffnScales
        -Lfm3Vision vision
        -Lfm3Audio audio
        -Lfm3Speech speech
        +loadModel(path or channel, arena)$
        +withMedia(mmproj, arena) Lfm3
        +ingest(State, Batch)
        ~ingest(State, Batch, afterBlock)
        ~ingestPrompts(State, tokens, prompts, media)
        +logits(State, output) MemoryView
        ~logits(State, output, tokens, out)
        ~hiddenState(State, out)
        -forward(State, Batch, afterBlock)
        -tail(State, start, rows)
        -block(State, layer, start, rows)
        -shortConv() attention() projectKeys()
        -twoWayAttention()
        -commitKv()
    }
    class Configuration {
        <<record>>
        embeddingLength feedForwardLength
        numberOfLayers numberOfHeads
        kvHeads[] (0 = short conv)
        isSwa[] headSize
        vocabularySize maxContextLength
        rmsNormEps ropeTheta ropeThetaSwa
        slidingWindow shortConvLCache
        causalAttention
        lastAttentionLayer() tailRows()
        cachePositions(layer, ctx)
    }
    class Weights {
        <<record>>
        tokenEmbeddings
        LayerWeights[] layers
        outputNorm
        output (tied if absent)
        RoPE.Schedule ropeFull ropeSwa
    }
    class LayerWeights {
        <<record>>
        attnNorm postAttnNorm
        ffnNorm postFfnNorm
        attention XOR shortConv
        ffnGate ffnUp ffnDown
    }
    class AttentionWeights {
        <<record>>
        query key value output
        gate queryNorm keyNorm
    }
    class ShortConvWeights {
        <<record>>
        kernel inProj outProj
    }
    class Placed {
        <<record>>
        at rows count
    }
    class State {
        <<static final>>
        residual normed branch logits
        query gate attnOut batchK batchV
        hidden hidden2
        shortConvIn shortConvOut
        ropeCos/Sin Full + Swa
        keyCache[] valueCache[] F16
        shortConvState[] F32
        decodeScratch
        prompts[][] media List
        scores valuesT lazily
        clearHistory() zeroes conv state
    }
    Lfm3 *-- Configuration
    Lfm3 *-- Weights
    Weights *-- LayerWeights
    LayerWeights *-- AttentionWeights
    LayerWeights *-- ShortConvWeights
    Lfm3 ..> State : newState
    Lfm3 ..> Placed : ingestPrompts
    State o-- Placed : media
    Configuration ..|> ContextConfiguration
```

### 2c. Media towers and speech head

```mermaid
classDiagram
    direction TB
    class MediaProjector~R~ { <<interface, external>> }
    class Lfm3Vision {
        <<final>>
        patch dim heads kvHeads headDim
        patchKernel first/secondKernel
        Layer[] layers (conv XOR attention)
        projector norm + up + down
        positions(Image) planId()
        project(Image, maxChunk, sink)
        encode(Image, arena, afterLayer)
        encode(first, second, size, arena) video pair
        videoSize(frames, frame)
    }
    class Lfm3VisionPreprocess {
        <<final>>
        targetSize(w, h, factor, min, max)$
        videoSize(frames, w, h, factor)$
        patches(rgb, w, h, patch, mean, std)$
    }
    class Lfm3Audio {
        <<final>>
        SAMPLE_RATE 16000
        LogMel nMel channels window
        conv0 depthwise1 depthwise2 pointwise
        Layer[] layers (short conv XOR attention)
        positions(Audio) mono() mel()
        encode(Audio, arena, beforeLayer)
    }
    class Lfm3Speech {
        <<final>>
        SAMPLE_RATE 24000 CODEBOOKS 12
        depth Stack, codec Stack
        fsq table, Sampling defaults
        newStream(Sampling) Stream
        speaker(model, state, ...) Speaker
    }
    class Stream { frame() decode() flush() feedback() }
    class Speaker { textNext() text() frame() flush() }
    class Feedback { of(codes) batch(codes) }
    class Istft { frame() flush() reset() }
    class Towers {
        <<final>>
        headNormRope()$
        ffn()$
        roundToHalf()$
        tapMajor()$
    }
    class GeluErf { <<final>> rows()$ erf()$ }
    class Lfm3 { <<final>> }

    MediaProjector <|.. Lfm3Vision
    MediaProjector <|.. Lfm3Audio
    Lfm3 o-- Lfm3Vision
    Lfm3 o-- Lfm3Audio
    Lfm3 o-- Lfm3Speech
    Lfm3Vision ..> Lfm3VisionPreprocess
    Lfm3Vision ..> Towers
    Lfm3Vision ..> GeluErf
    Lfm3Audio ..> Towers
    Lfm3Audio ..> GeluErf
    Lfm3Speech ..> Towers
    Lfm3Speech *-- Stream
    Lfm3Speech *-- Speaker
    Lfm3Speech *-- Feedback
    Stream *-- Istft
    Stream *-- Feedback
    Speaker --> Stream
    Speaker --> Lfm3 : hiddenState
```

### 2d. Decision model

```mermaid
classDiagram
    direction TB
    class DecisionModel { <<interface, external>> }
    class Question { <<sealed, external>> }
    class Lfm3Decision {
        <<final>>
        GROUP 16
        SHARED_BATCH 1024
        MAX_PIXELS 1638400
        -Lfm3 model
        -turnEnd[] yesNo[][]
        -predicate/choice/rating temperature
        -controls Specials
        newState(contextCapacity)
        decide(state, input, Question)
        decide(state, input, List)
        decideIndependently(...)
        projector() supports(modality)
    }
    class Asked { <<record>> body labels temperature }
    class Judged { <<record>> media text }
    class Head { <<record>> tokens media }
    class Predicate
    class Choice
    class Rating
    DecisionModel <|.. Lfm3Decision
    Question <|-- Predicate
    Question <|-- Choice
    Question <|-- Rating
    Lfm3Decision ..> Asked : asked(question)
    Lfm3Decision ..> Judged : judged(input)
    Lfm3Decision ..> Head : head(...)
    Lfm3Decision --> Lfm3 : ingestPrompts, logits(tokens)
    Head o-- Placed : media rows
```

---

## 3. Loading sequences

`Lfm3Provider` is one SPI entry for two faces. The face is decided by `lfm3.decision.type` / `lfm3.attention.causal` **before** any weight is read; the mmproj is optional and attaches towers afterwards.

```mermaid
sequenceDiagram
    participant M as Models (jinfer-chat)
    participant P as Lfm3Provider
    participant L as Lfm3
    participant ML as ModelLoader
    participant T as Lfm3ChatTemplate
    participant D as Lfm3Decision

    alt loadLanguage
        M->>P: loadLanguage(channel, gguf, path, arena, companions, tokenizer)
        P->>P: decision type set or attention.causal false?
        P--xM: IncompatibleModelException (it is a decision model)
        P->>L: loadModel(channel, gguf, arena, tokenizer)
        L->>L: tokenizer null? build from GGUF
        L->>L: readConfiguration(gguf, vocab) validates per-layer metadata
        L->>ML: loadTensors(channel, gguf, arena)
        L->>L: loadWeights(tensors, config) + ffnScales(tensors)
        L-->>P: Lfm3 (no towers)
        opt companions has media
            P->>L: withMedia(mmproj, arena)
            L->>ML: readGguf + loadTensors(mmproj)
            L->>L: clip.has_vision_encoder -> new Lfm3Vision
            L->>L: clip.has_audio_encoder -> new Lfm3Audio
            L->>L: clip.has_gen_audio_encoder -> new Lfm3Speech
            L-->>P: new Lfm3 sharing backbone weights
        end
        P->>T: new Lfm3ChatTemplate(model)
        P-->>M: LoadedModel(model, tokenizer, jinja template, stops im_end/endoftext, seed, template, SamplingDefaults.NONE)
    else loadDecisionModel
        M->>P: loadDecisionModel(...)
        P->>P: lfm3.decision.type empty?
        P--xM: IncompatibleModelException (it is a language model)
        P->>L: loadModel(...) [+ withMedia]
        P->>D: new Lfm3Decision(model, gguf)
        D->>D: temperatures noul / choice / score, control-token list, turnEnd, yes/no single tokens
        P-->>M: DecisionModel
    end
```

`readConfiguration` requires: architecture `lfm3`; key length equals value length; RoPE spans the head; per-layer `head_count_kv` (0 marks a short-conv layer) and `sliding_window_pattern` of length `block_count`; all attention layers share one KV-head count (one batch K/V buffer serves them all); SWA window a power of two; `shortconv.l_cache > 1`; no unknown `*.scale` tensors (only `ffn_{gate,up,down}.scale`, NVFP4).

---

## 4. Forward pass

`ingest(state, batch)` -> `state.exclusively(forward)`. Carried across calls: per-layer F16 KV cache (full layers linear, SWA layers ring) and the per-short-conv-layer F32 history of the last `lCache - 1` rows.

### 4.0 Global architecture

```mermaid
flowchart TB
    In["Batch: token ids | media embedding rows"] --> Emb["Embedding gather (+ tower rows overlaid)<br/>F32 residual stream rows x dim"]
    subgraph Stack["Hybrid trunk: N decoderBlocks, per-layer kind from Configuration"]
        direction TB
        subgraph Block["decoderBlock (pre-norm, sandwich post-norm)"]
            direction LR
            Mixer["rmsnorm -> mixer -> post-norm"] --> Add1(("+")) --> Ffn["rmsnorm -> SwiGLU FFN -> post-norm"] --> Add2(("+"))
        end
        Mixer2["mixer = short conv (kvHeads == 0)<br/>or gated GQA attention: SWA ring (isSwa) | full (partial RoPE)"]
    end
    Emb --> Stack
    Stack --> FN["outputNorm RMSNorm (at read time)"]
    FN --> H1["logits: full vocab gemv (tied embedding if no output.weight)"]
    FN --> H2["logits for a few tokens only (decision labels)"]
    FN --> H3["hiddenState: normed last row (speech head input)"]
    Stack -.-> St[("State: KV caches (F16) + short-conv history (F32)")]
```

### 4a. `ingest` — batch validation, input kinds, rope, layer loop, advance

```mermaid
flowchart TB
    In["ingest(state, Batch)"] --> Ex["state.exclusively"]
    Ex --> V{"rows in (0, batchCapacity]<br/>start + rows <= contextCapacity<br/>non-causal checkpoint: start == 0"}
    V -->|no| Err["IllegalArgumentException / IllegalStateException"]
    V -->|yes| Sw{"batch.input()"}
    Sw -->|Tokens| T["ids in [0, vocab)<br/>Convert.gatherToF32 -> residual<br/>state.media rows (decision) copied over their positions"]
    Sw -->|Embeddings| E["needs a tower loaded; not bidirectional<br/>width == dim; copyF32 -> residual"]
    Sw -->|Sequences| S["UnsupportedOperationException: LFM3 does not pack sequences"]
    T --> Rope
    E --> Rope
    Rope{"state.prompts == null?"}
    Rope -->|yes| R1["RoPE.fill full + swa tables<br/>positions start .. start+rows"]
    Rope -->|no| R2["positions restart at 0 per prompt"]
    R1 --> Gate
    R2 --> Gate
    Gate{"afterBlock null, causal,<br/>outputs LAST, a trailing conv exists,<br/>rows > tailRows?"}
    Gate -->|yes| Tail["tail(): blocks 0..lastAttn-1 on all rows;<br/>layer lastAttn: only K/V projected + committed;<br/>then last tailRows rows go through the rest"]
    Gate -->|no| Loop{"for layer in 0..numberOfLayers"}
    Loop --> DB["block(l): decoderBlock -> see 4b"]
    DB --> CB["afterBlock.accept(layer) if given"]
    CB --> Loop
    Loop -->|done| Adv
    Tail --> Adv["state.advance(batch): advanceContext(count, outputs)"]
    Adv -.->|later| Heads["logits / logits(tokens) / hiddenState -> see 5"]
```

`ingestPrompts` (decision checkpoints) calls `forward` with `Batch.score(tokens)` and sets `state.prompts` / `state.media` for the duration, so each packed prompt gets its own positions, its own empty conv history, and its own attention scope.

The **tail** shortcut exists because past the last attention layer a row only feeds its own output and the trailing convs' look-back: with `outputs == LAST`, `tailRows() = 1 + trailingConvLayers * (lCache - 1)` rows are re-run through those layers, the other rows are skipped, and the last row is moved back to `rows - 1` so logits and speech find it.

### 4b. `decoderBlock` — one layer

`block(state, layer, start, rows)` works in place on `residual`; layers differ in kind (short conv vs attention), see 4b-i and 4b-ii.

```mermaid
flowchart TB
    R0["residual rows x dim"] --> N1["Norms.rmsnormRowsGgml -> normed (attnNorm)"]
    N1 --> Kind{"w.shortConv != null?"}
    Kind -->|yes| SC["shortConv(): in_proj, scan, out_proj -> branch (4b-i)"]
    Kind -->|no| AT["attention(): gated GQA, KV commit -> branch (4b-ii)"]
    SC --> PA{"postAttnNorm != null"}
    AT --> PA
    PA -->|yes| PN1["rmsnormRowsGgml(branch, postAttnNorm)"]
    PA -->|no| A1
    PN1 --> A1["Ops.addRows: residual += branch"]
    A1 --> N2["rmsnormRowsGgml -> normed (ffnNorm)"]
    subgraph Ffn["SwiGLU FFN with optional NVFP4 scales"]
        direction TB
        N2 --> G["gemm ffnGate -> hidden"]
        G --> GS["x gateScale if != 1"]
        N2 --> U["gemm ffnUp -> hidden2"]
        GS --> SM["Activations.siluMultiplyRows(hidden, hidden2)"]
        U --> SM
        SM --> D["gemm ffnDown -> branch"]
        D --> OS["x outScale (up.scale * down.scale) if != 1"]
    end
    OS --> PF{"postFfnNorm != null"}
    PF -->|yes| PN2["rmsnormRowsGgml(branch, postFfnNorm)"]
    PF -->|no| A2
    PN2 --> A2["Ops.addRows: residual += branch"]
    A2 --> Tr["Trace.sum l_out-layer (if enabled)"]
```

### 4b-i. Short-conv layer

```mermaid
flowchart TB
    X["normed rows x dim"] --> IP["gemm inProj -> shortConvIn rows x 3dim<br/>split B | C | x"]
    IP --> P{"state.prompts == null?"}
    P -->|yes| S1["Convolutions.shortConvScan<br/>kernel dim x lCache, history shortConvState[l]<br/>over all rows"]
    P -->|no| S2["per prompt: zero history, scan that prompt's slice"]
    S1 --> OP["gemm outProj -> branch"]
    S2 --> OP
    S1 -.-> H[("shortConvState[l]: last lCache-1 rows F32")]
```

### 4b-ii. Attention layer

```mermaid
flowchart TB
    X["normed rows x dim"] --> Q["gemm query -> query"]
    X --> Gt["gemm gate -> gate rows x heads"]
    Q --> QN["headNormRope per row (parallel):<br/>per-head RMSNorm queryNorm then NeoX RoPE"]
    X --> PK["projectKeys: gemm key/value -> batchK/batchV<br/>keyNorm + RoPE on K"]
    QN --> Tab{"isSwa?"}
    Tab -->|yes| TS["cos/sin Swa (plain, ropeThetaSwa)<br/>window = slidingWindow, ringMask = window - 1"]
    Tab -->|no| TF["cos/sin Full (rope_freqs partial RoPE, ropeTheta)<br/>window 0"]
    TS --> Path
    TF --> Path
    Path{"non-causal and full layer?"}
    Path -->|yes| TW["twoWayAttention: scores = Q x K via MatMul.mm in blocks of 256 rows,<br/>softmaxRows, times transposed V; prompt parts: shared rows + own rows"]
    Path -->|no| Pr{"state.prompts != null?"}
    Pr -->|yes| PS["slidingWindowPrefill per prompt slice (window stays inside a prompt)"]
    Pr -->|no| One{"rows == 1?"}
    One -->|yes| FD["FlashAttention.flashDecode<br/>(from max 0, start-window+1 if SWA)"]
    One -->|no| SW["FlashAttention.slidingWindowPrefill (window 0 = full causal)"]
    PK --> TW
    PK --> PS
    PK --> FD
    PK --> SW
    TW --> Gate
    PS --> Gate
    FD --> Gate
    SW --> Gate
    PK --> KV{"causalAttention?"}
    KV -->|yes| CK["commitKv: full = linear rows at start;<br/>SWA = ring row (start+row) and ringMask, only the last window of a long chunk"]
    KV -->|no| NK["no commit: nothing would read the cache"]
    Gate["per row, per head: attnOut x sigmoid(gate)"] --> WO["gemm output -> branch"]
```

Details worth knowing:

- `commitKv` happens **inside** the layer (batchK/batchV are single buffers reused by every layer); it is skipped for non-causal checkpoints.
- Scale is `1/sqrt(headSize)`; `kvMul = heads / kvHeads[layer]`; `queryDim = heads * headSize`, `kvDim(layer) = kvHeads[layer] * headSize`.
- The attention gate is a per-head scalar (`attn_gate.weight` is `heads x dim`), applied to the whole head output.
- Short-conv layers use ggml's per-channel taps `c * lCache + k`; their history is state that `clearHistory()` must wipe on reset, unlike KV rows which are masked.

---

## 5. Heads

```mermaid
flowchart LR
    Res["residual row = lastBatchSize - outputCount + output"] --> FN["Norms.rmsnormGgml(outputNorm) -> normed"]
    FN --> LG["logits(state, output)<br/>MatMul.gemv(output weight) -> logits [1 x vocab]"]
    FN --> TG["logits(state, output, tokens, out)<br/>gatherToF32 label rows into hidden scratch,<br/>chunked dot products: no vocabulary matmul"]
    Res --> HS["hiddenState(state, out)<br/>rmsnormGgml of the LAST ingested row"]
    HS --> Sp["Lfm3Speech.Stream.frame"]
    TG --> Dec["Lfm3Decision.probabilities"]
```

`output.weight` falls back to the token embeddings when absent (tied head). `logits(...)` validates `0 <= output < outputCount`.

---

## 6. State and checkpoint layout

```mermaid
flowchart TB
    subgraph State["Lfm3.State (per context)"]
        direction TB
        subgraph scratch["Scratch sized by batchCapacity"]
            A["residual · normed · branch · query · gate · attnOut"]
            B["batchK · batchV (maxKvDim) · hidden · hidden2 (ffn)"]
            C["shortConvIn 3dim · shortConvOut · ropeCos/Sin Full + Swa · logits 1 x vocab"]
            D["scores / valuesT (lazy, two-way attention) · decodeScratch · prompts · media"]
        end
        subgraph persist["Persistent per layer"]
            KVF["full layer: keyCache/valueCache F16 [contextCapacity x kvDim]"]
            KVS["SWA layer: ring F16 [min(ctx, window) x kvDim]"]
            CV["short-conv layer: shortConvState F32 [(lCache-1) x dim]"]
        end
    end
    subgraph Blob["Lfm3CheckpointCodec blob"]
        direction TB
        B1["attention layers in order: K rows then V rows for positions [from, to)<br/>full: linear; SWA: KvTransfer.ringSpan"]
        B2["then every short-conv layer's history, fixed size convElements F32"]
    end
    KVF -->|capture / restore| B1
    KVS -->|ringSpan| B1
    CV -->|KvTransfer.transfer| B2
```

- `sizeOf(positions) = positions * bytesPerPosition + convBytes`, with `bytesPerPosition = sum over attention layers of 2 * 2 bytes * kvDim` and `convBytes = sum over conv layers of (lCache-1) * dim * 4`.
- `clearHistory()` zero-fills every `shortConvState` (KV rows past the cursor are masked and harmless; the conv history would leak the previous conversation).
- `newState(ctx, batch)` makes a cross-thread arena owned by the state; `newState(ctx, batch, arena)` borrows the caller's. `State` rejects `contextCapacity > maxContextLength`.
- A decision state uses `batchCapacity == contextCapacity` because a whole prompt is one batch.

---

## 7. Vision tower

`Lfm3Vision` (projector type `lfm3vl`): `encode` over normalised patches; `positions(image) = (w/(patch*2)) * (h/(patch*2))` rows after the 2x2 merge; the image cap is 4096 tokens.

```mermaid
flowchart TB
    Img["Media.Image"] --> TS["Lfm3VisionPreprocess.targetSize<br/>smart_resize to multiple of patch*2, min/max pixels"]
    TS --> RS["Resample.bicubic (antialiased, 8-bit RGB)"]
    RS --> PT["patches: (v/255 - mean)/std, channel-row-col per patch<br/>round to F16 unless kernel is BF16"]
    PT --> PE["gemm patchKernel (first + second temporal slice summed) -> x"]
    Vid["Video frame pair"] -.->|"first through firstKernel, second through secondKernel, summed"| PE
    PE --> EN["rmsnormRowsGgml embeddingNorm"]
    EN --> L{"for each vision layer"}
    L --> N1["rmsnorm norm1"]
    N1 --> K{"convIn != null?"}
    K -->|yes| CV["in_proj B|C|x; B*x; 3x3 depthwise conv over the patch grid (zero pad); x C; out_proj"]
    K -->|no| AT["gemm q/k/v; headNormRope with 2D RoPE (pairs j &lt; headDim/4 by y, rest by x);<br/>K,V rounded to F16; FlashAttention.bidirectionalPrefill; out"]
    CV --> R1["x += mixed"]
    AT --> R1
    R1 --> N2["rmsnorm norm2 -> Towers.ffn SiLU-gated"]
    N2 --> R2["x += ffn"]
    R2 --> L
    L -->|done| LN["layerNormRows (eps 1e-6)"]
    LN --> MG["merge 2x2 patches into one row of 4*dim"]
    MG --> MLP["gemm up + bias, GeluErf, gemm down + bias -> model width rows"]
    MLP --> Sink["project(): sliced into maxChunkSize chunks to the sink"]
```

---

## 8. Audio encoder

`Lfm3Audio` (projector type `lfm3a`): 16 kHz only (other rates are refused), channels averaged to mono, whole clip encoded at once.

```mermaid
flowchart TB
    PCM["Media.Audio pcm 16 kHz"] --> Mono["mono(): average channels"]
    Mono --> Mel["LogMel (N_FFT 400, HOP 160, Slaney filterbank)<br/>(ln/ln10 + 4) / 4, rounded to F16"]
    Mel --> C0["conv0: stride-2 3x3 full conv + SiLU"]
    C0 --> C1["depthwise stride-2 3x3"]
    C1 --> P1["pointwise 1x1 gemm + bias + SiLU"]
    P1 --> C2["depthwise stride-2 3x3"]
    C2 --> P2["pointwise 1x1 gemm + bias + SiLU"]
    P2 --> FL["flatten channel x freq per time step"]
    FL --> DO["gemm downOut + bias -> x [n x dim], n = ceil^3(frames / 2)"]
    DO --> RP["RoPE.fill positions 0..n (plain, theta 1e6)"]
    RP --> L{"for each audio layer"}
    L --> N1["rmsnorm norm1 (beforeLayer hook)"]
    N1 --> K{"convIn != null?"}
    K -->|yes| SC["short conv, taps 3, zero history per clip"]
    K -->|no| AT["GQA, headNormRope, F16 rounding,<br/>slidingWindowPrefill with window_size (causal, no cache)"]
    SC --> R1["x += mixed"]
    AT --> R1
    R1 --> FF["rmsnorm norm2, Towers.ffn, x += ffn"]
    FF --> L
    L -->|done| PN["post_ln rmsnorm"]
    PN --> LN["layerNormRows eps 1e-5"]
    LN --> MLP["gemm up + bias, GeluErf, gemm down + bias -> model width rows"]
```

---

## 9. Speech head

LFM3-Omni speaks: after each backbone step the final-normed last row (`Lfm3.hiddenState`) feeds the speech head, which returns one 80 ms frame of 12 codes and 24 kHz PCM, plus a feedback row that becomes the backbone's next input as a keyed `Batch` of embeddings.

```mermaid
sequenceDiagram
    participant E as Engine (SpeechOutput)
    participant Sp as Speaker
    participant L as Lfm3
    participant St as Stream
    participant I as Istft

    E->>L: speaker(state, transcript, random)
    L-->>E: Speaker (openers audio_start / mixed_start, text_end)
    loop each step
        E->>Sp: text(token)
        Sp-->>E: Lane TEXT / CONTROL / END (interleave: 6 text slots, 12 audio frames)
        E->>Sp: frame()
        Sp->>L: hiddenState(state, stream.hidden)
        Sp->>Sp: random codebook order (Fisher-Yates)
        Sp->>St: frame(hidden, order, sampling, rng, codes)
        St->>St: proj_in once, then 3 depthformer passes with 4 codebooks revealed each
        St->>St: penalize (CTRL repetition) then sample (argmax or top-k tempered)
        alt a code == vocab-1 (end code)
            St-->>Sp: false (end frame, all codes end)
            Sp->>St: flush() codec tail
        else normal frame
            St->>St: feedback(codes) = sum of 12 code embeddings (double accumulate)
            Sp->>St: decode(codes): FSQ lookup, quant out_proj, tfm in_proj, x6 upsample
            St->>St: run causal LFM codec stack (window 30, ring 32, conv taps 7)
            St->>I: 6 frames of log-magnitude + phase -> ISTFT hops
            I-->>St: PCM samples
        end
        Sp-->>E: Frame(codes, clamped pcm, feedback row)
    end
    E->>Sp: close() closes the Stream arena
```

The depthformer stack and the codec stack are both built from the same `Block` record (conv XOR attention, optional sandwich norms) and driven by one `run(Pass, rows, start)`; the depthformer is bidirectional and stateless, the codec is causal and keeps its window and conv state in the `Stream`. `Feedback` rows are keyed by the codes (`ContentKey`) so a replayed spoken turn matches the prompt cache.

---

## 10. Chat template

```mermaid
flowchart TB
    Conv["Conversation"] --> Emp{"messages empty?"}
    Emp -->|yes| X1["UnsupportedConversation"]
    Emp -->|no| Bos["PromptWriter: id(bos)"]
    Bos --> Sys{"system text or non-deferred tools?"}
    Sys -->|yes| SY["im_start system: text, then tools block<br/>one jinja-JSON function per line in &lt;tools&gt;; im_end"]
    Sys -->|no| LU
    SY --> LU["lastUser = index of last USER message"]
    LU --> Loop{"for each message"}
    Loop --> R{"role"}
    R -->|USER / SYSTEM| WC["writeContent: text verbatim,<br/>media framed vision_start rows vision_end (cachedMedia)"]
    R -->|ASSISTANT| WA["writeAssistant: replay a spoken turn if recorded;<br/>else think reasoning kept after lastUser, dropped (empty think pair) before;<br/>text with think markers as ids; tool_call function=NAME parameter=K blocks"]
    R -->|TOOL| WT["writeToolResult: tool_response wrapper unless already present"]
    WC --> E["id(im_end), text newline, flush"]
    WA --> E
    WT --> E
    E --> Loop
    Loop -->|done| Gen["im_start assistant newline, id(think)"]
    Gen --> Th{"conversation.thinking()?"}
    Th -->|no| ET["id(endThink): closed empty think"]
    Th -->|yes| RS
    ET --> RS["ReplyState(prefix, ReplyLanguage.Spans parser)"]
```

Other contract methods: `mediaPositions` asks the projector; `replyForbidden()` bans the speech lane tokens (`audio_start`, `mixed_start`, `mixed_end`, `text_start`, `text_end`, `audio`, `audio_out`) in text replies; `constrainedReply` / `forcedCall` delegate to `ReplyLanguage.Spans` (header `\n<function=NAME`); `ttsInstruction(language, voice)` renders the GGUF `tts` system turn for parity tests.

---

## 11. Decision flow (d1)

```mermaid
flowchart TB
    Q["decide(state, input, questions)"] --> J["judged(input): media in order + joined text (null if none)"]
    J --> Ask["asked(q): Predicate (no/yes label tokens), Choice (codes A, B.. + spaced forms), Rating (digits 0..9)"]
    Ask --> Hd["head(): bos, im_start user, tower rows between vision_start / vision_end<br/>video: frame pairs stamped with seconds; picture capped at MAX_PIXELS"]
    Hd --> Grp["group questions: GROUP 16 per prompt (1 for decideIndependently)<br/>prompt = head + text + QUESTION: + body + turnEnd"]
    Grp --> Pk{"held + length &gt; min(SHARED_BATCH, batchCapacity)?"}
    Pk -->|yes| Run["answer(): flush the batch"]
    Pk -->|no| More["append prompt to the batch"]
    More --> Pk
    Run --> Rst["state.reset(); model.ingestPrompts(batch, parts, media)"]
    Rst --> Rd["per question: row = base + end - 1"]
    Rd --> Pr["probabilities(): logits(tokens) of label forms,<br/>best per option / temperature, softmax"]
    Pr --> Ans["float[] per question"]
```

Within a packed prompt the shared rows see the whole prompt and a question's rows see the shared rows plus their own (two-way full-attention layers); sliding-window and short-conv layers stay causal. Control-token text in caller input is broken with a zero-width space (`plain`) so user text cannot close the turn.
