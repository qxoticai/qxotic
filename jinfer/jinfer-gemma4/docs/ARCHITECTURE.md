# jinfer-gemma4 — class architecture

Package `com.qxotic.jinfer.models.gemma4` (15 main classes, ~4,700 lines). It is the `jinfer-<model>` provider for GGUF
architecture `gemma4`: a generative decoder with **hybrid sliding-window / full attention**, **shared-KV tail layers**,
optional **per-layer input embeddings (PLE)**, an optional **MoE branch** next to the dense FFN, a **final logit
softcap**, optional **image / audio / video** input through an `mmproj` sidecar, and optional **MTP self-speculative
decoding** through a `gemma4-assistant` sidecar.

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends;
classes marked `<<external>>` live outside this package.

1. [Position in the stack](#1-position-in-the-jinfer-stack)
2. [Class diagrams](#2-class-diagrams) — 2a contracts, 2b model internals, 2c media towers, 2d speculative decoding, 2e chat template and tool syntax
3. [Loading sequences](#3-loading-sequences)
4. [Forward pass](#4-forward-pass) — 4.0 global, 4a `ingest`, 4b `decoderBlock`, 4c MoE branch, 4d per-layer inputs
5. [State and checkpoint layout](#5-state-and-checkpoint-layout)
6. [Media projection flows](#6-media-projection-flows)
7. [Speculative decoding loop](#7-speculative-decoding-loop)
8. [Chat template encode flow](#8-chat-template-encode-flow)

| Class | Role |
|---|---|
| `Gemma4` | The model: `LanguageModel` + `Multimodal` + `SpeculativeDecoding` impl with nested `Configuration`, `Weights`, `LayerWeights`, `MoeWeights`, `State`, `Towers`; the whole forward pass, loaders, sidecar attach/pairing checks |
| `Gemma4Provider` | `ModelProvider` SPI: arch `gemma4`, companion files `media` (mmproj) and `speculation` (mtp), stop-token set, chat template, sampling defaults |
| `Gemma4ChatTemplate` | `ChatTemplate`: `<\|turn>` framing, system/think block, tool declarations/calls/responses, thought channel, structural image/audio/video, reply parser |
| `Gemma4ToolSyntax` | Byte-exact port of the GGUF template's compact tool notation (`declaration`, `call`, `response`) plus the `parseBlock` call parser |
| `Gemma4CheckpointCodec` | `CheckpointCodec<State>`: serialises own-KV layers only, ring-aware for sliding-window layers |
| `Gemma4Mtp` | MTP sidecar reader (arch `gemma4-assistant`): `Configuration`, `Weights`, shape-verified `loadSidecar` |
| `Gemma4MtpDecoder` | One draft step of the 4-layer Q-only draft transformer attending the backbone's KV rings |
| `Gemma4Speculative` | Sample-and-accept speculative loop (`generate`/`walk`) with per-state `Scratch` |
| `VisionTower` | Interface for image projectors that size images to a token budget (`atBudget` for video frames) |
| `Gemma4Vision` | SigLIP-style tower (`gemma4v`): 2D RoPE bidirectional ViT, windowed mean pooling, projection |
| `Gemma4VisionUnified` | Unified projector (`gemma4uv`): patch embedding + norms + positions + projection, no transformer layers |
| `Gemma4Conformer` | Conformer audio tower (`gemma4a`): 2 conv subsampling stages, chunked-local attention blocks |
| `Gemma4Audio` | Raw-waveform audio projector (`gemma4ua`): framed PCM, RMS norm, one projection |
| `VisionPreprocess` | Aspect-preserving target size, Pillow-style bicubic resize to CHW, token budgets (system properties) |
| `AudioPreprocess` | Mono 16 kHz conversion, Butterworth resample, Hann/mel filterbank, 30 s log-mel chunks |
| `Clamped` | Projection weight with optional QAT input/output clamps (`gemm`) |

---

## 1. Position in the jinfer stack

```mermaid
flowchart TB
    Callers["Callers: cli / server / langchain4j / spring-ai"]
    subgraph chat["jinfer-chat"]
        Models["Models (ServiceLoader)"]
        MP["ModelProvider"]
        LM_["LoadedModel"]
        CT["ChatTemplate"]
    end
    subgraph core["jinfer-core"]
        LM["LanguageModel / Multimodal / SpeculativeDecoding"]
        CS["ContextState"]
        CC["CheckpointCodec"]
        B["Batch (Tokens | Embeddings | Sequences)"]
        GEN["Generator / Sampler"]
    end
    subgraph g4["jinfer-gemma4"]
        P["Gemma4Provider"]
        G["Gemma4"]
        T["Gemma4ChatTemplate"]
        C["Gemma4CheckpointCodec"]
        Tw["Towers: vision + audio"]
        Mt["Gemma4Mtp / Gemma4MtpDecoder / Gemma4Speculative"]
    end
    subgraph kern["jinfer-kernels"]
        K["MatMul, Norms, RoPE, FlashAttention, Activations<br/>Convert, Ops, Moe, KvTransfer, ModelLoader"]
    end
    Tok["toknroll Tokenizer"]
    Side1[("mmproj GGUF")]
    Side2[("MTP sidecar GGUF")]

    Callers --> Models --> P
    P -.->|implements| MP
    P -->|produces| LM_
    LM_ --> CT
    G -.->|implements| LM
    T -.->|implements| CT
    C -.->|extends| CC
    G --> K
    G --> Tok
    G --> Tw
    G --> Mt
    Tw --> Side1
    Mt --> Side2
    GEN --> LM
    G -->|"consumes"| B
```

---

## 2. Class diagrams

### 2a. Contracts and relations

```mermaid
classDiagram
    direction LR
    class ModelProvider { <<interface, external>> }
    class LanguageModel { <<interface, external>> }
    class Multimodal { <<interface, external>> }
    class SpeculativeDecoding { <<interface, external>> }
    class ContextState { <<abstract, external>> }
    class CheckpointCodec { <<abstract, external>> }
    class ChatTemplate { <<interface, external>> }
    class MediaProjector { <<interface, external>> }
    class Gemma4Provider {
        <<final>>
        architectures() gemma4
        companionFiles() media, speculation
        loadLanguage(...) LoadedModel
    }
    class Gemma4 { <<final>> }
    class State { <<Gemma4.State>> }
    class Towers {
        <<record Gemma4.Towers>>
        vision
        audio
        frames(defaultBudget)
    }
    class Gemma4ChatTemplate { <<package-private>> }
    class Gemma4ToolSyntax { <<utility>> }
    class Gemma4CheckpointCodec { <<package-private>> }
    class Gemma4Mtp { <<package-private>> }
    class Gemma4MtpDecoder { <<package-private>> }
    class Gemma4Speculative { <<utility>> }

    ModelProvider <|.. Gemma4Provider
    LanguageModel <|.. Gemma4
    Multimodal <|.. Gemma4
    Multimodal <|.. Towers
    SpeculativeDecoding <|.. Gemma4
    ContextState <|-- State
    CheckpointCodec <|-- Gemma4CheckpointCodec
    ChatTemplate <|.. Gemma4ChatTemplate
    Gemma4Provider ..> Gemma4 : loadModel, withMedia, attachMtp
    Gemma4Provider ..> Gemma4ChatTemplate : new
    Gemma4 *-- Towers
    Gemma4 o-- Gemma4Mtp : null unless attached
    Gemma4 ..> State : newState
    Gemma4 ..> Gemma4CheckpointCodec : checkpointCodec()
    Gemma4 ..> Gemma4MtpDecoder : mtpDecoder(allocator)
    Gemma4 ..> Gemma4Speculative : speculate
    Gemma4CheckpointCodec ..> State : reads/writes
    Gemma4ChatTemplate --> Gemma4ToolSyntax
    Gemma4ChatTemplate --> Towers : media
    Towers --> MediaProjector
```

### 2b. Model internals — `Gemma4` and its nested types

```mermaid
classDiagram
    direction TB
    class Gemma4 {
        <<final>>
        -Configuration configuration
        -Tokenizer tokenizer
        -Weights weights
        -Towers towers
        -Gemma4Mtp mtp
        +loadModel(path or channel, arena)$
        +loadWithMtp(text, mtp, arena)$
        +loadTowers(mmproj, width, arena)$
        +withMedia(mmproj, arena) Gemma4
        +attachMtp(sidecar, arena) Gemma4
        +ingest(State, Batch)
        +logits(State, output) MemoryView
        ~logitsAll(State, dst)
        -forward(State, ids...)
        -forwardEmbeddings(State, rows...)
        -forwardLayers()
        -layer() attention() feedForward()
        -mergePerLayerInput() commitKv()
        -moeFeedForward() denseMlp()
        -buildPerLayerInputs() fillRotaryTables()
    }
    class Configuration {
        <<record>>
        embeddingLength, numberOfLayers
        feedForwardLength[] per layer
        numberOfHeads
        numberOfKeyValueHeadsPerLayer[]
        headSizeFull, headSizeSwa
        slidingWindow power of 2
        isSwa[] ownKvLayers
        ropeThetaFull ropeThetaSwa
        logitSoftcapping
        embeddingLengthPerLayer
        expertCount expertUsedCount
        expertFeedForwardLength
        hasKv() kvSourceLayer() kvDim()
        kvCachePositions() kvCacheIndex()
    }
    class Weights {
        <<record>>
        tokenEmbeddings
        LayerWeights[] layers
        finalNorm
        ropeFull ropeSwa Schedule
        classifier tied to tokenEmbeddings
        perLayerTokenEmbeddings
        perLayerModelProjection
        perLayerProjectionNorm
    }
    class LayerWeights {
        <<record>>
        attnNorm wq wk wv wo
        qNorm kNorm postAttnNorm
        ffnNorm gate up down postFfnNorm
        outputScale float
        inputGate projection postProjectionNorm
        MoeWeights moe nullable
    }
    class MoeWeights {
        <<record>>
        router routerScale
        gateUp[] down[] per expert
        downScale
        postNorm1 preNorm2 postNorm2
    }
    class State {
        <<static final>>
        residual normed branchOut attnOut query
        hidden hidden2 logits
        ropeCos/Sin Full and Swa
        keyCache[] valueCache[] F16, ownKv only
        batchK[] batchV[] F32, ownKv only
        perLayerInputs perLayerTokenRows plGate plProjection
        moe* scratch + Moe.Routing
        decodeScratch
        specScratch lastTokens
        clearHistory() no-op
    }
    Gemma4 *-- Configuration
    Gemma4 *-- Weights
    Weights *-- LayerWeights
    LayerWeights o-- MoeWeights
    Gemma4 ..> State : newState
    Configuration ..|> ContextConfiguration
```

### 2c. Media towers

```mermaid
classDiagram
    direction TB
    class MediaProjector { <<interface, external>> }
    class VisionTower {
        <<interface>>
        positions(image, budget)
        project(image, budget, maxChunk, sink)
        geometry()
        atBudget(budget) MediaProjector
    }
    class Gemma4Vision {
        <<final, gemma4v>>
        patchSize visionDim headCount merge
        patchEmbedding positionEmbedding
        stdBias stdScale optional pair
        Layer[] layers
        inputProjection Clamped
    }
    class Gemma4VisionUnified {
        <<final, gemma4uv>>
        patchEmbedding patchBias
        positionEmbedding inputProjection
        patch_norm 1..3 LayerNorm
    }
    class Gemma4Conformer {
        <<final, gemma4a>>
        CHUNK 12 PAST 12 RPE 13
        conv0 128ch conv1 32ch
        Block[] blocks
        outputProjection modelProjection
    }
    class Gemma4Audio {
        <<final, gemma4ua>>
        modelDim frameSize
        inputProjection
    }
    class Clamped {
        <<record>>
        weight
        inputMin inputMax outputMin outputMax
        gemm(...)
    }
    class VisionPreprocess {
        <<utility>>
        IMAGE_TOKEN_BUDGET 280
        videoBudget()
        targetSize() toCHW() im2col()
    }
    class AudioPreprocess {
        SAMPLE_RATE 16000
        N_FFT 512 HOP 160
        toMono16k() logMel()
        MelChunk record
    }
    MediaProjector <|-- VisionTower
    VisionTower <|.. Gemma4Vision
    VisionTower <|.. Gemma4VisionUnified
    MediaProjector <|.. Gemma4Conformer
    MediaProjector <|.. Gemma4Audio
    Gemma4Vision *-- Clamped
    Gemma4Conformer *-- Clamped
    Gemma4Vision ..> VisionPreprocess
    Gemma4VisionUnified ..> VisionPreprocess
    Gemma4Conformer *-- AudioPreprocess
    Gemma4Audio ..> AudioPreprocess
```

### 2d. Speculative decoding (MTP)

```mermaid
classDiagram
    direction LR
    class SpeculativeDecoding { <<interface, external>> }
    class Gemma4 {
        <<final>>
        speculationReady() mtp != null
        speculate(state, sampler, constraints, depth, listener, audit)
        mtpDecoder(allocator)
        logitsAll(state, dst)
    }
    class Gemma4Mtp {
        <<final>>
        loadSidecar(path, backboneVocab, arena)$
        configuration() weights()
    }
    class MtpConfiguration {
        <<record Gemma4Mtp.Configuration>>
        embeddingLength backboneDim
        numberOfLayers numberOfHeads
        headSizeFull headSizeSWA
        isSWA[] slidingWindow
    }
    class MtpWeights {
        <<Gemma4Mtp.Weights>>
        tokenEmbeddings tied
        preProjection postProjection
        outputNorm
        wq wo ffnGate ffnUp ffnDown per layer
        layerOutputScales ropeFreqFactors
    }
    class Gemma4MtpDecoder {
        <<final>>
        draft(state, hidden, off, token, position)
        chainedHidden()
    }
    class Gemma4Speculative {
        <<utility>>
        generate(model, state, budget, timeout, stops, depth, sampler, listener, audit)
        walk(...)
    }
    class Scratch {
        <<Gemma4Speculative.Scratch>>
        decoder vlogits row h depth
    }
    SpeculativeDecoding <|.. Gemma4
    Gemma4 o-- Gemma4Mtp
    Gemma4Mtp *-- MtpConfiguration
    Gemma4Mtp *-- MtpWeights
    Gemma4 ..> Gemma4MtpDecoder : mints per generation
    Gemma4MtpDecoder --> Gemma4Mtp : weights
    Gemma4MtpDecoder --> Gemma4 : backbone KV rings, token embeddings
    Gemma4 ..> Gemma4Speculative : speculate
    Gemma4Speculative *-- Scratch : stored in State.specScratch
    Scratch --> Gemma4MtpDecoder
```

### 2e. Chat template and tool syntax

```mermaid
classDiagram
    direction LR
    class ChatTemplate { <<interface, external>> }
    class Gemma4ChatTemplate {
        <<final>>
        -promptStart bos
        -turnOpen turnClose thinkSeed
        -imageOpen imageClose audioOpen audioClose
        -noThinkingPrefix thoughtTail
        -scaffoldsNonThinking
        encode(conversation, batchCapacity, mediaCache, sink) ReplyState
        parser(tokenizer) ReplyParser
        constrainedReply(grammar, calls)
        forcedCall(tools)
        thinkMarkers() channel
        mediaPositions(media) int
    }
    class Gemma4ToolSyntax {
        <<utility>>
        declaration(tool, sink)
        call(name, args, sink)
        response(name, text, sink)
        parseBlock(payload) ToolCall list
    }
    class Sink {
        <<interface>>
        text(s)
        quote()
    }
    class PromptWriter { <<external>> }
    class ReplyLanguage { <<external>> }
    ChatTemplate <|.. Gemma4ChatTemplate
    Gemma4ChatTemplate ..> PromptWriter : id / text / media
    Gemma4ChatTemplate --> Gemma4ToolSyntax
    Gemma4ToolSyntax --> Sink
    Gemma4ChatTemplate ..> ReplyLanguage : Spans with parseBlock
```

---

## 3. Loading sequences

The provider loads the backbone first, then attaches the optional sidecars by **copying** the model (`withMedia`, `attachMtp` commute, and a model never changes under a state minted from it). Both sidecars are pairing-checked from headers before any tensor is mapped (media) or by dimension checks (MTP).

```mermaid
sequenceDiagram
    participant M as Models (jinfer-chat)
    participant P as Gemma4Provider
    participant G as Gemma4
    participant L as ModelLoader
    participant T as Towers
    participant X as Gemma4Mtp

    M->>P: loadLanguage(channel, gguf, path, arena, companions, tokenizer)
    P->>G: loadModel(channel, gguf, arena, tokenizer)
    G->>G: tokenizer from GGUF if null
    G->>G: loadConfiguration(gguf, tokenizer)
    Note over G: isSwa from sliding_window_pattern, else q_norm width, else l mod 5 != 4<br/>kvHeads from attn_k shape, ownKv = layers - shared_kv_layers<br/>shared layer must have the same kvDim as its source
    G->>L: loadTensors(channel, gguf, arena)
    G->>G: loadWeights(tensors, c): rope schedules, PLE, MoE, output scale
    G-->>P: Gemma4 (Towers.NONE, mtp null)
    opt companion media
        P->>G: withMedia(mmproj, arena)
        G->>T: loadTowers(mmproj, embeddingLength, arena)
        T->>T: validateMediaPairing: projector types, projection_dim == model width
        T->>L: loadTensors(mmproj)
        alt vision type gemma4v or gemma4uv
            T->>T: Gemma4Vision / Gemma4VisionUnified.loadModel
        else audio type gemma4a or gemma4ua
            T->>T: Gemma4Conformer / Gemma4Audio.loadModel
        else other type
            T--xG: IllegalArgumentException unsupported projector
        end
        G-->>P: new Gemma4 sharing weights, towers attached
    end
    opt companion speculation
        P->>G: attachMtp(mtp, arena)
        G->>X: loadSidecar(path, vocab, arena) arch must be gemma4-assistant
        G->>G: check backboneDim == embeddingLength
        G->>G: check draft head sizes == KV head sizes at ownKv-2 and ownKv-1
        G-->>P: new Gemma4 with mtp
    end
    P->>P: stops: turn close, end_of_turn, eos, endoftext, tool_response (first present = endTurn)
    P->>P: scaffoldsNonThinking = chat_template contains "not enable_thinking"
    P-->>M: LoadedModel(model, tokenizer, source, stops, seed, Gemma4ChatTemplate, sampling 1.0 / 0.95 / 64)
```

---

## 4. Forward pass

`ingest(state, batch)` runs `state.exclusively(forward)`. Unlike a plain dense decoder, a Gemma 4 layer stack has **three kinds of layers**: sliding-window own-KV, full-attention own-KV, and **KV-shared tail layers** (index `>= ownKvLayers`) that compute only Q and attend the K/V rings of the last own-KV layer of their kind (`ownKv-2` for SWA, `ownKv-1` for full). Every layer is attention, then FFN (dense, plus MoE when present), then the optional per-layer input merge, then a scalar output scale.

### 4.0 Global architecture

```mermaid
flowchart TB
    In["Batch: token ids or media embedding rows"] --> Emb["Embedding: gather x sqrt(dim)<br/>or copy of projected rows"]
    Emb --> Rope["RoPE tables: full theta + SWA theta"]
    Emb --> PLE["Per-layer inputs (PLE)<br/>projection + token table, optional"]
    subgraph Stack["Layer stack"]
        direction TB
        subgraph Own["Own-KV layers 0..ownKv-1"]
            direction LR
            OA["Attention: own K/V<br/>SWA ring or full"] --> OF["FFN dense (+ MoE)"] --> OP["PLE merge, output scale"]
        end
        subgraph Shared["Shared-KV tail layers ownKv..N-1"]
            direction LR
            SA["Attention: Q only<br/>reads source layer ring"] --> SF["FFN dense (+ MoE)"] --> SP["PLE merge, output scale"]
        end
        Own --> Shared
    end
    Rope --> Stack
    PLE --> Stack
    Emb --> Stack
    Stack --> FN["Final RMSNorm (finalNorm)"]
    FN --> Head["Classifier GEMV (tied embeddings if no output.weight)"]
    Head --> Cap["Softcap logits"]
    Cap --> Out["logits [vocab]"]
    Own -.->|"commitKv F32 to F16"| KV[("State: keyCache / valueCache per own layer")]
    KV -.->|"read by shared layers and MTP draft"| Shared
    FN -.->|"pre-norm residual row = MTP seed"| Mtp["Gemma4MtpDecoder"]
```

### 4a. `ingest` — batch validation, input kinds, rope/embedding, layer loop, advance

```mermaid
flowchart TB
    In["ingest(state, Batch)"] --> Ex["state.exclusively"]
    Ex --> V{"0 < count <= batchCapacity<br/>position + count <= contextCapacity"}
    V -->|no| Err["IllegalArgumentException"]
    V -->|yes| Lo["lastOnly = outputs == LAST"]
    Lo --> Sw{"batch.input()"}
    Sw -->|Tokens| T["state.lastTokens = ids<br/>check each id in [0, vocab)"]
    Sw -->|Embeddings| E["state.lastTokens = null<br/>towers == NONE ? Unsupported<br/>row width == embeddingLength<br/>pad id from vocabulary, default 0"]
    Sw -->|Sequences| S["UnsupportedOperationException<br/>generative, no packed sequences"]
    T --> R1["fillRotaryTables(startPos, n)<br/>RoPE.fill full (headSizeFull/2) and SWA (headSizeSwa/2)"]
    E --> R2["fillRotaryTables"]
    R1 --> Em1["embed: Convert.gatherToF32 then multiply by sqrt(dim)"]
    R2 --> Em2["Convert.copyF32 rows to residual<br/>bidirectional flag from the batch"]
    Em1 --> Pl1["buildPerLayerInputs(tokens)"]
    Em2 --> Pl2["buildPerLayerInputs(pad token for every row)"]
    Pl1 --> FL["forwardLayers"]
    Pl2 --> FL
    FL --> Q{"lastOnly and seqLen > 1 ?"}
    Q -->|no| A1["every = numberOfLayers"]
    Q -->|yes| A2["every = ownKvLayers"]
    A1 --> Loop["for layer in 0..every: layer(state, l)  see 4b"]
    A2 --> Loop
    Loop --> Cm["commitKv for all own-KV layers"]
    Cm --> Z{"every == numberOfLayers ?"}
    Z -->|yes| Adv
    Z -->|no| Mv["move last row to row 0:<br/>residual, perLayerInputs, batchK/V of own layers<br/>refill rope for position startPos+last, 1 row"]
    Mv --> Tail["for layer in ownKv..N-1: layer(state, l, seqLen = 1)"]
    Tail --> Back["move residual row 0 back to last row"]
    Back --> Adv["state.advance(batch): advanceContext(count, outputs)"]
    Adv -.->|"later"| Lg["logits(state, output) / logitsAll"]
```

Pruned prefill: shared-KV layers write no cache, so when only the last row is read the earlier rows are dead after the own-KV layers; just that row continues through the tail as a decode step.

### 4b. `decoderBlock` — one layer

Gemma4 has no single `decoderBlock` method; a block is `layer(state, l, ...)` = `attention` + `feedForward` + `mergePerLayerInput` + output scale. Residual additions all use the "post-norm the branch, then add" idiom `normThenAddRows`.

```mermaid
flowchart TB
    R0["residual [rows x dim]"] --> N1["Norms.rmsnormRows to normed (attnNorm)"]
    subgraph Attn["attention(l)"]
        direction TB
        N1 --> Q["gemm wq to query"]
        N1 --> HK{"hasKv(l): l below ownKvLayers ?"}
        HK -->|yes| K["gemm wk to batchK[l]"]
        HK -->|yes| V{"wv present ?"}
        V -->|yes| V1["gemm wv to batchV[l]"]
        V -->|no| V2["copy batchK to batchV (K = V layers)"]
        Q --> QN["headNormRope: per-head RMSNorm (qNorm) + NeoX RoPE<br/>cos/sin SWA or full by isSwa[l]"]
        K --> KN["headNormRope: per-head RMSNorm (kNorm) + NeoX RoPE"]
        V1 --> VN["per-head weightless RMSNorm on V"]
        V2 --> VN
        HK -->|no| Src["kvLayer = kvSourceLayer(l)<br/>ownKv-2 if SWA else ownKv-1"]
        KN --> Fa
        VN --> Fa
        Src --> Fa
        QN --> Fa{"seqLen > 1 ?"}
        Fa -->|yes| P1["FlashAttention.slidingWindowPrefill<br/>window = slidingWindow (SWA) or 0 (full)<br/>bidirectional flag, scale 1.0<br/>K/V = cache[kvLayer] + batch[kvLayer]"]
        Fa -->|no| P2["FlashAttention.flashDecode<br/>start = max(0, pos - window + 1) for SWA<br/>reads cache[kvLayer] + batch[kvLayer]"]
        P1 --> AO["attnOut"]
        P2 --> AO
        AO --> WO["gemm wo to branchOut"]
    end
    WO --> A1["normThenAddRows: residual += rmsnorm(branchOut, postAttnNorm)"]
    A1 --> FF{"layer has MoE ?"}
    subgraph Ffn["feedForward(l)"]
        direction TB
        FF -->|no| D["denseMlp: rmsnorm ffnNorm, gate and up gemm,<br/>geluMultiply, down gemm"]
        D --> A2["normThenAddRows: residual += rmsnorm(out, postFfnNorm)"]
        FF -->|yes| MO["moeFeedForward  see 4c"]
    end
    A2 --> PL["mergePerLayerInput (4d)"]
    MO --> PL
    PL --> Sc{"outputScale != 1 ?"}
    Sc -->|yes| Mul["residual *= outputScale"]
    Sc -->|no| Done["next layer"]
    Mul --> Done
    K -.->|"commitKv after the loop (own layers only):<br/>f32ToF16 into ring slot pos & (W-1) for SWA, absolute pos for full"| KVc[("keyCache / valueCache")]
    V1 -.-> KVc
```

Details worth knowing:

- `commitKv` runs **after all processed layers**: each own layer has its own `batchK[l]/batchV[l]`, and the commit is one job per layer so a batch longer than the window writes ring slots in order.
- Rows of the shared scratch (`query`, `attnOut`, `hidden`) are addressed at the view's own stride, since SWA layers pack narrower rows than the max width.
- GQA via `kvMul = numberOfHeads / numberOfKeyValueHeadsPerLayer[l]`; head size and `kvDim` differ between SWA and full layers.
- Attention scale is 1.0 (QK-norm replaces `1/sqrt(d)`).

### 4c. MoE branch (`moeFeedForward`)

The dense MLP is **always** computed; on MoE layers its output is a *shared* branch and a routed expert branch is added to it.

```mermaid
flowchart TB
    R["residual"] --> Dm["denseMlp to moeShared"]
    Dm --> N1["rmsnormRows moeShared (postNorm1)"]
    R --> Rms["per row: rms = 1/sqrt(mean(x^2)+eps)"]
    Rms --> In["moeInput = residual * preNorm2 * rms"]
    Rms --> RI["moeRouterInput = residual * routerScale * rms / sqrt(dim)"]
    RI --> Rt["gemm router to moeRouter [rows x experts]"]
    Rt --> Tk["Moe.softmaxSelectTopK (expertUsedCount)<br/>rowTopE, rowTopP, expertCounts"]
    Tk --> Dp["Moe.dispatch(routing, ..., downScale, per-expert callback)"]
    In --> Dp
    subgraph Exp["per expert e with n gathered rows"]
        direction TB
        G1["gemm gateUp[e] to moeHidden (gate | up)"] --> G2["geluMultiply gate with up"] --> G3["gemm down[e] to out"]
    end
    Dp --> Exp
    Exp --> Mo["moeOut (weighted sum by routing probabilities)"]
    Mo --> A1["normThenAddRows: moeShared += rmsnorm(moeOut, postNorm2)"]
    N1 --> A1
    A1 --> A2["normThenAddRows: residual += rmsnorm(moeShared, postFfnNorm)"]
```

### 4d. Per-layer inputs (PLE)

Active only when `embeddingLengthPerLayer > 0` and `per_layer_token_embd.weight` exists; a layer merges its slice only if it has `inputGate`.

```mermaid
flowchart TB
    subgraph Build["buildPerLayerInputs (once per batch, before the layers)"]
        direction TB
        B1["gemm perLayerModelProjection: residual to perLayerInputs [rows x plDim*N]"] --> B2["x 1/sqrt(dim)"]
        B2 --> B3["per layer slice: RMSNorm (perLayerProjectionNorm)"]
        T1["gather perLayerTokenEmbeddings[token] to perLayerTokenRows<br/>(one scratch row per thread)"] --> T2["x sqrt(plDim)"]
        B3 --> Sum["perLayerInputs += token rows"]
        T2 --> Sum
        Sum --> Sc["x 1/sqrt(2)"]
    end
    subgraph Merge["mergePerLayerInput(l) (after the FFN)"]
        direction TB
        M1["gemm inputGate: residual to plGate [rows x plDim]"] --> M2["geluMultiply(plGate, perLayerInputs slice l)"]
        M2 --> M3["gemm projection to plProjection [rows x dim]"]
        M3 --> M4["normThenAddRows: residual += rmsnorm(plProjection, postProjectionNorm)"]
    end
    Build --> Merge
```

Media rows: PLE is built with the `<pad>` token id for every row, so media positions get the pad's per-layer embedding.

---

## 5. State and checkpoint layout

`State extends ContextState`; capacities come from `newState(contextCapacity, batchCapacity[, arena])` (a context above `maxContextLength` is refused). The cross-thread arena is owned only by the 2-argument form. There is **no recurrent state**: the only thing carried across calls is the KV rings of the own-KV layers.

```mermaid
flowchart LR
    subgraph Scratch["Per-batch scratch (batchCapacity rows, F32)"]
        direction TB
        S1["residual, normed, branchOut: dim"]
        S2["query, attnOut: maxQueryDim"]
        S3["hidden, hidden2: maxHiddenDim"]
        S4["ropeCos/Sin Full and Swa"]
        S5["perLayerInputs, plGate, plProjection, perLayerTokenRows (if PLE)"]
        S6["moeShared, moeInput, moeRouterInput, moeRouter, moeOut, moeGather,<br/>moeExpertOut, moeHidden (2*expertFf), Moe.Routing (if MoE)"]
        S7["logits [1 x vocab], decodeScratch"]
    end
    subgraph KVs["KV rings, own-KV layers only, F16"]
        direction TB
        K1["SWA layer: min(contextCapacity, slidingWindow) positions<br/>slot = position AND (window-1)"]
        K2["Full layer: contextCapacity positions<br/>slot = position"]
        K3["batchK[l], batchV[l]: batchCapacity x kvDim(l), F32 staging"]
    end
    subgraph Spec["Speculation seam"]
        direction TB
        P1["lastTokens: row to token map of the last token batch (null after media)"]
        P2["specScratch: Gemma4Speculative.Scratch, lazily built in the state arena"]
    end
    Scratch --- KVs --- Spec
```

```mermaid
flowchart TB
    CC["Gemma4CheckpointCodec(config)"] --> BPP["bytesPerPosition = sum over own-KV layers of 2 x 2 bytes x kvDim(l)"]
    BPP --> Sz["sizeOf(positions) = positions x bytesPerPosition"]
    CC --> Tr["transfer(state, from, to, blob, capture)"]
    Tr --> Lp{"for each own-KV layer"}
    Lp -->|"isSwa"| R["KvTransfer.ringSpan for key then value<br/>rows go THROUGH ring slots pos AND (W-1)"]
    Lp -->|"full"| F["KvTransfer.transfer key then value<br/>absolute element offset from x kvDim"]
    R --> Lp
    F --> Lp
    Lp --> Note["Shared-KV tail layers serialise nothing.<br/>RoPE is baked into K at absolute positions,<br/>so every block is a resume point: no residue."]
```

Blob layout: for each position range, layers appear in order `0..ownKv-1`, each as `[key span][value span]`, F16.

---

## 6. Media projection flows

`Gemma4.Towers` holds an optional image projector and an optional audio projector; video is read as frames through the vision tower at the video budget (`-Djinfer.gemma4.videoTokenBudget`, template default 70 tokens per frame; still images default 280, `-Djinfer.gemma4.imageTokenBudget`). Projected rows enter the model as `Batch.Input.Embeddings` (see 4a).

### 6a. Vision towers

```mermaid
flowchart TB
    Img["Media.Image + token budget"] --> TS["VisionPreprocess.targetSize<br/>aspect-preserving, multiple of patch (x merge for gemma4v)"]
    TS --> Im["im2col: bicubic resize to CHW, patch vectors [count x 3*patch^2]"]
    Im --> Kind{"projector_type"}
    subgraph V["gemma4v (Gemma4Vision)"]
        direction TB
        V1["gemm patchEmbedding to current [count x visionDim]"] --> V2["addGridPositions (x and y position table)"]
        V2 --> VL{"for each Layer"}
        VL --> V3["rmsnorm norm1, Clamped q/k/v gemm,<br/>per-head q/k RMSNorm, weightless V norm,<br/>2D RoPE (first half by column, second by row)"]
        V3 --> V4["FlashAttention.bidirectionalPrefill, Clamped output gemm"]
        V4 --> V5["normThenAddRows (attentionPostNorm)"]
        V5 --> V6["rmsnorm norm2, Clamped gate/up, geluMultiply, Clamped down,<br/>normThenAddRows (ffnPostNorm)"]
        V6 --> VL
        VL -->|done| V7["windowedMeanPool merge x merge, x sqrt(visionDim),<br/>optional (x - stdBias) * stdScale, weightless RMSNorm"]
        V7 --> V8["Clamped inputProjection to modelDim rows"]
    end
    subgraph U["gemma4uv (Gemma4VisionUnified)"]
        direction TB
        U1["LayerNorm patch_norm.1 on patch vectors"] --> U2["gemm patchEmbedding + patchBias"]
        U2 --> U3["LayerNorm patch_norm.2"] --> U4["addGridPositions"]
        U4 --> U5["LayerNorm patch_norm.3"] --> U6["weightless RMSNorm"]
        U6 --> U7["gemm inputProjection to modelDim rows"]
    end
    Kind -->|gemma4v| V
    Kind -->|gemma4uv| U
    V8 --> Sink["sink.accept(rows) in a cross-thread scratch arena, closed after"]
    U7 --> Sink
```

### 6b. Audio towers

```mermaid
flowchart TB
    Au["Media.Audio"] --> Kind{"projector_type"}
    subgraph UA["gemma4ua (Gemma4Audio)"]
        direction TB
        A1["AudioPreprocess.toMono16k: channel mean, Butterworth resample"] --> A2["frames of frameSize samples (frameSize = projection width / modelDim)"]
        A2 --> A3["weightless RMSNorm per frame"] --> A4["gemm inputProjection to modelDim"]
        A4 --> A5["sink per chunk of at most maxChunkSize rows"]
    end
    subgraph CF["gemma4a (Gemma4Conformer)"]
        direction TB
        C1["AudioPreprocess.logMel: 30 s chunks, 512 FFT, hop 160, mel filterbank"] --> C2["conv2d stride 2 (128 ch) + channel LayerNorm + ReLU"]
        C2 --> C3["conv2d stride 2 (32 ch) + channel LayerNorm + ReLU"]
        C3 --> C4["channel-last flatten, Clamped inputProjection"]
        C4 --> CB{"for each Block"}
        CB --> H1["half FFN: rmsnorm, up, SiLU, down, post norm, x + 0.5 * out"]
        H1 --> AT["chunked local attention: chunk 12, past 12, relative position,<br/>per-dim query scale, logit softcap 50, o-proj, post norm, residual"]
        AT --> CV["conv module: norm, pointwise, GLU, causal depthwise k=5, norm, SiLU, pointwise, residual"]
        CV --> H2["second half FFN (x + 0.5 * out)"]
        H2 --> ON["output rmsnorm"]
        ON --> CB
        CB -->|done| C5["outputProjection + bias, weightless RMSNorm, optional softEmbNorm,<br/>modelProjection to model width"]
        C5 --> C6["sink chunks of at most maxChunkSize rows"]
    end
    Kind -->|gemma4ua| UA
    Kind -->|gemma4a| CF
```

---

## 7. Speculative decoding loop

`speculate(...)` is a self-speculative loop over the MTP draft head (depth 1..8, clipped to `batchCapacity - 1`). The first draft step of each iteration is a *warm-up* over the last committed token and its hidden; the chain then drafts greedily. The verify pass is one `Batch.score` of `drafts+1` candidates, and logits come from a single `logitsAll` head GEMM. A draft survives only if it equals the **sampled** target token at its row.

```mermaid
flowchart TB
    Start["generate(model, state, budget, timeout, stops, depth, sampler, listener, audit)"] --> Chk{"speculationReady and depth in 1..8"}
    Chk -->|no| Err["IllegalStateException / IllegalArgumentException"]
    Chk -->|yes| Sc["Scratch: reuse state.specScratch or allocate in state arena"]
    Sc --> Seed["seed from last ingested row:<br/>tLast = lastTokens[lastRow]<br/>h = residual row (pre-final-norm hidden)<br/>next = sampler.sampleToken(logits of last output)"]
    Seed --> Lp{"emitted < maxTokens, position < context, before deadline"}
    Lp -->|no| Fin["result: LENGTH or TIMEOUT"]
    Lp -->|yes| Dr["drafts = min(depth, context left - 1, tokens left - 1)"]
    Dr --> W["decoder.draft(state, h, tLast, pos): warm-up"]
    W --> Ch["cand[0] = next; chain: draft(chainedHidden, dTok) then argmax, cand[i]"]
    Ch --> Ver["model.ingest(Batch.score(cand)) : ONE backbone forward"]
    Ver --> LA["logitsAll to vlogits [drafts+1 x vocab]"]
    LA --> Acc{"row r: target = sampler.sampleToken(row r)"}
    Acc -->|"cand[r+1] == target"| Up["accepted++"]
    Up --> Acc
    Acc -->|"mismatch or all accepted"| NA["nextAfter = target (or sample the last row)"]
    NA --> Sd["new seed BEFORE rollback: h = residual row accepted, tLast = cand[accepted]"]
    Sd --> Stp{"stop token within cand[0..accepted] ?"}
    Stp --> Rb["state.resumeAt(basePos + keep): drop rejected KV tail"]
    Rb --> Em["emit tokens to listener / audit; abort, stop and length checks"]
    Em --> Lp
```

The MTP draft step (`Gemma4MtpDecoder.draft`):

```mermaid
flowchart TB
    In["token, backbone hidden, position"] --> R["RoPE.fill SWA and full, one row"]
    In --> X["xh = concat( backbone.tokenEmbeddings[token] x sqrt(backboneDim), hidden )"]
    X --> Pre["gemv preProjection: 2*backboneDim to draft dim"]
    Pre --> L{"for each draft layer (4)"}
    L --> N["rmsnorm attnNorm, gemv wq, per-head q norm, NeoX RoPE"]
    N --> At["flashDecode, no own KV:<br/>SWA layers read backbone ring ownKv-2, full layer reads ownKv-1<br/>start = max(0, pos - window + 1) for SWA"]
    At --> O["gemv wo, post norm, residual"]
    O --> F["rmsnorm, gate/up gemv, geluMultiply, down gemv, post norm, residual, x layerOutputScale"]
    F --> L
    L -->|done| On["output norm"]
    On --> Lg["draftLogits = gemv tied tokenEmbeddings (no softcap)"]
    On --> Nx["hNext = gemv postProjection (chainedHidden for depth above 1)"]
```

---

## 8. Chat template encode flow

`encode(conversation, batchCapacity, mediaCache, sink)` writes through a `PromptWriter`: text segments are plain-encoded, control tokens are trusted ids, media becomes `Embeddings` batches from the projector (cached per content key).

```mermaid
flowchart TB
    S["requireSupported(conversation)"] --> B["out.ids(promptStart = bos)"]
    B --> Sy{"system first, or tools, or thinking ?"}
    Sy -->|yes| SB["systemBlock: turn system, think seed (if thinking),<br/>trimmed system text, one tool declaration block per tool<br/>(Gemma4ToolSyntax.declaration)"]
    Sy -->|no| Lp
    SB --> Lp{"for each non-tool message"}
    Lp --> Kd{"plain turn ?<br/>no calls, no continuation, no reasoning"}
    Kd -->|yes| WT["writeTurn: turn open, role (assistant = model), content parts, turn close"]
    WT --> Lp
    Kd -->|no| Mt["model turn (merged with a following assistant message)"]
    Mt --> Th["reasoning kept only after the last user message:<br/>channel open, thought, reasoning, channel close"]
    Th --> Cl["tool calls: call open, Gemma4ToolSyntax.call, call close"]
    Cl --> Rs["fold consecutive tool results: response open, Gemma4ToolSyntax.response, response close<br/>turn left open awaiting results"]
    Rs --> Lp
    WT -.-> Md["media parts: image block / audio block / video = timestamped image frames"]
    Lp -->|done| Tail{"conversation ends how ?"}
    Tail -->|"closed turn"| Open["reopen turn model"]
    Tail -->|"open tool turn"| Resume["thinking on: open channel (thought tail)<br/>thinking off: nothing"]
    Open --> Pre["thinking off and scaffold: noThinkingPrefix<br/>else thinking on: model opens the channel itself"]
    Pre --> Rep["ReplyState: reply parser = ReplyLanguage.Spans(channel, tool call, ends)"]
    Resume --> Rep
```

Reply parsing: `parser(tokenizer)` returns the `Spans` parser (reasoning in `<|channel>thought ... <channel|>`, calls in `<|tool_call> ... <tool_call|>` decoded by `Gemma4ToolSyntax.parseBlock`, ends at `<turn|>`, `<end_of_turn>`, `<|endoftext|>` or `<eos>`); `constrainedReply` and `forcedCall` build grammar-constrained and forced-call selections from the same spans.
