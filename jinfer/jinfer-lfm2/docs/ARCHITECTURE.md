# jinfer-lfm2 — class architecture

Package `com.qxotic.jinfer.models.lfm2` (8 classes, ~3860 lines). It is the `jinfer-<model>` provider for the GGUF
architectures `lfm2` and `lfm2moe` (Liquid LFM2 / LFM2.5). One backbone serves **four faces**: generative chat (dense or
MoE), a bidirectional sentence **embedder** (LFM2.5-Embedding, CLS pooling), a **ColBERT** late-interaction reranker
(LFM2.5-ColBERT, per-token `dense_2` vectors), and an optional **LFM2-VL** vision sidecar (`mmproj`). The backbone is a
**hybrid**: each layer is either GQA attention (`kvHeads > 0`) or a gated short-convolution mixer (`kvHeads == 0`), and
its FFN is either dense SwiGLU or top-k MoE.

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends;
classes marked `<<external>>` live outside this package.

1. [Position in the stack](#1-position-in-the-jinfer-stack)
2. [Class diagrams](#2-class-diagrams) — 2a contracts, 2b model internals, 2c vision, retrieval and chat template
3. [Loading sequences](#3-loading-sequences) — 3a provider flows, 3b configuration and weights
4. [Forward pass](#4-forward-pass) — 4.0 global, 4a `ingest`, 4b `decoderBlock` (4b.1 short-conv, 4b.2 MoE, 4b.3 non-causal)
5. [Heads: logits, CLS embedding, ColBERT row](#5-heads)
6. [State and checkpoint layout](#6-state-and-checkpoint-layout)
7. [Chat template and tool codec](#7-chat-template-and-tool-codec)
8. [Vision sidecar](#8-vision-sidecar)
9. [ColBERT scoring](#9-colbert-scoring)

| Class | Role |
|---|---|
| `Lfm2` | The model: `LanguageModel` + `EmbeddingModel` + `Multimodal` impl with nested `Configuration`, `Weights` (+ `AttentionWeights`, `ShortConvWeights`, `DenseFfnWeights`, `MoeFfnWeights`, `LayerWeights`), `State`, `EmbedScratch`; the whole forward pass (causal, tail-only, segmented non-causal, image rows) and the heads |
| `Lfm2Provider` | `ModelProvider` SPI: `architectures` = `lfm2`/`lfm2moe`, companion `media` = `mmproj`, `retrieval(gguf)`, `loadLanguage`, `loadEmbedder`, `loadReranker`, sampling defaults |
| `Lfm2ChatTemplate` | `ChatTemplate`: token-exact ChatML codec with a per-checkpoint `Dialect`, think spans, tool calls, image blocks |
| `Lfm2ToolCodec` | Python-style tool-call wire: `parse` (JSON or Python literals), `renderCalls`, `renderTools` |
| `Lfm2Vision` | `MediaProjector<Media.Image>`: SigLIP2-NaFlex tower + pixel-unshuffle projector into the text residual width |
| `Lfm2VisionPreprocess` | Aligned resize, 512 px tile plan (`Plan`/`Part`/`Geometry`/`Options`), patch extraction |
| `Lfm2Colbert` | `Reranker<State>`: MaxSim over per-token `dense_2` vectors (query padded to 32, docs cut to 512, punctuation skiplist) |
| `Lfm2CheckpointCodec` | `CheckpointCodec<State>`: F16 KV rows of attention layers plus the F32 short-conv history of conv layers |

---

## 1. Position in the jinfer stack

```mermaid
flowchart TB
    Callers["Callers: cli / server / langchain4j / spring-ai"]
    subgraph chat["jinfer-chat"]
        Models["Models (ServiceLoader)<br/>load / loadEmbedder / loadReranker"]
        MP["ModelProvider"]
        LM1["LoadedModel (chat template, stops, sampling defaults)"]
        LE["LoadedEmbedder"]
        LR["LoadedReranker"]
    end
    subgraph core["jinfer-core"]
        LMI["LanguageModel / EmbeddingModel / Reranker / Multimodal"]
        CS["ContextState"]
        CC["CheckpointCodec"]
        B["Batch (Tokens | Sequences | Embeddings)"]
        MED["Media / MediaProjector"]
    end
    subgraph lfm["jinfer-lfm2"]
        P["Lfm2Provider"]
        M["Lfm2"]
        V["Lfm2Vision + Lfm2VisionPreprocess"]
        T["Lfm2ChatTemplate + Lfm2ToolCodec"]
        CB["Lfm2Colbert"]
        C["Lfm2CheckpointCodec"]
    end
    subgraph kern["jinfer-kernels"]
        K["MatMul, Norms, RoPE, FlashAttention, Convolutions, Moe, Activations, Convert, Ops, ModelLoader"]
    end
    Tok["toknroll Tokenizer"]
    Callers --> Models --> P
    P -.->|implements| MP
    P -->|produces| LM1
    P -->|produces| LE
    P -->|produces| LR
    M -.->|implements| LMI
    V -.->|implements| MED
    CB -.->|implements| LMI
    C -.->|extends| CC
    M --> K
    V --> K
    M --> Tok
    LM1 --> M
    LM1 --> T
    LE --> M
    LR --> CB
```

---

## 2. Class diagrams

### 2a. Contracts and relations

```mermaid
classDiagram
    direction LR
    class ModelProvider { <<interface, external>> }
    class LanguageModel~C,W,S~ { <<interface, external>> }
    class EmbeddingModel~C,W,S~ { <<interface, external>> }
    class Multimodal { <<interface, external>> }
    class MediaProjector~R~ { <<interface, external>> }
    class Reranker~S~ { <<interface, external>> }
    class ChatTemplate { <<interface, external>> }
    class ContextState { <<abstract, external>> }
    class CheckpointCodec~S~ { <<abstract, external>> }
    class Lfm2Provider {
        <<final>>
        architectures() lfm2, lfm2moe
        companionFiles() media to mmproj
        loadLanguage() LoadedModel
        retrieval(gguf) Optional
        loadEmbedder() LoadedEmbedder
        loadReranker() LoadedReranker
        samplingDefaults()$
    }
    class Lfm2 { <<final>> }
    class State { <<Lfm2.State>> }
    class Lfm2Vision { <<package-private final>> }
    class Lfm2ChatTemplate { <<package-private final>> }
    class Lfm2ToolCodec { <<package-private utility>> }
    class Lfm2Colbert { <<package-private final>> }
    class Lfm2CheckpointCodec { <<package-private final>> }
    ModelProvider <|.. Lfm2Provider
    LanguageModel <|.. Lfm2
    EmbeddingModel <|.. Lfm2
    Multimodal <|.. Lfm2
    MediaProjector <|.. Lfm2Vision
    Reranker <|.. Lfm2Colbert
    ChatTemplate <|.. Lfm2ChatTemplate
    ContextState <|-- State
    CheckpointCodec <|-- Lfm2CheckpointCodec
    Lfm2Provider ..> Lfm2 : loadModel
    Lfm2Provider ..> Lfm2ChatTemplate : fromGguf
    Lfm2Provider ..> Lfm2Colbert : new
    Lfm2 *-- Lfm2Vision : optional sidecar
    Lfm2 ..> Lfm2CheckpointCodec : checkpointCodec if causal
    Lfm2 *-- State : newState
    Lfm2CheckpointCodec ..> State : transfer
    Lfm2Colbert --> Lfm2 : model
    Lfm2ChatTemplate --> Lfm2Vision : vision
    Lfm2ChatTemplate ..> Lfm2ToolCodec : parse / render
```

### 2b. Model internals — `Lfm2` and its nested types

```mermaid
classDiagram
    direction TB
    class Lfm2 {
        <<final>>
        -Configuration configuration
        -Tokenizer tokenizer
        -Weights weights
        -Lfm2Vision vision
        +loadModel(path, arena)$
        +loadModel(text, mmproj, arena)$
        +withMedia(mmproj, arena) Lfm2
        +ingest(State, Batch)
        +logits(State, output) MemoryView
        +projectEmbedding(State, idx, consumer)
        +embedAll(State, Sequences, sink)
        +projector(Class) Optional
        -forward(State, Batch)
        -forward(State, ids, startPos, seqLen, lastOnly)
        -tail()
        -forwardEmbeddings()
        -forwardSegmented()
        -layer()
        -shortConvMixer()
        -shortConvMixerCentered()
        -attention()
        -attentionBidirectional()
        -attentionProject() projectKeys() attentionFinish()
        -headNormRope()
        -feedForward() moeFeedForward()
        -commitKv()
        ~colbertRow(State, row)
        ~forEachSequence(State, seqs, visitor)
    }
    class Configuration {
        <<record>>
        embeddingLength, numberOfLayers, numberOfHeads
        feedForwardLength[] per layer
        numberOfKeyValueHeadsPerLayer[] 0 = conv layer
        headSize, ropeTheta, rmsNormEps
        logitSoftcapping, shortConvLCache
        expertCount, expertUsedCount, expertFeedForwardLength
        leadingDenseBlockCount, expertGatingFunc
        causalAttention, poolingType, embeddingLengthOut
        isRecurrentLayer(l) isMoELayer(l)
        isEmbedder() isColbert()
        lastAttentionLayer() tailRows()
    }
    class Weights {
        <<record>>
        tokenEmbeddings
        LayerWeights[] layers
        finalNorm
        RoPE.Schedule rope
        wcls (tied if absent)
        dense2 (ColBERT only)
    }
    class LayerWeights {
        <<record>>
        attnNorm postAttnNorm
        ffnNorm postFfnNorm
        attention OR shortConv
        dense OR moe
    }
    class AttentionWeights { <<record>> wq wk wv(optional) wo qNorm kNorm }
    class ShortConvWeights { <<record>> kernel inProj outProj }
    class DenseFfnWeights { <<record>> gate up down }
    class MoeFfnWeights { <<record>> router gateExps[] upExps[] downExps[] expProbsBias }
    class State {
        <<static final>>
        residual normed branchOut attnOut query
        hidden hidden2 logits ropeCos ropeSin
        keyCache[] valueCache[] F16, attention layers
        batchK[] batchV[] F32, attention layers
        shortConvState[] F32, conv layers, causal only
        shortConvTmp shortConvOut
        moe scratch and Moe.Routing, MoE only
        embedScratch lazy
        decodeScratch
        clearHistory() zeroes conv state
    }
    class EmbedScratch {
        <<static final>>
        segQ segK segV segOut
        embOut colbertOut colbertRows
        posOf[] segRow0[]
    }
    Lfm2 *-- Configuration
    Lfm2 *-- Weights
    Weights *-- LayerWeights
    LayerWeights o-- AttentionWeights
    LayerWeights o-- ShortConvWeights
    LayerWeights o-- DenseFfnWeights
    LayerWeights o-- MoeFfnWeights
    Lfm2 ..> State : newState
    State o-- EmbedScratch : lazy
    Configuration ..|> ContextConfiguration
```

### 2c. Vision, retrieval and chat-template subsystems

```mermaid
classDiagram
    direction LR
    class Lfm2Vision {
        <<final>>
        patchSize visionDim headCount headDim ffnDim
        merge projectorDim modelDim positionSide
        imageMean imageStd
        patchWeight patchBias positionEmbedding[]
        postNorm projectorNorm optional
        projectorUp projectorDown
        Layer[] layers
        positions(image) int
        plan(image) Plan
        planId() String
        project(image, maxChunk, sink)
        embed(part, maxChunk, sink)
        -encode(image, scratch)
        loadModel(path, gguf, tensors)$
    }
    class Linear { <<record>> weight bias outputDim inputDim }
    class Layer { <<record>> ln1 q k v attnOut ln2 ffnUp ffnDown }
    class Lfm2VisionPreprocess {
        <<utility>>
        defaults(patch, merge)$
        plan(image, patch, merge, options)$
        positions()$
        smartResize()$
        closestGrid()$
        resizeRegion()$
        patches()$
    }
    class Options { <<record>> minPixels maxPixels tileSize minTiles maxTiles maxPixelsTolerance }
    class Part { <<record>> image row column thumbnail() }
    class Geometry { <<record>> overviewWidth overviewHeight rows columns }
    class Plan { <<record>> parts tiled() }
    class Lfm2Colbert {
        <<final>>
        QUERY_LENGTH 32
        DOCUMENT_LENGTH 512
        queryMarker documentMarker bos pad
        skiplist punctuation ids
        scoreAll(state, instr, query, docs, sink)
        maxSim()
        querySequence()$
        documentSequence()
    }
    class Lfm2ChatTemplate {
        <<final>>
        Dialect dialect
        encode(conversation, batchCapacity, cache, sink) ReplyState
        thinkingPolicy()
        parser() constrainedReply() forcedCall()
        mediaPositions(media)
        fromGguf()$
    }
    class Dialect { <<record>> promptOpensThinking lastAssistantKeepsThinking escapesArguments rendersThinking modelOpensThinking }
    class Lfm2ToolCodec {
        <<utility>>
        parse(payload)$
        renderCalls(calls, escape)$
        renderTools(tools)$
    }
    class Python { <<inner parser>> Python-style call literals }
    Lfm2Vision *-- Linear
    Lfm2Vision *-- Layer
    Lfm2Vision *-- Options
    Lfm2Vision ..> Lfm2VisionPreprocess : plan / positions / patches
    Lfm2VisionPreprocess *-- Options
    Lfm2VisionPreprocess ..> Plan
    Lfm2VisionPreprocess ..> Geometry
    Plan *-- Part
    Lfm2ChatTemplate *-- Dialect
    Lfm2ChatTemplate --> Lfm2Vision
    Lfm2ChatTemplate ..> Lfm2ToolCodec
    Lfm2ToolCodec *-- Python
```

---

## 3. Loading sequences

### 3a. Provider flows

`Models` calls `retrieval(gguf)` first, and `loadLanguage` refuses a non-causal (retrieval) header **before** any weight is read.

```mermaid
sequenceDiagram
    participant M as Models (jinfer-chat)
    participant P as Lfm2Provider
    participant L as Lfm2
    participant ML as ModelLoader
    participant V as Lfm2Vision
    participant T as Lfm2ChatTemplate
    participant R as Lfm2Colbert

    M->>P: retrieval(gguf)
    alt attention.causal true or absent
        P-->>M: empty (generative)
    else pooling_type == 2 (CLS)
        P-->>M: EMBEDDING
    else embedding_length_out greater than 0
        P-->>M: RERANKING
    else
        P-->>M: empty
    end
    alt loadLanguage
        M->>P: loadLanguage(channel, gguf, path, arena, companions, tokenizer)
        P->>P: non-causal header: throw IncompatibleModelException (before any weight is read)
        P->>L: loadModel(channel, gguf, arena, tokenizer)
        L->>L: tokenizer from GGUF if null, then readConfiguration(gguf, vocab)
        L->>ML: loadTensors(channel, gguf, arena)
        L->>L: loadWeights(tensors, config)
        L-->>P: Lfm2
        opt companions has media (mmproj)
            P->>L: withMedia(mmproj, arena)
            L->>ML: readGguf + loadTensors (mmproj)
            L->>V: loadModel(path, gguf, tensors) (projector_type must be lfm2, projection_dim == model width)
            L-->>P: new Lfm2 sharing config, tokenizer, weights + vision
        end
        P->>T: fromGguf(model, gguf) (Dialect.of(gguf), special tokens required)
        P-->>M: LoadedModel(model, tok, template source, stops, modelSeed, template, samplingDefaults)
    else loadEmbedder
        M->>P: loadEmbedder(...)
        P->>L: loadModel(...)
        P->>P: isEmbedder() else IllegalArgumentException
        P-->>M: LoadedEmbedder(prefix BOS, no suffix, dim = embeddingLength, "query: ", "document: ")
    else loadReranker
        M->>P: loadReranker(...)
        P->>L: loadModel(...)
        P->>R: new Lfm2Colbert(model, bos, pad) (requires isColbert and dense2)
        R->>R: resolve [Q] / [D] marker ids by decoded text, build punctuation skiplist
        P-->>M: LoadedReranker(model, colbert, file name)
    end
```

Stop tokens for chat: `SpecialTokens.stops(tok, -1, "<|im_end|>", "<eos>", "<|endoftext|>", "<end_of_turn>")`. Sampling defaults
(`samplingDefaults`, overridden by GGUF `general.sampling.*`): non-LFM2.5 text 0.3 temp / min-p 0.15; LFM2.5 dense 0.1 / top-k 50;
LFM2.5 MoE 0.2 / top-k 80; vision with a large vocab 0.2 / top-k 50, other vision 0.1 / min-p 0.15.

### 3b. Configuration and weights

```mermaid
flowchart TB
    G["GGUF metadata + tensor index"] --> A{"general.architecture"}
    A -->|"lfm2 / lfm2moe"| D["core dims: context_length, embedding_length,<br/>head_count, block_count (all greater than 0)<br/>headSize = dim / heads (even)"]
    A -->|other| X["IllegalArgumentException"]
    D --> O["optional keys with defaults:<br/>rms eps 1e-5, rope freq_base 1e6, softcapping 0,<br/>shortconv.l_cache 3, expert_count 0, gating_func 1,<br/>attention.causal true, pooling_type 0, embedding_length_out 0"]
    O --> F["feed_forward_length: int[] per layer or one scalar broadcast"]
    F --> K["per layer: blk.i.attn_k.weight present?<br/>shape[1] / headSize = kv heads, else 0 = short-conv layer"]
    K --> C["Configuration record + consistency checks<br/>dense: arch lfm2, no expert keys<br/>MoE: arch lfm2moe, topK le count, gating 1 or 2"]
    C --> W["loadWeights"]
    W --> W1["token_embd (tied wcls unless output.weight)<br/>finalNorm: output_norm or token_embd_norm"]
    W --> W2{"layer i"}
    W2 -->|"kvHeads == 0"| SC["ShortConvWeights: shortconv.conv (dim x lCache F32),<br/>in_proj (3*dim x dim), out_proj"]
    W2 -->|"kvHeads greater than 0"| AT["AttentionWeights: attn_q/k/(v optional)/output, q_norm, k_norm"]
    W2 --> FF{"isMoELayer(i)<br/>expertCount greater than 0 and i ge leadingDenseBlockCount"}
    FF -->|yes| MO["MoeFfnWeights: ffn_gate_inp, gate/up/down_exps sliced per expert,<br/>exp_probs_b.bias optional"]
    FF -->|no| DE["DenseFfnWeights: ffn_gate / up / down"]
    W --> W3["dense_2.weight if embedding_length_out greater than 0"]
```

RoPE is `RoPE.plain(headSize, ropeTheta)`. Every weight shape is checked against the expected `[rows, columns]` at load.

---

## 4. Forward pass

`ingest(state, batch)` runs `state.exclusively(forward)`. Layers are heterogeneous: conv layers keep a rolling F32 history of `lCache-1` rows, attention layers keep an F16 KV cache.

### 4.0 Global architecture

```mermaid
flowchart TB
    Tok["token ids or image embedding rows"] --> Emb["Embed: gather token rows to F32 residual<br/>(no scaling) or copy projected rows"]
    subgraph Stack["Hybrid trunk: N blocks, kind chosen per layer by kvHeads"]
        direction TB
        subgraph Block["decoderBlock = layer(l): pre-norm residual x2"]
            direction LR
            Mix{"kvHeads[l] == 0 ?"}
            Mix -->|yes| SC["RMSNorm attn_norm, gated short-conv<br/>(in-proj B|C|x, causal FIR, out-proj)"]
            Mix -->|no| GQ["RMSNorm attn_norm, GQA attention<br/>QK-norm + NeoX RoPE, KV in F16 cache"]
            SC --> Add1(("+"))
            GQ --> Add1
            Add1 --> Ff{"layer routes MoE ?"}
            Ff -->|no| DF["RMSNorm ffn_norm, dense SwiGLU"]
            Ff -->|yes| MF["RMSNorm ffn_norm, top-k MoE SwiGLU"]
            DF --> Add2(("+"))
            MF --> Add2
        end
    end
    Emb --> Stack
    Stack --> FN["Final RMSNorm (output_norm or token_embd_norm)"]
    FN --> H1["generative: tied LM head, softcap, logits"]
    FN --> H2["embedder: L2-normalised CLS (first row) vector"]
    FN --> H3["ColBERT: dense_2 per-token vector, L2-normalised"]
    Stack -.-> St[("State: F16 KV per attention layer,<br/>F32 conv history per conv layer")]
```

### 4a. `ingest` — batch validation, input kinds, RoPE/embedding, layer loop, advance

```mermaid
flowchart TB
    In["ingest(state, Batch)"] --> Ex["state.exclusively(forward)"]
    Ex --> V{"count in (0, batchCapacity]<br/>position + count le contextCapacity"}
    V -->|no| Err["IllegalArgumentException"]
    V -->|yes| Sw{"batch.input()"}
    Sw -->|Tokens| TK{"causalAttention ?"}
    TK -->|no| U1["UnsupportedOperationException<br/>(retrieval needs packed sequences)"]
    TK -->|yes| RT["requireTokens: ids in [0, vocab)"]
    RT --> RP["RoPE.fill(ropeCos, ropeSin, startPos, seqLen)<br/>ONCE per batch, a contiguous range"]
    RP --> EM["embedTokens: Convert.gatherToF32 to residual"]
    EM --> LO{"outputs == LAST and has attention layer<br/>and seqLen gt tailRows ?"}
    LO -->|no| LP["for l in 0..N-1: layer(l)  see 4b"]
    LP --> CK["commitKv: all attention layers<br/>F32 batch K/V to F16 cache at startPos"]
    LO -->|yes| TL["tail(): layers 0..cut-1 full batch,<br/>layer cut: norm + projectKeys only, commitKv,<br/>move last tailRows residual rows to 0..tail-1,<br/>RoPE.fill at startPos+skip, layers cut..N-1 on tail rows,<br/>move row tail-1 back to seqLen-1"]
    Sw -->|Sequences| SQ{"causalAttention ?"}
    SQ -->|yes| U2["UnsupportedOperationException<br/>(needs the embedding checkpoint)"]
    SQ -->|no| FS["requireCovers, requireTokens<br/>forwardSegmented  see 4b.3"]
    Sw -->|Embeddings| EB{"vision loaded and width == embeddingLength ?"}
    EB -->|no| U3["UnsupportedOperation or IllegalArgument"]
    EB -->|yes| FE["forwardEmbeddings: RoPE.fill range,<br/>copy rows to residual, layer(l, bidirectional flag)"]
    FE --> CK2["commitKv"]
    CK --> Adv["state.advance(batch): advanceContext(count, outputs)"]
    TL --> Adv
    FS --> Adv
    CK2 --> Adv
    Adv -.->|later| Heads["logits / projectEmbedding / colbertRow  see 5"]
```

The **tail optimisation**: when only the last row is read, rows past the last attention layer's K/V only feed their own output and the trailing convs' look-back. `tailRows = 1 + (trailing conv layers) × (lCache − 1)` rows are finished alone, reading the earlier rows' keys back from the cache as a next batch would.

### 4b. `decoderBlock` — one layer (`layer(l)`)

```mermaid
flowchart TB
    R0["residual [rows x dim]"] --> L{"isRecurrentLayer(l)<br/>kvHeads == 0 ?"}
    L -->|yes| CV["shortConvMixer (see 4b.1)<br/>adds into residual"]
    L -->|no| N1["Norms.rmsnormRows to normed (attnNorm)"]
    subgraph Attn["attention(l)"]
        direction TB
        N1 --> Q["gemm wq to query, then headNormRope(qNorm)"]
        N1 --> K["gemm wk to batchK[l]"]
        N1 --> V["gemm wv to batchV[l]<br/>(wv absent: V = copy of K before norm)"]
        K --> KN["headNormRope(kNorm) on K only (no V-norm)"]
        Q --> Path{"seqLen gt 1 ?"}
        KN --> Path
        V --> Path
        Path -->|yes| P1["FlashAttention.slidingWindowPrefill<br/>window 0 = full causal, reads keyCache + batch K/V,<br/>bidirectional flag for image rows"]
        Path -->|no| P2["FlashAttention.flashDecode<br/>(cache + current K/V, decodeScratch)"]
        P1 --> AO["attnOut"]
        P2 --> AO
        AO --> WO["gemm wo to branchOut"]
        WO --> PN{"postAttnNorm present ?"}
        PN -->|yes| PN1["rmsnormRows in place"]
        PN -->|no| AD
        PN1 --> AD["Ops.addRows: residual += branchOut"]
    end
    CV --> N2
    AD --> N2["Norms.rmsnormRows to normed (ffnNorm)"]
    N2 --> M{"weights.layers[l].moe() != null ?"}
    M -->|no| DF["Dense FFN: gemm gate to hidden, gemm up to hidden2,<br/>siluMultiply per row at hiddenStride, gemm down to normed,<br/>optional postFfnNorm"]
    M -->|yes| MO["moeFeedForward (see 4b.2)"]
    DF --> A2["Ops.addRows: residual += normed"]
    MO --> A3["residual += moeOutB rows"]
    Note["commitKv is NOT in the layer: it runs once after the loop,<br/>because each attention layer has its own batchK/batchV[l] buffer"]
    A2 -.-> Note
```

Details worth knowing:

- Scale is `1/sqrt(headSize)`; GQA via `kvMul = heads / kvHeadsPerLayer[l]`; `kvDim(l) = kvHeads[l] × headSize` varies per layer, and `hidden`/`hidden2` are sized to `maxHiddenDim` (dense layers may differ in width, so rows are addressed at the buffer stride).
- `attn_norm` is the mixer pre-norm of conv layers too.
- The FFN pre-norm output is shared by dense and MoE paths.

### 4b.1 Short-conv mixer (recurrent layer)

```mermaid
flowchart TB
    R["residual"] --> N["rmsnormRows (attnNorm as mixer pre-norm) to normed"]
    N --> IP["gemm inProj (3*dim x dim) to shortConvTmp [rows x 3*dim]<br/>split per row: B | C_gate | x blocks, each dim wide"]
    IP --> SCAN["Convolutions.shortConvScan(kernel [dim x lCache], shortConvState[l], tmp, branchOut)<br/>gate B*x, causal FIR over lCache taps,<br/>history = lCache-1 previous rows, rolled forward"]
    SCAN --> OP["gemm outProj (dim x dim) to shortConvOut"]
    OP --> ADD["Ops.addRows: residual += shortConvOut"]
    SCAN <-.->|"F32 [(lCache-1) x dim] per conv layer"| ST[("shortConvState[l]")]
```

### 4b.2 MoE feed-forward (LFM-style)

No shared MLP and no expert pre/post norms; the router runs on the `ffnNorm` output.

```mermaid
flowchart TB
    N["rmsnormRows (ffnNorm) to normed"] --> RT["gemm router [experts x dim] to moeRouterB"]
    RT --> G{"expertGatingFunc"}
    G -->|"2"| SG["sigmoid per expert"]
    G -->|"1 (default)"| SM["softmax over experts"]
    SG --> B{"expProbsBias present ?"}
    SM --> B
    B -->|yes| SEL["moeSelectionB = probs + bias<br/>(bias steers selection only)"]
    B -->|no| SEL2["selection = probs"]
    SEL --> TOP["Moe.selectTopK(selection, probs, topK)<br/>moeRowTopE / moeRowTopP / moeExpertCounts<br/>weights read from UNBIASED probs"]
    SEL2 --> TOP
    TOP --> NP["Moe.normalizeTopP: renormalise the k weights"]
    NP --> DSP["Moe.dispatch (CSR routing by expert):<br/>gather rows to moeGather"]
    DSP --> EX["per expert e with n rows:<br/>gemm gateExps[e], gemm upExps[e], siluMultiply at expertFF width,<br/>gemm downExps[e]"]
    EX --> SC["scatter prob-weighted rows to moeOutB"]
    SC --> PF{"postFfnNorm ?"}
    PF -->|yes| PN["rmsnorm each row of moeOutB"]
    PF -->|no| AD
    PN --> AD["residual += moeOutB row"]
```

The first `leadingDenseBlockCount` layers are dense even in an `lfm2moe` model.

### 4b.3 Non-causal segmented forward (embedding and ColBERT checkpoints)

```mermaid
flowchart TB
    In["forwardSegmented(state, ids, seqLen[], n)"] --> PO["EmbedScratch: posOf[] restarts at 0 per sequence,<br/>segRow0[g] = first row of sequence g"]
    PO --> RF["RoPE.fill(posOf)"]
    RF --> EM["embedTokens"]
    EM --> Lp{"for each layer l"}
    Lp -->|conv| CC["shortConvMixerCentered:<br/>same pre-norm / in-proj / out-proj,<br/>Convolutions.segmentedShortConv: centred window pad = (dConv-1)/2,<br/>zeros beyond each sequence edge, no rolling state"]
    Lp -->|attention| AB["attentionBidirectional: attentionProject for all rows,<br/>then per sequence: copy q/K/V rows to segQ/segK/segV,<br/>FlashAttention.bidirectionalPrefill, copy back to attnOut,<br/>attentionFinish"]
    CC --> FF["feedForward(l) dense or MoE"]
    AB --> FF
    FF --> Lp
    Lp -->|done| NoKv["no KV cache, no conv state read or written<br/>(State allocates none when causalAttention is false)"]
```

Each sequence must be forwarded whole. `forEachSequence` packs whole sequences greedily into groups capped by `min(batchCapacity, contextCapacity)`, calls `state.reset()` per group, ingests with all outputs, and hands each sequence's first retained row to a visitor; a sequence longer than the cap is refused by name. `embedAll` uses it to emit one CLS vector per sequence.

---

## 5. Heads

All heads read the retained residual row `lastBatchSize - outputCount + output`.

```mermaid
flowchart LR
    Res["residual row = lastBatchSize - outputCount + output"] --> FN["Norms.rmsnorm(finalNorm) to normed"]
    FN --> LG["logits(state, i): MatMul.gemv(wcls, normed) to logits [vocab]<br/>Activations.softcap(logitSoftcapping)"]
    Res --> EN["projectEmbedding: rmsnorm(finalNorm) to embOut,<br/>multiply by 1/L2 norm (0 for zero vector)<br/>CLS = first retained row of the sequence (its BOS)"]
    Res --> CR["colbertRow(row): rmsnorm(finalNorm),<br/>gemv dense2 to embeddingLengthOut,<br/>L2-normalise, reused buffer colbertOut"]
    EN --> EA["embedAll: forEachSequence, one CLS vector per sequence in input order"]
    CR --> CL["Lfm2Colbert: MaxSim"]
```

---

## 6. State and checkpoint layout

```mermaid
flowchart TB
    subgraph State["Lfm2.State (per context)"]
        direction TB
        subgraph scratch["Scratch sized by batchCapacity"]
            A["residual, normed, branchOut [batch x dim]<br/>attnOut, query [batch x queryDim]<br/>hidden, hidden2 [batch x maxHiddenDim]"]
            B["ropeCos, ropeSin [batch x headSize/2]<br/>logits [1 x vocab]<br/>shortConvTmp [batch x 3*dim], shortConvOut"]
            C["batchK[l], batchV[l] F32 [batch x kvDim(l)], attention layers"]
            D["MoE only: moeRouterB, moeSelectionB, moeGather, moeDownB, moeOutB,<br/>moeHidden, moeHidden2, topE, topP, expertCounts, Moe.Routing"]
            E["embedScratch lazy: segQ segK segV segOut embOut colbertOut colbertRows posOf segRow0"]
        end
        subgraph persist["Persistent, only when causalAttention"]
            KV["keyCache[l], valueCache[l]: F16 [contextCapacity x kvDim(l)]<br/>attention layers; null on conv layers"]
            CS["shortConvState[l]: F32 [(lCache-1) x dim]<br/>conv layers; null on attention layers"]
        end
    end
    subgraph Blob["Lfm2CheckpointCodec blob (only when causalAttention)"]
        direction TB
        B1["1. attention layers in order: K rows then V rows<br/>for positions [from, to): positions x 2 x 2 bytes x kvDim(l)"]
        B2["2. conv layers in order: full shortConvState[l] as FP32<br/>(lCache-1) x dim x 4 bytes, position independent"]
    end
    KV -->|"transfer(capture)"| B1
    B1 -->|"transfer(restore)"| KV
    CS -->|"transfer(capture)"| B2
    B2 -->|"transfer(restore)"| CS
```

- `sizeOf(positions) = positions × bytesPerPosition + residueBytes`, where `bytesPerPosition = Σ 2·2·kvDim(l)` over attention layers and `residueBytes = convLayers × (lCache−1) × dim × 4`.
- `clearHistory()` zeroes every conv history (stale KV rows past the cursor are masked by attention and overwritten); `State` rejects `contextCapacity > maxContextLength`.
- `checkpointCodec()` is empty for non-causal (retrieval) checkpoints.
- `newState(ctx, batch)` makes a cross-thread arena owned by the state; `newState(ctx, batch, arena)` borrows the caller's.

---

## 7. Chat template and tool codec

`Lfm2ChatTemplate` mirrors the ChatML Jinja template token-exactly. `Dialect.of(gguf)` reads from the template source: `promptOpensThinking` (source contains `<|im_start|>assistant\n<think>`), `lastAssistantKeepsThinking` (`last_assistant_index`), `escapesArguments` (no `format_arg_value`, or `replace(`), `rendersThinking` (`<think>` present), and `modelOpensThinking` (architecture is `lfm2moe`). `thinkingPolicy()` is `ALWAYS` for reasoning-only (MoE, prompt does not open the span), `OPTIONAL` when thinking is rendered, else `NONE`.

```mermaid
flowchart TB
    E["encode(conversation, batchCapacity, mediaCache, sink)"] --> RS["requireSupported per message:<br/>Text; assistant Reasoning / ToolCall; tool ToolResult;<br/>user image only if vision loaded"]
    RS --> S0["ids: startoftext"]
    S0 --> SY{"system text or tools ?"}
    SY -->|yes| SYS["im_start + 'system\n' + system text + tools line<br/>'List of tools: [json, ...]' + im_end + newline"]
    SY -->|no| LU
    SYS --> LU["lastUser, lastAssistant indices"]
    LU --> T{"each message i"}
    T --> KT["keepThinking = lastAssistantKeepsThinking ? i == lastAssistant : i gt lastUser"]
    KT --> RO{"role"}
    RO -->|assistant| AS["replay verbatim from prompt cache if possible,<br/>else reasoning span (think ids) if kept,<br/>visible text (thinking stripped if not kept),<br/>tool calls: call_start + renderCalls + call_end"]
    RO -->|"user / tool"| UT["Text and ToolResult as text,<br/>Image via writeMedia (see 8)"]
    AS --> CL["im_end + newline"]
    UT --> CL
    CL --> T
    T -->|done| GP["im_start + 'assistant\n'"]
    GP --> PT{"dialect.promptOpensThinking ?"}
    PT -->|"yes, thinking on"| P1["+ think_open, replyPrefix = [think_open]"]
    PT -->|"yes, thinking off"| P2["+ think_open + think_close,<br/>replyPrefix = both"]
    PT -->|no| P3["bare header, empty replyPrefix"]
    P1 --> FIN["out.finish(), ReplyState(prefix, ReplyParser.spans)"]
    P2 --> FIN
    P3 --> FIN
    FIN --> TC{"tools empty ?"}
    TC -->|yes| DIS["parser.disableToolCalls()"]
    TC -->|no| OK["parser parses call_start..call_end via Lfm2ToolCodec.parse"]
```

Tool wire: calls render as `[name(arg=value, ...)]` between `<|tool_call_start|>` and `<|tool_call_end|>`; `escape` selects the 2.6B argument formatting (escaped strings, JSON lists) over the 8B one (raw strings, Python repr lists); mappings are JSON in both. `parse` tries JSON first (payload starting with `{` or `[{`) and falls back to the inner Python-literal parser; failure yields no calls. `forcedCall` constrains a reply to start with `[<toolName>`.

---

## 8. Vision sidecar

Loaded only when an `mmproj` companion is supplied (`withMedia`). Metadata comes from `clip.vision.*` (defaults: patch 16, dim 1152, 16 heads, 27 blocks, ffn 4304, merge 2).

```mermaid
flowchart TB
    IMG["Media.Image (float RGB)"] --> GEO["geometry: smartResize to multiples of patch*merge,<br/>pixels in [64,256] x factor^2"]
    GEO --> TL{"rounded area le maxPixels x tolerance 2 ?"}
    TL -->|yes| ONE["Plan: single overview part"]
    TL -->|no| GR["closestGrid(tileSize 512, 1..10 tiles)<br/>rows x columns by aspect ratio"]
    GR --> TILES["Plan: tiles row-major (row+1, col+1) resized from region,<br/>then overview (row 0 = thumbnail)"]
    ONE --> PART
    TILES --> PART["for each Part: embed(part, maxChunk, sink)"]
    PART --> PA["patches: normalise with mean/std,<br/>flatten 3 x patch x patch"]
    PA --> PE["gemm patchWeight + bias"]
    PE --> POS["+ interpolated learned position table<br/>(positionSide^2 grid resampled to patchesX x patchesY)"]
    POS --> LY["for each vision Layer (SigLIP2 NaFlex):<br/>LayerNorm1, q/k/v linear (+bias), K/V to F16,<br/>FlashAttention.bidirectionalPrefill, out linear, residual,<br/>LayerNorm2, ffnUp, GELU, ffnDown, residual"]
    LY --> PL["post LayerNorm"]
    PL --> MG["merge: pixel-unshuffle merge x merge patches<br/>to one row of visionDim x merge^2"]
    MG --> PR["optional LayerNorm 1e-5, mm.1 linear, GELU, mm.2 linear to modelDim"]
    PR --> CH["slice rows into chunks of maxChunkSize to sink"]
    CH --> BE["Lfm2ChatTemplate: Batch.embeddings(rows, count, causal false, contentKey)<br/>between image_start / thumbnail or row_col marker ids / image_end"]
    BE --> ING["Lfm2.ingest: Embeddings branch, forwardEmbeddings"]
```

`planId()` serialises patch, merge, position side, pixel bounds, tiling and mean/std so the media cache key changes with the preprocessing. Image rows are projected per `Part` into a per-call cross-thread scratch arena that is closed after the chunks are emitted.

---

## 9. ColBERT scoring

```mermaid
sequenceDiagram
    participant C as caller
    participant R as Lfm2Colbert
    participant L as Lfm2
    participant S as State

    C->>R: scoreAll(state, instruction, query, docs, sink)
    R->>R: blank instruction required, else UnsupportedOperationException
    R->>R: querySequence: BOS, Q marker, query ids, padded with pad to 32
    R->>L: forEachSequence(state, query) (reset, ingest packed with all outputs)
    loop each of 32 query rows
        R->>L: colbertRow(state, row)
        L-->>R: normalised dense_2 vector (copied to colbertRows)
    end
    R->>R: documentSequence: BOS, D marker, doc ids, truncated to 512
    R->>L: forEachSequence(state, docs): greedy groups under min(batch, context) tokens
    loop each group (state.reset, ingest) and each doc in it
        loop each document row not in punctuation skiplist
            R->>L: colbertRow(state, row)
            R->>R: best[q] = max over rows of dot(query q, row)
        end
        R-->>C: sink.accept(sum over q of best[q])
    end
```

Scores are unbounded MaxSim sums (a 32-row query tops out at 32), not probabilities. There is no instruction slot (`hasInstructionSlot() == false`) and no prefix reuse: the forward is bidirectional and conv history is not addressable by position, so `scoreAll` is owned outright instead of using the cross-encoder template. The `[Q] `/`[D] ` markers are added tokens found by decoded text, scanning the vocabulary from the top.
