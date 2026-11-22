# jinfer-gemma-embedding2 — class architecture

Package `com.qxotic.jinfer.models.gemmaembedding2` (4 classes, ~1240 lines). It is the `jinfer-<model>` provider for GGUF
architecture `gemma-embedding2` (Google EmbeddingGemma 2): a **bidirectional** Gemma 4 text encoder, mean-pooled into one
L2-normalised vector per input. It is **not a generative model** (no logits, no LM head, **no KV cache**, no checkpoint codec)
and it is **multimodal**: an optional Gemma 4 `mmproj` companion adds image, audio and video towers.

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends;
classes marked `<<external>>` live outside this package.

1. [Position in the stack](#1-position-in-the-jinfer-stack)
2. [Class diagrams](#2-class-diagrams) — 2a contracts, 2b model internals, 2c template and media
3. [Loading sequence](#3-loading-sequence)
4. [Forward pass](#4-forward-pass) — 4.0 global, 4a `ingest` / `embed`, 4b `decoderBlock`, 4c `GemmAttention`
5. [Pooling heads and `embedAll`](#5-pooling-and-embedall)
6. [State layout](#6-state-layout)

| Class | Role |
|---|---|
| `GemmaEmbedding2` | The model: `EmbeddingModel` + `Multimodal` impl with nested `Configuration`, `LayerWeights`, `Weights`, `State`; staging, the whole encoder forward, pooling, task-prompt helpers, GGUF loading |
| `GemmaEmbedding2Provider` | `ModelProvider` SPI: always `Retrieval.EMBEDDING`, declares the `media` companion (mmproj), builds `LoadedEmbedder` |
| `GemmaEmbedding2Template` | Input template: special-aware text encoding, BOS/EOS framing, interleaved text/image/audio/video placement into a `PromptWriter` |
| `GemmAttention` | Package-private kernel: unscaled non-causal attention over per-query key ranges `[first, end)`, direct dot products or batched GEMMs, with its `Scratch` record |

---

## 1. Position in the jinfer stack

```mermaid
flowchart TB
    Callers["Callers: cli / server / langchain4j / spring-ai"]
    subgraph chat["jinfer-chat"]
        Models["Models (ServiceLoader)<br/>loadEmbedder"]
        MP["ModelProvider"]
        LE["LoadedEmbedder"]
        PW["PromptWriter"]
    end
    subgraph core["jinfer-core"]
        EM["EmbeddingModel / Multimodal"]
        CS["ContextState"]
        B["Batch (Tokens | Embeddings | Sequences)"]
        MPJ["MediaProjector / Media"]
    end
    subgraph ge["jinfer-gemma-embedding2"]
        P["GemmaEmbedding2Provider"]
        M["GemmaEmbedding2"]
        T["GemmaEmbedding2Template"]
        GA["GemmAttention"]
    end
    G4["jinfer-gemma4<br/>Gemma4.Towers / loadTowers"]
    subgraph kern["jinfer-kernels"]
        K["MatMul · Norms · RoPE · Ops<br/>Activations · Convert · ModelLoader"]
    end
    Tok["toknroll Tokenizer / Specials"]

    Callers --> Models --> P
    P -.->|implements| MP
    P -->|produces| LE
    LE --> M
    M -.->|implements| EM
    M --> T
    M --> GA
    T --> PW
    M --> G4
    G4 --> MPJ
    M --> K
    GA --> K
    M --> Tok
    T --> Tok
```

---

## 2. Class diagrams

### 2a. Contracts and relations

```mermaid
classDiagram
    direction LR
    class ModelProvider { <<interface, external>> }
    class EmbeddingModel~C,W,S~ { <<interface, external>> }
    class Multimodal { <<interface, external>> }
    class ContextState { <<abstract, external>> }
    class ContextConfiguration { <<interface, external>> }
    class GemmaEmbedding2Provider {
        <<final>>
        architectures() Set
        retrieval(gguf) EMBEDDING
        companionFiles() media to mmproj
        loadEmbedder(channel, gguf, path, arena, companions, tokenizer)
    }
    class GemmaEmbedding2 {
        <<final>>
        ARCHITECTURE SEARCH_QUERY DOCUMENT
        bos eos
        loadModel(path or channel, ...)$
        withMedia(mmproj, arena)
        newState(ctx, batch)
        ingest(State, Batch)
        embed(State, parts, consumer) int
        embedAll(State, Sequences, sink)
        projectEmbedding(State, idx, consumer)
        projector(modality) / supports(modality)
    }
    class GemmaEmbedding2Template { <<package-private>> }
    class GemmAttention { <<package-private, static>> }
    class State { <<GemmaEmbedding2.State>> }
    class Configuration { <<record>> }

    ModelProvider <|.. GemmaEmbedding2Provider
    EmbeddingModel <|.. GemmaEmbedding2
    Multimodal <|.. GemmaEmbedding2
    ContextState <|-- State
    ContextConfiguration <|.. Configuration
    GemmaEmbedding2Provider ..> GemmaEmbedding2 : loadModel / withMedia
    GemmaEmbedding2 *-- Configuration
    GemmaEmbedding2 *-- GemmaEmbedding2Template : template
    GemmaEmbedding2 ..> State : newState
    GemmaEmbedding2 ..> GemmAttention : prefill
    State *-- GemmAttention : Scratch
```

### 2b. Model internals — `GemmaEmbedding2` and its nested types

```mermaid
classDiagram
    direction TB
    class GemmaEmbedding2 {
        <<final>>
        -stage(State, Batch)
        -stageTokens(State, ids, from, rows, at)
        -embedding(State)
        -pool(State, start, rows)
        -forward(State, lengths)
        -attention(State, l, rows)
        -feedForward(State, l, rows)
        -perLayerInput(State, l, rows)
        -normThenAdd(State, branch, norm, rows, scale)
        -loadConfiguration(gguf, vocab)$
        -swaLayers(gguf, layers)$
        -loadWeights(tensors, c)$
    }
    class Configuration {
        <<record>>
        embeddingLength, feedForwardLength
        numberOfLayers, numberOfHeads
        numberOfKeyValueHeads[] per layer
        headSizeFull, headSizeSwa
        ropeThetaFull, ropeThetaSwa
        slidingWindow, isSwa[]
        embeddingLengthPerLayer
        embeddingLengthOut
        vocabularySize, maxContextLength, rmsNormEps
        headSize(layer)
    }
    class Weights {
        <<record>>
        tokenEmbedding
        perLayerModelProjection
        perLayerProjectionNorm
        outputNorm, output
        LayerWeights[] layers
    }
    class LayerWeights {
        <<record>>
        attnNorm, query, key, value
        queryNorm, keyNorm, output, postAttnNorm
        ffnNorm, gate, up, down, postFfnNorm
        inputGate, projection, postProjectionNorm
        float outputScale
    }
    class State {
        <<static final>>
        input residual normed branch
        query key value attentionOut
        hidden hidden2
        perLayerInput perLayerGate
        pooled embedding
        ropeCos/Sin Full + Swa
        first[] end[] windowFirst[] windowEnd[]
        queryStride kvStride
        ropeUpTo(positions)
        clearHistory() no-op
    }
    class Scratch {
        <<GemmAttention.Scratch record>>
        valuesT scores outputT
        directQueries[] directScores[][]
    }
    GemmaEmbedding2 *-- Configuration
    GemmaEmbedding2 *-- Weights
    Weights *-- LayerWeights
    GemmaEmbedding2 ..> State : newState
    State *-- Scratch : attentionScratch
```

### 2c. Template and media subsystem

```mermaid
classDiagram
    direction LR
    class Towers { <<Gemma4.Towers, external>> }
    class MediaProjector~R~ { <<interface, external>> }
    class Media { <<sealed, external>> }
    class PromptWriter { <<external>> }
    class Specials { <<toknroll, external>> }
    class GemmaEmbedding2Template {
        <<package-private final>>
        VIDEO_FRAME_TOKENS = 140
        prefix = BOS, suffix = EOS
        bosSpelling
        withMedia(towers)
        encodeText(String) IntSequence
        encodeParts(parts, batchCapacity, sink)
        -media(out, media, projector)
        -tower(projector, medium)$
    }
    class GemmaEmbedding2 { <<final>> }
    Media <|-- Image
    Media <|-- Audio
    Media <|-- Video
    GemmaEmbedding2 *-- GemmaEmbedding2Template
    GemmaEmbedding2 o-- Towers : NONE until withMedia
    GemmaEmbedding2Template --> Towers : projector(Class), frames(140)
    Towers ..> MediaProjector : creates
    GemmaEmbedding2Template --> Specials : special-aware encode
    GemmaEmbedding2Template ..> PromptWriter : ids(), id(), media(), finish()
    GemmaEmbedding2Template ..> MediaProjector : media rows
```

---

## 3. Loading sequence

`general.architecture` is checked twice: by `Models` (provider routing) and by `loadModel`. The provider answers `retrieval(gguf)` with `EMBEDDING` unconditionally, and advertises the `media` companion (an mmproj GGUF).

```mermaid
sequenceDiagram
    participant M as Models (jinfer-chat)
    participant P as GemmaEmbedding2Provider
    participant G as GemmaEmbedding2
    participant L as ModelLoader
    participant T as GemmaEmbedding2Template
    participant G4 as Gemma4

    M->>P: retrieval(gguf)
    P-->>M: EMBEDDING
    M->>P: companionFiles()
    P-->>M: media to mmproj
    M->>P: loadEmbedder(channel, gguf, path, arena, companions, tokenizer)
    P->>G: loadModel(channel, gguf, arena, tokenizer)
    G->>G: check general.architecture == gemma-embedding2
    opt tokenizer is null
        G->>G: GGUFTokenizerLoader.fromGGUF(gguf)
    end
    G->>G: loadConfiguration (validation: causal=false, pooling_type=1, no shared KV, widths == head size)
    G->>L: loadTensors(channel, gguf, arena)
    G->>G: loadWeights (blk.N.* + token_embd, per_layer_model_proj, per_layer_proj_norm, output_norm, output)
    G->>T: new(tokenizer, bos, eos, Towers.NONE)
    G-->>P: GemmaEmbedding2
    opt companions has media
        P->>G: withMedia(mmproj, arena)
        G->>G4: loadTowers(mmproj, embeddingLength, arena)
        G-->>P: new model sharing weights, template.withMedia(towers)
    end
    P-->>M: LoadedEmbedder(model, tokenizer, prefix BOS, suffix EOS, embeddingLengthOut, minDim 128, file name, SEARCH_QUERY, DOCUMENT, template.encodeText)
```

---

## 4. Forward pass

There is **no autoregressive decoding and no KV cache**. `ingest` only *stages* input rows into `State.input`; the encoder runs over the **whole staged input** at once when an embedding is requested (`embed`, `projectEmbedding`) or over packed sequences (`embedAll`). Every row may attend to rows after it (bidirectional).

### 4.0 Global architecture

```mermaid
flowchart TB
    In["text ids / media rows / raw embeddings"] --> Stage["Staging into State.input [rows x dim]<br/>token rows scaled by sqrt(dim); media rows as projected"]
    Stage --> Seg["Segments: first/end per row, windowFirst/windowEnd<br/>RoPE positions restart per segment"]
    Seg --> Res["copy input to residual"]
    subgraph Stack["N layers (isSwa[l]: 5 sliding in 6 by default, 6th full)"]
        direction TB
        subgraph Block["per layer"]
            direction LR
            A["attention: pre-norm, QK-norm + NeoX RoPE,<br/>V-norm, unscaled bidirectional, post-norm, add"] --> F["feedForward: pre-norm, GELU-gated, post-norm, add"] --> PL["perLayerInput: gated projection of INPUT rows,<br/>post-norm, add, times outputScale"]
        end
    end
    Res --> Stack
    Stack --> FN["Final RMSNorm (outputNorm)"]
    FN --> Pool["Mean pool over the segment's rows"]
    Pool --> Proj["gemv output: dim to embeddingLengthOut"]
    Proj --> L2["L2 normalise"]
    L2 --> Out["embedding (state-owned view)"]
```

### 4a. `ingest` / `embed` — batch validation, input kinds, RoPE, layer loop

```mermaid
flowchart TB
    Emb["embed(state, parts, consumer)"] --> Ex1["state.exclusively; state.reset()"]
    Ex1 --> Enc["template.encodeParts(parts, batchCapacity, sink)"]
    Enc --> Cnt["needed += batch.count()<br/>stage only while needed &lt;= contextCapacity"]
    Cnt --> Stg
    In["ingest(state, Batch)"] --> Ex["state.exclusively"] --> Stg["stage(state, batch)"]
    Stg --> V{"0 &lt; rows &lt;= batchCapacity<br/>position + rows &lt;= contextCapacity"}
    V -->|no| Err["IllegalArgumentException"]
    V -->|yes| Sw{"batch.input()"}
    Sw -->|Tokens| T["stageTokens: ids in [0, vocab)<br/>Convert.gatherToF32(tokenEmbedding) into input @ position<br/>multiplyInPlace sqrt(dim)"]
    Sw -->|Embeddings| X["width must equal dim<br/>Convert.copyF32 into input (no scaling)"]
    Sw -->|Sequences| U["UnsupportedOperationException<br/>use embedAll"]
    T --> Adv["state.advance(batch)<br/>advanceContext(count, outputs)"]
    X --> Adv
    Cnt -->|"needed &gt; capacity at end"| Over["IllegalArgumentException<br/>positions needed vs available"]
    Cnt --> Fin["after last batch: embedding(state)"]
    Adv -.-> Fin
    Fin --> Fw["forward(state, lengths = rows)"]
    PE["projectEmbedding(state, idx, consumer)<br/>idx in [0, outputCount)"] --> Fin
    Fw --> Segs["per segment: first = start, end = start + length<br/>windowFirst = max(start, i - radius)<br/>windowEnd = min(end, i + radius + 1)<br/>radius = slidingWindow / 2"]
    Segs --> Rope["state.ropeUpTo(longest segment)<br/>fills full + swa cos/sin tables incrementally"]
    Rope --> Cp["Convert.copyF32 input to residual"]
    Cp --> Loop{"for l in 0..numberOfLayers"}
    Loop --> DB["attention(l); feedForward(l); perLayerInput(l)  see 4b"]
    DB --> Loop
    Loop -->|done| Pool["pool(state, 0, rows)  see 5"]
```

Position of a row inside its RoPE rotation is `r - first[r]`, so each packed segment restarts at 0.

### 4b. `decoderBlock` — one layer

There is no single `decoderBlock` method; a block is `attention(l)`, `feedForward(l)`, `perLayerInput(l)`. Each branch closes with `normThenAdd` (post-norm, add to residual, optionally scale). Full and sliding layers differ in head size, RoPE base and key range.

```mermaid
flowchart TB
    R0["residual [rows x dim]"] --> N1["Norms.rmsnormRows (attnNorm) into normed"]
    subgraph Attn["attention(l)"]
        direction TB
        N1 --> Q["gemm query"]
        N1 --> K["gemm key"]
        N1 --> V["gemm value"]
        Q --> QN["per row, per head: rmsnorm (queryNorm)<br/>RoPE.applyNeox, position r - first[r]"]
        K --> KN["per row, per kv head: rmsnorm (keyNorm)<br/>RoPE.applyNeox"]
        V --> VN["per row, per kv head: rmsnormNoWeight"]
        Sw{"isSwa[l]?"}
        Sw -->|yes| RS["cos/sin Swa, headSizeSwa, ropeThetaSwa<br/>range = windowFirst, windowEnd"]
        Sw -->|no| RF["cos/sin Full, headSizeFull, ropeThetaFull<br/>range = first, end"]
        QN --> GA["GemmAttention.prefill (unscaled, no KV commit: K/V stay in scratch)"]
        KN --> GA
        VN --> GA
        RS --> GA
        RF --> GA
        GA --> WO["gemm output (width = heads x headSize) into branch"]
    end
    WO --> NA["normThenAdd: rmsnorm (postAttnNorm); residual += branch"]
    NA --> N2["rmsnormRows (ffnNorm) into normed"]
    subgraph Ffn["feedForward(l) — GELU-gated"]
        direction TB
        N2 --> G["gemm gate into hidden"]
        N2 --> U["gemm up into hidden2"]
        G --> GM["Activations.geluMultiply(hidden, hidden2)"]
        U --> GM
        GM --> D["gemm down into branch"]
    end
    D --> NF["normThenAdd: rmsnorm (postFfnNorm); residual += branch"]
    subgraph PLI["perLayerInput(l)"]
        direction TB
        NF --> PM["gemm perLayerModelProjection slice l on INPUT rows<br/>into perLayerInput [rows x plDim]"]
        NF --> IG["gemm inputGate on residual into perLayerGate"]
        PM --> SC["times 1/sqrt(dim); rmsnorm (perLayerProjectionNorm)"]
        SC --> GE["geluMultiply(perLayerGate, perLayerInput)"]
        IG --> GE
        GE --> PJ["gemm projection into branch"]
    end
    PJ --> NP["normThenAdd: rmsnorm (postProjectionNorm); residual += branch; residual *= outputScale"]
```

Details worth knowing:

- No KV cache: keys and values of the staged input live only in `State.key/value` and are recomputed by every `embed`.
- Attention is **unscaled** (no `1/sqrt(headSize)`); V is RMS-normed without weight; GQA via `h / (heads / kvHeads)`, with `numberOfKeyValueHeads` per layer.
- The per-layer input is projected from the **input rows** (`state.input`), not from the residual; there is no per-layer token table.
- `State` strides (`queryStride`, `kvStride`) are the widest layer's; `attentionOut` is re-viewed at each layer's width.

### 4c. `GemmAttention.prefill` — per-query key ranges

```mermaid
flowchart TB
    In["prefill(q, k, v, out, rows, heads, kvHeads, headSize, first[], end[], scratch)"] --> Loop["walk sequences: s..e where first[e] == e starts the next"]
    Loop --> Dec{"attended keys &lt;= DIRECT_KEYS (384) x queries?"}
    Dec -->|yes| DQ["append queries to directQueries"]
    Dec -->|no| TR["once: Ops.transposeCopy(v) into valuesT"]
    TR --> Blk["per block of up to 1024 queries<br/>k0 = first[q0], width = end[last] - k0"]
    Blk --> Hd["for each head h (sequential)"]
    Hd --> S1["MatMul.mm: scores = q . k^T"]
    S1 --> S2["Parallel.forLoop over queries:<br/>zero outside [lo, hi), softmaxInPlace inside"]
    S2 --> S3["MatMul.mm: outputT = valuesT . scores^T"]
    S3 --> Tp["Ops.transposeCopy outputT into out"]
    DQ --> Dir["after the loop: Parallel.forLoop over heads x directQueries"]
    Dir --> D1["Ops.dot per key; max; exp; sum"]
    D1 --> D2["Ops.saxpyInPlace weighted values into out"]
```

---

## 5. Pooling and `embedAll`

Every retained output of the staged input projects the **same** vector (`projectEmbedding` ignores which output index, only range-checks it). `embedAll` packs whole sequences, each attending only to itself.

```mermaid
sequenceDiagram
    participant C as caller
    participant G as GemmaEmbedding2
    participant S as State

    C->>G: embedAll(state, Sequences, sink)
    G->>G: requireComplete (all lengths positive, sum equals token count)
    G->>G: each length within contextCapacity
    G->>S: exclusively then reset()
    loop groups of consecutive sequences that fit contextCapacity
        G->>S: stageTokens(ids, at, rows, position 0)
        G->>G: forward(state, lengths of the group)
        loop each sequence in group
            G->>G: pool(state, start, length)
            Note over G: rmsnormRows (outputNorm), mean of rows, gemv (output), L2 normalise
            G-->>C: sink.accept(embedding view)
        end
    end
```

```mermaid
flowchart LR
    Res["residual rows [start, start+n)"] --> FN["Norms.rmsnormRows (outputNorm) into normed"]
    FN --> Sum["pooled = sum of rows (Ops.addInPlace)"]
    Sum --> Mean["pooled *= 1/n"]
    Mean --> Gv["MatMul.gemv(output, pooled) into embedding [embeddingLengthOut]"]
    Gv --> Sq["Norms.sumOfSquares"]
    Sq --> L2["embedding *= 1/sqrt(squares), or 0 if squares == 0"]
```

Template framing (`encodeText` / `encodeParts`): BOS + tokens + EOS, unless the text begins with the BOS spelling (reference processor behaviour). Media parts are wrapped in `<|image>`/`<image|>` or `<|audio>`/`<audio|>`; a video is its frames, each an image block (140-token budget). Without the matching tower, `UnsupportedOperationException` ("load the mmproj with withMedia"). Task prompts (`searchQuery`, `question`, `claim`, `codeQuery`, `document`, `classification`, `clustering`, `similarity`) are static string helpers applied to text only.

---

## 6. State layout

```mermaid
flowchart TB
    subgraph State["GemmaEmbedding2.State (per context, all sized by contextCapacity rows)"]
        direction TB
        subgraph rowsb["Row buffers F32"]
            A["input [rows x dim]  staged rows (persist across ingest)"]
            B["residual · normed · branch [rows x dim]"]
            C["query [rows x queryStride] · key, value [rows x kvStride] · attentionOut"]
            D["hidden · hidden2 [rows x feedForwardLength]"]
            E["perLayerInput · perLayerGate [rows x embeddingLengthPerLayer]"]
        end
        subgraph outb["Output"]
            F["pooled [1 x dim] · embedding [1 x embeddingLengthOut]"]
        end
        subgraph rp["RoPE and ranges"]
            G["ropeCos/Sin Full [rows x headSizeFull/2] and Swa [rows x headSizeSwa/2]<br/>grown on demand by ropeUpTo"]
            H["first[] end[] windowFirst[] windowEnd[] (rebuilt per forward)"]
        end
        subgraph sc["GemmAttention.Scratch"]
            I["valuesT · scores · outputT · directQueries · directScores per thread"]
        end
    end
```

- There is **no checkpoint codec** and no persistent per-layer state: `clearHistory()` is a no-op (staged rows past the cursor are overwritten by the next ingest), and the provider does not override `checkpointCodec`.
- `newState(ctx, batch)` makes a cross-thread arena owned by the state (closed if construction fails); `newState(ctx, batch, arena)` borrows the caller's.
- `State` rejects `contextCapacity > maxContextLength`.
- `ingest` advances the cursor with `advanceContext(count, outputs)`; `embed` and `embedAll` call `reset()` first, so each call is independent.
