# jinfer-qwen3 — class architecture

Package `com.qxotic.jinfer.models.qwen3` (4 classes, ~1170 lines). It is the `jinfer-<model>` provider for GGUF
architecture `qwen3` — **not a chat model**: it serves the Qwen3 *retrieval family* (Qwen3-Embedding: last-token pooled
vectors; Qwen3-Reranker: a yes/no judge) over one dense backbone. `loadLanguage` throws `IncompatibleModelException`;
generative Qwen models live in `jinfer-qwen35`.

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends;
classes marked `<<external>>` live outside this package.

1. [Position in the stack](#1-position-in-the-jinfer-stack)
2. [Class diagrams](#2-class-diagrams) — 2a contracts, 2b model internals
3. [Loading sequences](#3-loading-sequences)
4. [Forward pass](#4-forward-pass) — 4a `ingest`, 4b `decoderBlock`
5. [Heads: embedding, targeted logit, reranker](#5-heads)
6. [State and checkpoint layout](#6-state-and-checkpoint-layout)

| Class | Role |
|---|---|
| `Qwen3` | The model: `LanguageModel` + `EmbeddingModel` impl with nested `Configuration`, `Weights`, `LayerWeights`, `State`; the whole forward pass and the three heads |
| `Qwen3Provider` | `ModelProvider` SPI: `retrieval(gguf)` (embedding vs reranking), `loadEmbedder`, `loadReranker` |
| `Qwen3Reranker` | `Reranker<State>`: prompt framing + yes/no scoring on one shared query prefix |
| `Qwen3CheckpointCodec` | `CheckpointCodec<State>`: serialises KV rows only (pure attention, no recurrent state) |

---

## 1. Position in the jinfer stack

```mermaid
flowchart TB
    Callers["Callers: cli / server / langchain4j / spring-ai"]
    subgraph chat["jinfer-chat"]
        Models["Models (ServiceLoader)<br/>loadEmbedder / loadReranker"]
        MP["ModelProvider"]
        LE["LoadedEmbedder"]
        LR["LoadedReranker"]
    end
    subgraph core["jinfer-core"]
        LM["LanguageModel / EmbeddingModel / Reranker"]
        CS["ContextState"]
        CC["CheckpointCodec"]
        B["Batch (Tokens | Sequences)"]
    end
    subgraph q3["jinfer-qwen3"]
        P["Qwen3Provider"]
        Q["Qwen3"]
        R["Qwen3Reranker"]
        C["Qwen3CheckpointCodec"]
    end
    subgraph kern["jinfer-kernels"]
        K["MatMul · Norms · RoPE · FlashAttention<br/>Activations · Convert · Ops · ModelLoader"]
    end
    Tok["toknroll Tokenizer"]

    Callers --> Models --> P
    P -.->|implements| MP
    P -->|produces| LE
    P -->|produces| LR
    Q -.->|implements| LM
    R -.->|implements| LM
    C -.->|extends| CC
    Q --> K
    Q --> Tok
    LE --> Q
    LR --> R
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
    class Reranker~S~ { <<interface, external>> }
    class ContextState { <<abstract, external>> }
    class CheckpointCodec~S~ { <<abstract, external>> }
    class Qwen3Provider {
        <<final>>
        architectures() Set
        retrieval(gguf) Optional
        loadLanguage() throws
        loadEmbedder(...) LoadedEmbedder
        loadReranker(...) LoadedReranker
    }
    class Qwen3 { <<final>> }
    class Qwen3Reranker {
        <<package-private>>
        -imStart imEnd thinkOpen thinkClose
        -yes no
        prefix(instruction, query) Batch
        document(text) Batch
        scoreAll(state, instr, query, docs, sink)
    }
    class Qwen3CheckpointCodec { <<package-private>> }
    class State { <<Qwen3.State>> }

    ModelProvider <|.. Qwen3Provider
    LanguageModel <|.. Qwen3
    EmbeddingModel <|.. Qwen3
    Reranker <|.. Qwen3Reranker
    ContextState <|-- State
    CheckpointCodec <|-- Qwen3CheckpointCodec
    Qwen3Provider ..> Qwen3 : loadModel
    Qwen3Provider ..> Qwen3Reranker : new(model)
    Qwen3Reranker --> Qwen3 : model
    Qwen3 ..> Qwen3CheckpointCodec : checkpointCodec()
    Qwen3CheckpointCodec ..> State : reads/writes
    Qwen3 *-- State : newState
```

### 2b. Model internals — `Qwen3` and its nested types

```mermaid
classDiagram
    direction TB
    class Qwen3 {
        <<final>>
        -Configuration configuration
        -Tokenizer tokenizer
        -Weights weights
        +loadModel(path/channel, arena)$
        +ingest(State, Batch)
        +logits(State, output) MemoryView
        +logit(State, token) float
        +projectEmbedding(State, idx, consumer)
        -forward(State, Batch)
        -cutPieces(seqLen, cs, n, State)$
        -forward(State, ids, startPos, seqLen, nPieces)
        -embedTokens()
        -attention() attentionProject() attentionFinish()
        -headNormRope()
        -feedForward()
        -commitKv()
    }
    class Configuration {
        <<record>>
        embeddingLength, numberOfLayers
        numberOfHeads, numberOfKeyValueHeads
        headSize, ropeDim, ropeTheta
        queryDim, kvDim, kvMul
        hiddenDim, rmsNormEps
        vocabularySize, maxContextLength
    }
    class Weights {
        <<record>>
        tokenEmbeddings
        LayerWeights[] layers
        finalNorm
        RoPE.Schedule rope
        wcls (tied if absent)
    }
    class LayerWeights {
        <<record>>
        attnNorm
        wq wk wv wo
        attnQNorm attnKNorm
        ffnNorm
        w1 w2 w3
    }
    class State {
        <<static final>>
        residual normed branchOut
        attnOut query hidden hidden2
        logits pooled
        ropeCos ropeSin
        batchK batchV
        keyCache[] valueCache[] F16
        decodeScratch
        posOf[] pieceRow0[] pieceLen[]
        pieceKv[] piecePrior[]
        clearHistory() no-op
    }
    Qwen3 *-- Configuration
    Qwen3 *-- Weights
    Weights *-- LayerWeights
    Qwen3 ..> State : newState
    Configuration ..|> ContextConfiguration
```

---

## 3. Loading sequences

The same GGUF architecture (`qwen3`) backs two faces. `Models` calls `retrieval(gguf)` first and refuses the wrong entry point **before** any weight is read.

```mermaid
sequenceDiagram
    participant M as Models (jinfer-chat)
    participant P as Qwen3Provider
    participant Q as Qwen3
    participant L as ModelLoader
    participant R as Qwen3Reranker

    M->>P: retrieval(gguf)
    alt qwen3.pooling_type present
        P-->>M: 4 → RERANKING, else EMBEDDING
    else absent
        P-->>M: general.name/basename contains rerank / embed, else empty
    end
    alt loadEmbedder
        M->>P: loadEmbedder(channel, gguf, path, arena, tokenizer)
        P->>Q: loadModel(channel, gguf, arena, tokenizer)
        Q->>L: loadTensors
        Q-->>P: Qwen3
        P->>P: eos = id of endoftext (last-token pooling wants a trailing EOS)
        P-->>M: LoadedEmbedder(model, tokenizer, prefix empty, suffix eos, dim, minDim 32, instructed-query prefix)
    else loadReranker
        M->>P: loadReranker(...)
        P->>Q: loadModel(...)
        P->>R: new Qwen3Reranker(model) (resolves im_start/end, think, single-token yes/no)
        P-->>M: LoadedReranker(model, reranker)
    else loadLanguage
        P--xM: IncompatibleModelException (use Qwen 3.5)
    end
```

---

## 4. Forward pass

`ingest(state, batch)` → `state.exclusively(forward)`. There is **no recurrent state**: every layer is attention + SwiGLU FFN, and the only thing carried across calls is the per-layer F16 KV cache.

### 4.0 Global architecture

```mermaid
flowchart TB
    Tok["token ids [rows]"] --> Emb["Token embedding<br/>gather → F32 residual stream [rows × dim]"]
    subgraph Stack["Dense trunk: N identical decoder blocks"]
        direction TB
        subgraph Block["decoderBlock (pre-norm residual ×2)"]
            direction LR
            A["RMSNorm → GQA attention<br/>QK-norm + NeoX RoPE, causal"] --> Add1(("+")) --> F["RMSNorm → SwiGLU FFN"] --> Add2(("+"))
        end
    end
    Emb --> Stack
    Stack --> FN["Final RMSNorm (finalNorm)"]
    FN --> H1["LM head → logits [vocab]"]
    FN --> H2["L2-normalised pooled vector<br/>(embedding, last-token pooling)"]
    FN --> H3["targeted logit(yes), logit(no)<br/>(reranker)"]
    Stack -.-> St[("State: F16 KV cache per layer<br/>carried across ingest calls")]
```

### 4a. `ingest` — batch validation, packed pieces, layer loop

```mermaid
flowchart TB
    In["ingest(state, Batch)"] --> Ex["state.exclusively"]
    Ex --> V{"count in (0, batchCapacity]<br/>position + count <= contextCapacity"}
    V -->|no| Err["IllegalArgumentException"]
    V -->|yes| Sw{"batch.input()"}
    Sw -->|Tokens| T["one piece: row0=0, len=n, kvStart=0<br/>prior=position, posOf[r]=position+r"]
    Sw -->|Sequences| S["requireCovers; cutPieces()<br/>one piece per packed sequence<br/>posOf restarts at 0 per sequence"]
    Sw -->|Embeddings| X["UnsupportedOperationException"]
    T --> RT["requireTokens: ids in [0, vocab)"]
    S --> RT
    RT --> Rope["RoPE.fill(ropeCos, ropeSin, posOf)<br/>ONCE per batch, per-row positions"]
    Rope --> Emb["embedTokens: Convert.gatherToF32 → residual"]
    Emb --> Loop{"for l in 0..numberOfLayers"}
    Loop --> DB["decoderBlock l<br/>attention(l) then feedForward(l)  → see 4b"]
    DB --> Loop
    Loop -->|done| Adv["state.advance(batch)<br/>advanceContext(count, outputs)"]
    Adv -.->|"later"| Heads["logits / projectEmbedding / logit  → see 5"]
```

Packed sequences (`Batch.Input.Sequences`): `cutPieces` groups the chunk rows into per-sequence *pieces* `{row0, len, kvStart, prior}` so sequences in one packed stream never attend to each other, even when one spans several chunks.

### 4b. `decoderBlock` — one layer

Qwen3 has no single `decoderBlock` method; a block is `attention(l)` followed by `feedForward(l)`, both pre-norm residual.

```mermaid
flowchart TB
    R0["residual [rows × dim]"] --> N1["Norms.rmsnormRows → normed (attnNorm)"]
    subgraph Attn["attention(l)"]
        direction TB
        N1 --> Q["gemm wq → query"]
        N1 --> K["gemm wk → batchK"]
        N1 --> V["gemm wv → batchV"]
        Q --> QN["headNormRope: per-head RMSNorm (attnQNorm)<br/>then NeoX RoPE"]
        K --> KN["headNormRope: per-head RMSNorm (attnKNorm)<br/>then NeoX RoPE"]
        KN --> CK["commitKv: f32ToF16Rows<br/>batchK/V → keyCache/valueCache[l] @ startPos"]
        V --> CK
        QN --> Path{"nPieces == 1<br/>and pieceKv[0] == 0?"}
        CK --> Path
        Path -->|"yes, seqLen > 1"| P1["FlashAttention.slidingWindowPrefill (window 0 = full causal)"]
        Path -->|"yes, seqLen == 1"| P2["FlashAttention.flashDecode"]
        Path -->|"no (packed)"| P3["per piece: slice q/out/batchK/V by rows<br/>+ prior rows sliced from cache → slidingWindowPrefill"]
        P1 --> AO["attnOut"]
        P2 --> AO
        P3 --> AO
        AO --> WO["gemm wo → branchOut"]
    end
    WO --> A1["Ops.addRows: residual += branchOut"]
    A1 --> N2["Norms.rmsnormRows → normed (ffnNorm)"]
    subgraph Ffn["feedForward(l) — SwiGLU"]
        direction TB
        N2 --> G["gemm w1 → hidden (gate)"]
        N2 --> U["gemm w3 → hidden2 (up)"]
        G --> SM["Activations.siluMultiply(hidden, hidden2)"]
        U --> SM
        SM --> D["gemm w2 → normed (down)"]
    end
    D --> A2["Ops.addRows: residual += normed"]
```

Details worth knowing:

- `commitKv` runs **inside** the layer: `batchK/batchV` are single buffers reused by every layer, so deferring the commit would write the last layer's values into every cache.
- Scale is `1/sqrt(headSize)`; GQA via `kvMul = heads / kvHeads`; `queryDim = heads × headSize` and `kvDim = kvHeads × headSize` are separate configuration fields.
- The old `headNormScale` multiply is dropped (hard-coded 1.0 — an identity).

---

## 5. Heads

All three read the retained residual row `lastBatchSize - outputCount + output` and apply `finalNorm`.

```mermaid
flowchart LR
    Res["residual row"] --> FN["Norms.rmsnorm(finalNorm) → normed"]
    FN --> LG["logits(state, i)<br/>MatMul.gemv(wcls) → full-vocab logits"]
    FN --> EMB["projectEmbedding<br/>L2-normalise → pooled (reused buffer)<br/>last pooling: last row of each sequence"]
    FN --> TG["logit(state, token)<br/>one row of wcls · normed → float<br/>(two dot products, no vocab matmul)"]
    TG --> Yes["logit(yes)"]
    TG --> No["logit(no)"]
    Yes --> Soft["softmax over {yes, no} → score"]
    No --> Soft
```

### Reranker flow — one shared prefix, many documents

```mermaid
sequenceDiagram
    participant C as caller
    participant R as Qwen3Reranker
    participant Q as Qwen3
    participant S as State

    C->>R: scoreAll(state, instruction, query, docs, sink)
    R->>S: exclusively, state.reset()
    R->>Q: ingest(prefix: system turn + Instruct, Query and Document header)
    R->>R: prefixLength = state.position()
    loop each document
        R->>R: check prefixLength + frame <= contextCapacity
        R->>S: resumeAt(prefixLength)  (KV of the prefix is kept)
        R->>Q: ingest(" doc" + im_end + assistant + empty think block)
        R->>Q: logit(yes), logit(no)
        R-->>C: sink.accept(softmax(yes, no)[yes])
    end
```

`ingest` chunks each framed batch with `Batch.prepare(..., batchCapacity)`. `yes`/`no` must each be a single token (checked at load).

---

## 6. State and checkpoint layout

```mermaid
flowchart TB
    subgraph State["Qwen3.State (per context)"]
        direction TB
        subgraph scratch["Scratch sized by batchCapacity"]
            A["residual · normed · branchOut · attnOut · query"]
            B["hidden · hidden2 · batchK · batchV · ropeCos/Sin"]
            C["logits [1 × vocab] · pooled [1 × dim] · decodeScratch"]
            D["posOf[] · pieceRow0/Len/Kv/Prior[]"]
        end
        subgraph persist["Persistent per layer"]
            KV["keyCache[l], valueCache[l]<br/>F16 [contextCapacity × kvDim]"]
        end
    end
    subgraph Blob["Qwen3CheckpointCodec blob"]
        B1["per layer in order: K rows then V rows for positions [from, to)<br/>= positions × bytesPerPosition"]
    end
    KV -->|"transfer(capture)"| B1
    B1 -->|"transfer(restore)"| KV
```

- `clearHistory()` is a no-op: stale rows past the cursor are masked by attention and overwritten on ingest.
- `newState(ctx, batch)` makes a cross-thread arena owned by the state; `newState(ctx, batch, arena)` borrows the caller's.
- `State` rejects `contextCapacity > maxContextLength`.
