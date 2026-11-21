# jinfer-maple — class architecture

Package `com.qxotic.jinfer.models.maple` (4 classes, ~750 lines). Provider for GGUF architecture `maple`: a
**sparse MoE transformer** (text only) with **hybrid sliding-window / global attention** (per-layer mask), per-head
QK RMS-norm, softmax-routed experts with clamped SwiGLU, and a **pure-MoE FFN** (no dense or shared expert).

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends;
classes marked `<<external>>` live outside this package.

1. [Position in the stack](#1-position-in-the-jinfer-stack)
2. [Class diagrams](#2-class-diagrams) — 2a contracts, 2b model internals
3. [Loading sequence](#3-loading-sequence)
4. [Forward pass](#4-forward-pass) — 4a `ingest`, 4b `decoderBlock` (+ MoE)
5. [State and checkpoint layout](#5-state-and-checkpoint-layout)

| Class | Role |
|---|---|
| `Maple` | The model: `LanguageModel` impl + nested `Configuration`, `Weights`, `LayerWeights`, `State`, loaders, forward pass |
| `MapleProvider` | `ModelProvider` SPI for `maple`; builds `LoadedModel` with stops `<\|im_end\|>`, `<\|endoftext\|>` |
| `MapleChatTemplate` | `ChatTemplate`: reply side only (`<think>`, `<tool_call>` spans, grammar); `encode` throws `UnsupportedConversation` so the whole-render (jinja) path frames prompts |
| `MapleCheckpointCodec` | `CheckpointCodec<State>`: KV rows; sliding layers go through a ring-buffer span |

---

## 1. Position in the jinfer stack

```mermaid
flowchart TB
    Callers["Callers: cli / server / langchain4j / spring-ai"]
    subgraph chat["jinfer-chat"]
        Models["Models (ServiceLoader)"]
        MP["ModelProvider"]
        LM_["LoadedModel"]
        CT["ChatTemplate / ReplyLanguage"]
    end
    subgraph core["jinfer-core"]
        LangModel["LanguageModel"]
        CS["ContextState"]
        CC["CheckpointCodec"]
    end
    subgraph mp["jinfer-maple"]
        P["MapleProvider"]
        M["Maple"]
        T["MapleChatTemplate"]
        C["MapleCheckpointCodec"]
    end
    subgraph kern["jinfer-kernels"]
        K["MatMul · Norms · RoPE · FlashAttention<br/>Moe · Activations · Ops · Convert · KvTransfer · ModelLoader"]
    end
    Tok["toknroll Tokenizer"]

    Callers --> Models --> P
    P -.->|implements| MP
    P --> M
    P --> T
    P -->|produces| LM_
    M -.->|implements| LangModel
    T -.->|implements| CT
    C -.->|extends| CC
    M --> C
    M --> K
    M --> Tok
```

---

## 2. Class diagrams

### 2a. Contracts and relations

```mermaid
classDiagram
    direction LR
    class ModelProvider { <<interface, external>> }
    class LanguageModel~C,W,S~ { <<interface, external>> }
    class ChatTemplate { <<interface, external>> }
    class ContextState { <<abstract, external>> }
    class CheckpointCodec~S~ { <<abstract, external>> }
    class MapleProvider {
        <<final>>
        architectures() Set
        loadLanguage(...) LoadedModel
    }
    class Maple { <<final>> }
    class MapleChatTemplate {
        <<final>>
        -Tokenizer tokenizer
        -Spans spans (lazy)
        encode() throws UnsupportedConversation
        parser(tokenizer)
        constrainedReply(grammar, calls)
    }
    class MapleCheckpointCodec {
        <<package-private>>
        -bytesPerPosition
        sizeOf(positions)
        transfer(State, from, to, blob, capture)
    }
    class State { <<Maple.State>> }

    ModelProvider <|.. MapleProvider
    LanguageModel <|.. Maple
    ChatTemplate <|.. MapleChatTemplate
    ContextState <|-- State
    CheckpointCodec <|-- MapleCheckpointCodec
    MapleProvider ..> Maple : loadModel
    MapleProvider ..> MapleChatTemplate : new(tokenizer)
    Maple ..> MapleCheckpointCodec : checkpointCodec()
    MapleCheckpointCodec --> Configuration
    MapleCheckpointCodec ..> State : reads/writes
    Maple *-- State : newState
    class Configuration { <<Maple.Configuration>> }
```

### 2b. Model internals

```mermaid
classDiagram
    direction TB
    class Maple {
        <<final>>
        -Configuration configuration
        -Tokenizer tokenizer
        -Weights weights
        +loadModel(path/channel, arena)$
        ~loadConfiguration(gguf, tokenizer)$
        ~loadWeights(tensors, config)$
        +ingest(State, Batch)
        +logits(State, output)
        -forward(State, Batch)
        -forward(State, ids, startPos, rows)
        -embed()
        -attention(State, layer, startPos, rows)
        -commitKv()
        -feedForward(State, layer, rows)
        -projectLogits()
    }
    class Configuration {
        <<record>>
        embeddingLength, numberOfLayers
        numberOfHeads, numberOfKeyValueHeads
        headSize, ropeDimension, ropeTheta
        slidingWindow (power of 2)
        boolean[] slidingMask
        expertCount, expertUsedCount
        expertFeedForwardLength, expertWeightsScale
        float[] swigluClamp per layer
        queryDim() kvDim()
        isSliding(layer)
        kvCachePositions(layer, cap)
        kvCacheIndex(layer, pos)
        attentionStart(layer, pos)
    }
    class Weights {
        <<record>>
        tokenEmbeddings
        LayerWeights[] layers
        outputNorm, outputWeight
        RoPE.Schedule rope
    }
    class LayerWeights {
        <<record>>
        attnNorm
        q k v o
        qNorm kNorm
        ffnNorm
        router
        gateExperts[] upExperts[] downExperts[]
    }
    class State {
        <<static final>>
        residual normed branch
        query attnOut head logits
        ropeCos ropeSin
        router gather expertOut moeOut
        hidden hidden2
        expertCounts topExperts topWeights
        Moe.Routing routing
        keyCache[] valueCache[] F16
        batchK[] batchV[] F32 per layer
        decodeScratch
        clearHistory() no-op
    }
    Maple *-- Configuration
    Maple *-- Weights
    Weights *-- LayerWeights
    Maple ..> State : newState
    Configuration ..|> ContextConfiguration
```

---

## 3. Loading sequence

```mermaid
sequenceDiagram
    participant M as Models (jinfer-chat)
    participant P as MapleProvider
    participant Q as Maple
    participant G as GGUFTokenizerLoader
    participant L as ModelLoader
    participant T as MapleChatTemplate

    M->>P: loadLanguage(channel, gguf, path, arena, companions, tokenizer)
    P->>Q: loadModel(channel, gguf, arena, tokenizer)
    opt tokenizer == null
        Q->>G: fromGGUF(gguf)
    end
    Q->>Q: loadConfiguration: maple.* keys<br/>key_length == value_length, require expert_weights_norm and softmax gating (1)
    Q->>L: loadTensors(channel, gguf, arena)
    L-->>Q: Map name → MemoryView
    Q->>Q: loadWeights: per layer attn, qk-norm, router, experts
    Q-->>P: Maple
    P->>P: eos id, stops = {eos, im_end, endoftext}
    P->>T: new MapleChatTemplate(tokenizer)
    P-->>M: LoadedModel(model, tok, chat_template source, stops, seed, template, NONE)
```

---

## 4. Forward pass

`ingest` → `state.exclusively(forward)`. Maple is **text-only, one sequence**: `Sequences` and `Embeddings` batches throw `UnsupportedOperationException`.

### 4.0 Global architecture

```mermaid
flowchart TB
    Tok["token ids [rows]"] --> Emb["Token embedding<br/>gather → F32 residual stream [rows × dim]"]
    subgraph Stack["Trunk: N decoder blocks, attention kind chosen per layer by slidingMask[]"]
        direction TB
        subgraph Block["decoderBlock (pre-norm residual ×2)"]
            direction LR
            A["RMSNorm → GQA attention + QK-norm"] --> Add1(("+")) --> F["RMSNorm → MoE FFN<br/>softmax router, top-k, clamped SwiGLU"] --> Add2(("+"))
        end
        SW["sliding layer: NeoX RoPE, window ring KV"]
        GL["global layer: no RoPE, full KV"]
        A -.- SW
        A -.- GL
    end
    Emb --> Stack
    Stack --> FN["Final RMSNorm (outputNorm)"]
    FN --> Head["LM head gemv(outputWeight)"]
    Head --> Logits["logits [vocab]"]
    Stack -.-> St[("State: F16 KV per layer<br/>carried across ingest calls")]
```

### 4a. `ingest`

```mermaid
flowchart TB
    In["ingest(state, Batch)"] --> Ex["state.exclusively"]
    Ex --> V{"0 < rows <= batchCapacity<br/>position + rows <= contextCapacity"}
    V -->|no| Err["IllegalArgumentException"]
    V -->|yes| Sw{"batch.input()"}
    Sw -->|Tokens| Chk["every id in [0, vocab)"]
    Sw -->|"Sequences / Embeddings"| X["UnsupportedOperationException"]
    Chk --> Rope["RoPE.fill(ropeCos, ropeSin, startPos, rows, ropeDim/2)<br/>ONCE per batch"]
    Rope --> Emb["embed: Convert.copyToF32 token rows → residual"]
    Emb --> Loop{"for layer in 0..numberOfLayers"}
    Loop --> A["attention(layer)  → 4b"]
    A --> F["feedForward(layer)  MoE → 4c"]
    F --> Loop
    Loop -->|done| Adv["state.advance(batch)"]
    Adv -.->|"later: logits(state, i)"| Out["rmsnorm(outputNorm, row) → MatMul.gemv(outputWeight) → logits"]
```

### 4b. `decoderBlock` — attention half

There is no `decoderBlock` method; a block is `attention(layer)` + `feedForward(layer)` (pre-norm residual each).
`isSliding(layer)` (from the GGUF `sliding_window_pattern`) drives **three** differences: RoPE, window, cache size.

```mermaid
flowchart TB
    R0["residual"] --> N1["Norms.rmsnormRows → normed (attnNorm)"]
    N1 --> Q["gemm q → query"]
    N1 --> K["gemm k → batchK[layer]"]
    N1 --> V["gemm v → batchV[layer]"]
    Q --> QN["per head: RMSNorm(qNorm)"]
    K --> KN["per head: RMSNorm(kNorm)"]
    QN --> S1{"isSliding(layer)?"}
    KN --> S1
    S1 -->|yes| RP["RoPE.applyNeox on q and k"]
    S1 -->|"no (global layer)"| NP["no RoPE (NoPE)"]
    RP --> FA{"rows > 1?"}
    NP --> FA
    FA -->|yes| Pre["FlashAttention.slidingWindowPrefill<br/>window = sliding ? slidingWindow : 0"]
    FA -->|no| Dec["FlashAttention.flashDecode<br/>start = attentionStart(layer, pos)"]
    Pre --> WO["gemm o → branch"]
    Dec --> WO
    WO --> Add["Ops.addRows: residual += branch"]
    Add --> Commit["commitKv: f32ToF16 per row into<br/>slot = kvCacheIndex(layer, pos)<br/>(sliding: pos & (window-1); global: pos)"]
```

`commitKv` runs **after** attention (batchK/V are per layer, so nothing is overwritten); the attention call sees the chunk's own K/V from the F32 batch buffers plus earlier rows from the cache.

### 4c. `decoderBlock` — MoE feed-forward half

```mermaid
flowchart TB
    R["residual"] --> N2["rmsnormRows → normed (ffnNorm)"]
    N2 --> Rt["gemm router → router [rows × experts]"]
    Rt --> Sm["Ops.softmaxInPlace per row"]
    Sm --> Top["Moe.selectTopK(topK) → topExperts / topWeights / expertCounts"]
    Top --> Nrm["per row: w *= expertWeightsScale / Σw (renormalise)"]
    Nrm --> Disp["Moe.dispatch(routing, normed → gather → expertOut → moeOut)"]
    subgraph Exp["per selected expert e (count rows)"]
        direction TB
        G["gemm gateExperts[e] → hidden"]
        U["gemm upExperts[e] → hidden2"]
        Cl["clamp: hidden ≤ c<br/>hidden2 ∈ [-c, c]<br/>c = swigluClamp[layer]"]
        Si["siluMultiplyRows(hidden, hidden2)"]
        D["gemm downExperts[e] → out"]
        G --> Cl
        U --> Cl --> Si --> D
    end
    Disp --> Exp
    Exp --> Add["weighted scatter-add → moeOut<br/>Ops.addRows: residual += moeOut"]
```

---

## 5. State and checkpoint layout

```mermaid
flowchart TB
    subgraph State["Maple.State (per context)"]
        direction TB
        subgraph scratch["Scratch sized by batchCapacity"]
            A["residual · normed · branch · query · attnOut · head · logits"]
            B["ropeCos/Sin · router · gather · expertOut · moeOut · hidden · hidden2"]
            C["expertCounts · topExperts · topWeights · Moe.Routing"]
            D["batchK[l] · batchV[l] F32 (one per layer)"]
        end
        subgraph persist["Persistent per layer"]
            G["global layer: keyCache/valueCache F16 [contextCapacity × kvDim]"]
            S["sliding layer: F16 ring [min(contextCapacity, slidingWindow) × kvDim]"]
        end
    end
    subgraph Blob["MapleCheckpointCodec blob"]
        B1["per layer: K rows then V rows for [from, to)<br/>= positions × 2 × 2 B × layers × kvDim"]
    end
    G -->|"transfer"| B1
    S -->|"KvTransfer.ringSpan"| B1
```

- `clearHistory()` is a no-op: stale rows are masked by attention and overwritten on ingest.
- Sliding layers are rings; slot = `position & (slidingWindow - 1)` (hence the power-of-two check in `Configuration`).
- `newState(ctx, batch)` owns a cross-thread arena; `newState(ctx, batch, arena)` borrows one.
