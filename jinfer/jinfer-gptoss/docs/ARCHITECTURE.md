# jinfer-gptoss — class architecture

Package `com.qxotic.jinfer.models.gptoss` (4 classes, ~1050 lines). It is the `jinfer-<model>` provider for GGUF
architecture `gpt-oss` (OpenAI GPT-OSS): a **text-only, generative mixture-of-experts** model. Every layer is
attention (GQA, biases, per-head attention sinks, alternating sliding-window / full) followed by a top-k MoE FFN
(router, clamped SwiGLU experts with biases). There is no vision, audio or speculative subsystem in this module; the
Harmony reply grammar (analysis / commentary / final / call channels) lives in `GptOssChatTemplate`.

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends;
classes marked `<<external>>` live outside this package.

1. [Position in the stack](#1-position-in-the-jinfer-stack)
2. [Class diagrams](#2-class-diagrams) — 2a contracts, 2b model internals, 2c Harmony chat template
3. [Loading sequence](#3-loading-sequence)
4. [Forward pass](#4-forward-pass) — 4.0 global, 4a `ingest`, 4b `decoderBlock`, 4c output tail
5. [State and checkpoint layout](#5-state-and-checkpoint-layout)
6. [Harmony reply language](#6-harmony-reply-language)

| Class | Role |
|---|---|
| `GptOss` | The model: `LanguageModel` impl with nested records `Configuration`, `AttentionWeights`, `MoeFfnWeights`, `LayerWeights`, `Weights` and the class `State`; the whole forward pass, GGUF loading, expert top-k routing |
| `GptOssProvider` | `ModelProvider` SPI for arch `gpt-oss`: `loadLanguage` builds the `LoadedModel` (stop tokens, seed, chat template, sampling defaults) |
| `GptOssChatTemplate` | `ChatTemplate`: framing is delegated to the GGUF template (whole render); declares the Harmony reply language, parser, constrained reply and forced calls |
| `GptOssCheckpointCodec` | `CheckpointCodec<State>`: serialises F16 K/V rows of every layer, ring-aware for sliding-window layers |

---

## 1. Position in the jinfer stack

```mermaid
flowchart TB
    Callers["Callers: cli / server / langchain4j / spring-ai"]
    subgraph chat["jinfer-chat"]
        Models["Models (ServiceLoader)<br/>loadLanguage"]
        MP["ModelProvider"]
        LM2["LoadedModel + SamplingDefaults"]
        CT["ChatTemplate / ReplyLanguage"]
    end
    subgraph core["jinfer-core"]
        LM["LanguageModel"]
        CS["ContextState"]
        CC["CheckpointCodec"]
        B["Batch (Tokens | Sequences | Embeddings)"]
    end
    subgraph go["jinfer-gptoss"]
        P["GptOssProvider"]
        G["GptOss"]
        T["GptOssChatTemplate"]
        C["GptOssCheckpointCodec"]
    end
    subgraph kern["jinfer-kernels"]
        K["MatMul · Norms · RoPE · FlashAttention<br/>Moe · Activations · Convert · Ops · KvTransfer · ModelLoader"]
    end
    Tok["toknroll Tokenizer"]

    Callers --> Models --> P
    P -.->|implements| MP
    P -->|produces| LM2
    G -.->|implements| LM
    T -.->|implements| CT
    C -.->|extends| CC
    G --> K
    C --> K
    G --> Tok
    LM2 --> G
    LM2 --> T
```

---

## 2. Class diagrams

### 2a. Contracts and relations

```mermaid
classDiagram
    direction LR
    class ModelProvider { <<interface, external>> }
    class LanguageModel~C,W,S~ { <<interface, external>> }
    class ContextState { <<abstract, external>> }
    class CheckpointCodec~S~ { <<abstract, external>> }
    class ChatTemplate { <<interface, external>> }
    class ContextConfiguration { <<interface, external>> }
    class GptOssProvider {
        <<final>>
        architectures() Set gpt-oss
        loadLanguage(channel, gguf, path, arena, companions, tokenizer) LoadedModel
    }
    class GptOss { <<final>> }
    class GptOssChatTemplate { <<package-private>> }
    class GptOssCheckpointCodec { <<package-private>> }
    class State { <<GptOss.State>> }
    class Configuration { <<GptOss.Configuration record>> }

    ModelProvider <|.. GptOssProvider
    LanguageModel <|.. GptOss
    ChatTemplate <|.. GptOssChatTemplate
    ContextState <|-- State
    CheckpointCodec <|-- GptOssCheckpointCodec
    ContextConfiguration <|.. Configuration
    GptOssProvider ..> GptOss : loadModel
    GptOssProvider ..> GptOssChatTemplate : new(tokenizer)
    GptOss ..> GptOssCheckpointCodec : checkpointCodec()
    GptOssCheckpointCodec --> Configuration
    GptOssCheckpointCodec ..> State : reads/writes KV
    GptOss *-- State : newState
```

### 2b. Model internals — `GptOss` and its nested types

```mermaid
classDiagram
    direction TB
    class GptOss {
        <<final>>
        -Configuration configuration
        -Tokenizer tokenizer
        -Weights weights
        +loadModel(path, arena)$
        +loadModel(channel, gguf, arena, tokenizer)$
        +ingest(State, Batch)
        +logits(State, output) MemoryView
        -forward(State, Batch)
        -forward(State, ids, tokenOffset, startPos, seqLen)
        -embed()
        -layer() attention() moeFeedForward()
        -selectExperts()
        -writeKv() commitKv()
        -tailAt() projectLogits()
        -loadConfiguration()$ loadWeights()$
    }
    class Configuration {
        <<record>>
        embeddingLength numberOfLayers
        numberOfHeads numberOfKeyValueHeads headSize
        vocabularySize maxContextLength rmsNormEps
        ropeTheta ropeScalingFactor ropeOrigCtx
        slidingWindow swaMask[]
        expertCount expertUsedCount
        expertFeedForwardLength expertWeightsScale
        queryDim() kvDim() isSWA(l)
        kvCachePositions(l, cap) kvCacheIndex(l, pos)
        attentionStart(l, pos)
    }
    class Weights {
        <<record>>
        tokenEmbeddings
        LayerWeights[] layers
        outputNorm
        outputWeight (tied if absent)
        RoPE.Schedule rope (YaRN)
    }
    class LayerWeights {
        <<record>>
        attnNorm
        postAttnNorm
        AttentionWeights attention
        MoeFfnWeights moe
    }
    class AttentionWeights {
        <<record>>
        wq wk wv wo
        qBias kBias vBias oBias
        sinks
    }
    class MoeFfnWeights {
        <<record>>
        router routerBias
        gateExps[] gateBias
        upExps[] upBias
        downExps[] downBias
    }
    class State {
        <<static final>>
        residual normed branchOut
        attnOut query logits
        tail tailScratch
        ropeCos ropeSin
        keyCache[] valueCache[] F16
        batchK[] batchV[] F32
        decodeScratch
        moeRouter moeGather moeExpertOut moeOut
        hidden hidden2
        moeExpertCounts moeRowTopE moeRowTopP
        Moe.Routing moeRouting
        clearHistory() no-op
        advance(batch)
    }
    GptOss *-- Configuration
    GptOss *-- Weights
    Weights *-- LayerWeights
    LayerWeights *-- AttentionWeights
    LayerWeights *-- MoeFfnWeights
    GptOss ..> State : newState
```

### 2c. Harmony chat template

```mermaid
classDiagram
    direction LR
    class ChatTemplate { <<interface, external>> }
    class ReplyLanguage { <<external>> }
    class ReplyParser { <<external>> }
    class GptOssChatTemplate {
        <<package-private>>
        -Tokenizer tokenizer
        -Selection autoReply (memoized)
        encode() throws UnsupportedConversation
        thinkingPolicy() ALWAYS
        parser(tokenizer) ReplyParser
        constrainedReply(grammar) Optional
        constrainedReply(grammar, calls) Optional
        forcedCall(callableTools) Optional
        -harmonyLanguage(hole, allowCalls) Node
        -message(reopened, kind, channel, calls, hole) Node
        ~harmonyCalls(payload)$ List~ToolCall~
    }
    ChatTemplate <|.. GptOssChatTemplate
    GptOssChatTemplate ..> ReplyLanguage : builds Node trees
    GptOssChatTemplate ..> ReplyParser : autoReply.walk()
```

---

## 3. Loading sequence

```mermaid
sequenceDiagram
    participant M as Models (jinfer-chat)
    participant P as GptOssProvider
    participant G as GptOss
    participant L as ModelLoader
    participant TL as GGUFTokenizerLoader
    participant T as GptOssChatTemplate

    M->>P: loadLanguage(channel, gguf, path, arena, companions, tokenizer)
    P->>G: loadModel(channel, gguf, arena, tokenizer)
    opt tokenizer is null
        G->>TL: createBuilderWithBuiltins().build().fromGGUF(gguf)
        TL-->>G: Tokenizer
    end
    G->>G: loadConfiguration: gpt-oss.* keys, swaMask[l] = (l even), vocab from tokenizer
    G->>L: loadTensors(channel, gguf, arena)
    L-->>G: Map name to MemoryView
    G->>G: loadWeights: blk.N.* tensors, experts via sliceLeadingAxis
    G->>G: output.weight else token_embd.weight (tied)
    G->>G: RoPE.yarn(headSize, theta, factor, origCtx, 32, 1, 1, 1)
    G-->>P: GptOss
    P->>T: new GptOssChatTemplate(tokenizer)
    P-->>M: LoadedModel(model, tokenizer, tokenizer.chat_template, stops, modelSeed, Optional template, SamplingDefaults 1.0 / 1.0)
```

Stop tokens: `SpecialTokens.stops(tok, -1, "<|return|>", "<|call|>", "<|endoftext|>")`. Defaults for configuration keys:
`rms_epsilon` 1e-5, `rope.freq_base` 150000, `rope.scaling.factor` 1, `key_length` = dim / heads, `expert_*` 0.

---

## 4. Forward pass

`ingest(state, batch)` runs `state.exclusively(forward)`. Only `Batch.Input.Tokens` is accepted. All layers except the
last run the full block; the **last layer** only commits its K/V during `ingest`, and its attention + MoE are computed
lazily for a single row inside `logits(...)` (section 4c).

### 4.0 Global architecture

```mermaid
flowchart TB
    Tok["token ids (rows)"] --> Emb["Token embedding gather<br/>into F32 residual stream (rows x dim)"]
    Emb --> Stack
    subgraph Stack["Layers 0 .. N-2 (full block)"]
        direction TB
        subgraph Block["decoderBlock: pre-norm residual x2"]
            direction LR
            A["attnNorm then GQA attention<br/>biases + NeoX RoPE + sinks<br/>even layers sliding window, odd full"] --> Add1(("+")) --> F["postAttnNorm then top-k MoE<br/>clamped SwiGLU experts with biases"] --> Add2(("+"))
        end
    end
    Stack --> Last["Layer N-1 at ingest: attnNorm, K/V projection, RoPE, commit only"]
    Last --> Adv["state.advance(batch)"]
    Adv -.->|later| Tail["logits(output): tailAt(row) = last layer attention + MoE for one row"]
    Tail --> FN["outputNorm RMSNorm"]
    FN --> Head["outputWeight GEMV then logits (vocab)"]
    Stack -.-> St[("State: F16 KV per layer<br/>SWA layers hold a ring of slidingWindow rows")]
```

### 4a. `ingest` — validation, input kinds, rope/embedding, layer loop, advance

```mermaid
flowchart TB
    In["ingest(state, Batch)"] --> Ex["state.exclusively"]
    Ex --> V{"count in 1..batchCapacity<br/>position + count at most contextCapacity"}
    V -->|no| Err["IllegalArgumentException"]
    V -->|yes| Sw{"batch.input()"}
    Sw -->|Tokens| R["every id in 0..vocab-1<br/>else IllegalArgumentException"]
    Sw -->|Sequences| S["UnsupportedOperationException<br/>generative model, no packing"]
    Sw -->|Embeddings| X["UnsupportedOperationException<br/>text-only"]
    R --> Rope["RoPE.fill(ropeCos, ropeSin, startPos, seqLen, headSize/2, YaRN schedule)"]
    Rope --> Emb["Convert.gatherToF32(tokenEmbeddings, ids) into residual"]
    Emb --> Loop["for l in 0 .. N-2"]
    Loop --> Lay["layer(l): attention then moeFeedForward on residual"]
    Lay --> Tr{"Trace.ENABLED"}
    Tr -->|yes| Sum["Trace.sum l_out-l"]
    Tr -->|no| Next
    Sum --> Next{"more layers"}
    Next -->|yes| Loop
    Next -->|no| WK["writeKv(N-1): attnNorm, wk wv, biases, RoPE on K, commitKv"]
    WK --> Adv["state.advance(batch):<br/>advanceContext(count, outputs)"]
    Adv --> RF["Reference.reachabilityFence(this)"]
```

### 4b. `decoderBlock` — one layer (attention then MoE)

```mermaid
flowchart TB
    R0["residual (seqLen x dim)"] --> N1["Norms.rmsnormRows with attnNorm"]
    N1 --> Q["gemm wq, + qBias"]
    N1 --> K["gemm wk into batchK, + kBias"]
    N1 --> V["gemm wv into batchV, + vBias"]
    Q --> RQ["RoPE.applyNeox per row and query head"]
    K --> RK["RoPE.applyNeox per row and KV head"]
    RQ --> Mode{"seqLen"}
    RK --> Mode
    Mode -->|"more than 1"| PF["FlashAttention.slidingWindowPrefill<br/>cache + batchK/V, scale 1/sqrt(headSize), sinks<br/>swa: window W, span W-1; full: 0, 0"]
    Mode -->|"1"| FD["FlashAttention.flashDecode<br/>start = attentionStart(l, pos), sinks, decodeScratch"]
    V --> PF
    V --> FD
    PF --> O["gemm wo, + oBias"]
    FD --> O
    O --> A1["Ops.addRows into residual"]
    A1 --> KV["commitKv: Convert.f32ToF16 of batchK/V rows<br/>to cache slot kvCacheIndex(l, pos)"]
    A1 --> N2["Norms.rmsnormRows with postAttnNorm"]
    N2 --> Rt["gemm router, + routerBias"]
    Rt --> Sel["Moe.selectTopK then softmax over the k raw logits<br/>times expertWeightsScale"]
    Sel --> Disp["Moe.dispatch: gather rows per expert"]
    Disp --> E1["per expert e: gemm gateExps[e], gemm upExps[e]<br/>+ gateBias/upBias slice e"]
    E1 --> Act["Activations.clampedSwigluMultiply per row"]
    Act --> E2["gemm downExps[e], + downBias slice e"]
    E2 --> Comb["weighted scatter into moeOut"]
    Comb --> A2["Ops.addRows moeOut into residual"]
    A2 --> Out["residual to next layer"]
```

Layers differ only in the attention span: `swaMask[l] = (l % 2 == 0)`. Sliding layers keep a ring of
`min(contextCapacity, slidingWindow)` rows (`slidingWindow` must be a power of two, slot = `pos & (W-1)`), attend to
`[max(0, pos-W+1), pos]`; full layers keep `contextCapacity` rows at absolute positions. The MoE has no shared expert
and no dense layers.

### 4c. Output tail — `logits(state, output)`

```mermaid
flowchart TB
    L["logits(state, output)"] --> C{"output in 0..outputCount-1"}
    C -->|no| E["IllegalArgumentException"]
    C -->|yes| Row["row = lastBatchSize - outputCount + output"]
    Row --> T1["tailAt(row): rmsnorm of residual row with last attnNorm"]
    T1 --> T2["gemm wq + qBias, RoPE.fill at absolute position, applyNeox per head"]
    T2 --> T3["flashDecode against the already committed last-layer cache<br/>(no new K/V), sinks, window if SWA"]
    T3 --> T4["gemm wo + oBias, add into tail = residual row + attention"]
    T4 --> T5["moeFeedForward(l=N-1) on tail, 1 row"]
    T5 --> T6["Norms.rmsnorm tail with outputNorm into normed"]
    T6 --> T7["MatMul.gemv(outputWeight, normed) into logits (1 x vocab)"]
```

---

## 5. State and checkpoint layout

```mermaid
flowchart LR
    subgraph St["State (per context)"]
        direction TB
        subgraph SW["even layers: sliding window"]
            KS["keyCache F16: min(cap, W) x kvDim<br/>slot = pos AND W-1"]
            VS["valueCache F16: same shape"]
        end
        subgraph FL["odd layers: full attention"]
            KF["keyCache F16: cap x kvDim<br/>row = absolute pos"]
            VF["valueCache F16: same shape"]
        end
        Scr["F32 scratch: residual normed branchOut attnOut query<br/>batchK/V per layer, rope cos/sin, moe buffers<br/>logits 1 x vocab, tail, tailScratch"]
    end
    Cod["GptOssCheckpointCodec<br/>bytesPerPosition = 2 x 2 bytes x layers x kvDim"]
    Blob[("checkpoint blob: per layer K then V,<br/>layers in order")]
    SW -->|"KvTransfer.ringSpan"| Cod
    FL -->|"KvTransfer.transfer"| Cod
    Cod <--> Blob
    Hist["clearHistory(): no-op, no recurrent residue"]
    St --- Hist
```

`transfer(state, from, to, blob, capture)` loops layers `0..N-1`; per layer it moves the K span then the V span.
`sizeOf(positions) = positions * 2 * sizeof(short) * numberOfLayers * kvDim` (uniform; SWA layers are accounted at the
same per-position size). Sliding-window rows are stored through their ring slots, so the live window rebuilds from
restored rows alone and every block is a resume point; spans longer than W alias ring slots (see `KvTransfer.ringSpan`).

---

## 6. Harmony reply language

`encode()` throws `UnsupportedConversation`: prompt framing is the GGUF template's whole render. The template only
declares the *reply* grammar; `thinkingPolicy()` is `ALWAYS`.

```mermaid
flowchart TB
    Entry["harmonyLanguage(contentHole, allowCalls)"] --> Shapes["for reopened in false, true"]
    Shapes --> An["analysis: Kind.THINK, free body"]
    Shapes --> Co["commentary: Kind.CONTENT, free body"]
    Shapes --> Fi["final: Kind.CONTENT, contentHole (only if allowCalls)"]
    Shapes --> Ca["call: Kind.CALL, header commentary to=NAME, harmonyCalls parser (only if allowCalls)"]
    An --> Msg["message = optional start+assistant, channel marker, name,<br/>optional constrain json, message marker, body hole"]
    Co --> Msg
    Fi --> Msg
    Ca --> Msg
    Msg --> Stream["stream = repeat(message, optional separator end | call | return)"]
    Stream --> Calls{"allowCalls"}
    Calls -->|yes| Done["stream"]
    Calls -->|no| Req["stream then ONE required final message, optional separator"]
    subgraph Entries["Public entry points"]
        P1["parser(tokenizer): memoized free language, walk()"]
        P2["constrainedReply(grammar[, calls]): final channel takes the GBNF hole"]
        P3["forcedCall(tools): per-tool call region with schema-bound GBNF"]
    end
    Entries --> Entry
    Hc["harmonyCalls(payload): to=functions.NAME then first JSON object<br/>no name or no object means no call, reply continues"]
    Ca --> Hc
    P3 --> Hc
```
