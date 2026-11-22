# jinfer-laguna — class architecture

Package `com.qxotic.jinfer.models.laguna` (4 classes, ~1130 lines of main code). It is the `jinfer-<model>` provider for
GGUF architecture `laguna` (Poolside Laguna XS 2.1): a **text-only generative decoder**, hybrid **dense / MoE**
(leading dense FFN layers, then sigmoid-routed experts + one shared expert), with **alternating sliding-window and
full attention** layers, QK-norm, NeoX RoPE (YaRN on full layers, plain on sliding layers) and a **per-head softplus
attention gate**. There is no vision, audio, speculative or recurrent-state subsystem in this module.

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends;
classes marked `<<external>>` live outside this package.

1. [Position in the stack](#1-position-in-the-jinfer-stack)
2. [Class diagrams](#2-class-diagrams) — 2a contracts, 2b model internals, 2c chat template
3. [Loading sequence](#3-loading-sequence)
4. [Forward pass](#4-forward-pass) — 4.0 global, 4a `ingest`, 4b `decoderBlock`
5. [MoE FFN and logits head](#5-moe-ffn-and-logits-head)
6. [State and checkpoint layout](#6-state-and-checkpoint-layout)
7. [Chat template encoding](#7-chat-template-encoding)

| Class | Role |
|---|---|
| `Laguna` | The model: `LanguageModel` impl with nested `Configuration`, `Weights`, `LayerWeights`, `DenseWeights`, `MoeWeights`, `State`; GGUF loading, validation, the whole forward pass and the logits head |
| `LagunaProvider` | `ModelProvider` SPI for architecture `laguna`: `loadLanguage` builds the model, stop tokens, sampling defaults and the chat template |
| `LagunaChatTemplate` | `ChatTemplate`: token-exact encoder of the `<system>/<user>/<assistant>` XML format, think/tool-call reply parser, constrained and forced-call replies |
| `LagunaCheckpointCodec` | `CheckpointCodec<State>`: serialises full-attention KV rows and sliding-window KV rings as FP16 blobs |

---

## 1. Position in the jinfer stack

```mermaid
flowchart TB
    Callers["Callers: cli / server / langchain4j / spring-ai"]
    subgraph chat["jinfer-chat"]
        Models["Models (ServiceLoader)"]
        MP["ModelProvider"]
        LM_["LoadedModel"]
        CT["ChatTemplate / ReplyLanguage / ToolCallSyntax"]
    end
    subgraph core["jinfer-core"]
        LM["LanguageModel"]
        CS["ContextState"]
        CC["CheckpointCodec"]
        B["Batch (Tokens only here)"]
    end
    subgraph lag["jinfer-laguna"]
        P["LagunaProvider"]
        L["Laguna"]
        T["LagunaChatTemplate"]
        C["LagunaCheckpointCodec"]
    end
    subgraph kern["jinfer-kernels"]
        K["MatMul · Norms · RoPE · FlashAttention<br/>Activations · Convert · Ops · Moe · KvTransfer · ModelLoader"]
    end
    Tok["toknroll Tokenizer (GGUFTokenizerLoader)"]

    Callers --> Models --> P
    P -.->|implements| MP
    P -->|produces| LM_
    L -.->|implements| LM
    T -.->|implements| CT
    C -.->|extends| CC
    L --> K
    C --> K
    L --> Tok
    LM_ --> L
    LM_ --> T
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
    class ContextConfiguration { <<interface, external>> }
    class CheckpointCodec~S~ { <<abstract, external>> }
    class LoadedModel { <<record, external>> }
    class LagunaProvider {
        <<final>>
        architectures() Set
        loadLanguage(channel, gguf, path, arena, companions, tokenizer) LoadedModel
    }
    class Laguna {
        <<final>>
        loadModel(path, arena)$
        loadModel(channel, gguf, arena, tokenizer)$
        newState(ctx, batch)
        newState(ctx, batch, arena)
        ingest(State, Batch)
        logits(State, output) MemoryView
        checkpointCodec() Optional
    }
    class LagunaChatTemplate {
        <<package-private final>>
        DEFAULT_SYSTEM
        promptStart() IntSequence
        encode(conversation, batchCapacity, sink) ReplyState
        parser(tokenizer) ReplyParser
        constrainedReply(grammar) Optional
        forcedCall(tools) Optional
        defaultMaxReasoningTokens() -1
    }
    class LagunaCheckpointCodec {
        <<package-private final>>
        -bytesPerPosition
        sizeOf(positions) long
        transfer(state, from, to, blob, capture)
    }
    class State { <<Laguna.State>> }
    class Configuration { <<Laguna.Configuration record>> }

    ModelProvider <|.. LagunaProvider
    LanguageModel <|.. Laguna
    ChatTemplate <|.. LagunaChatTemplate
    ContextState <|-- State
    ContextConfiguration <|.. Configuration
    CheckpointCodec <|-- LagunaCheckpointCodec
    LagunaProvider ..> Laguna : loadModel
    LagunaProvider ..> LagunaChatTemplate : new(tokenizer, bos)
    LagunaProvider ..> LoadedModel : returns
    Laguna ..> LagunaCheckpointCodec : checkpointCodec()
    LagunaCheckpointCodec ..> State : reads/writes
    LagunaCheckpointCodec --> Configuration : config
    Laguna *-- State : newState
    Laguna *-- Configuration
```

### 2b. Model internals — `Laguna` and its nested types

```mermaid
classDiagram
    direction TB
    class Laguna {
        <<final>>
        -Configuration configuration
        -Tokenizer tokenizer
        -Weights weights
        +ingest(State, Batch)
        +logits(State, output) MemoryView
        -forward(State, Batch)
        -forward(State, tokens, startPos, rows)
        -decoderBlock(State, layer, startPos, rows)
        -attention(State, layer, startPos, rows)
        -commitKv(State, layer, startPos, rows)
        -denseFfn(State, DenseWeights, rows)
        -moe(State, MoeWeights, rows)
        -projectLogits(State, output)
        loadConfiguration(gguf, vocab, arch)$
        loadWeights(tensors, c)$
        yarnKernelAttentionFactor()$
    }
    class Configuration {
        <<record>>
        embeddingLength, numberOfLayers
        headCount[] per layer, keyValueHeadCount
        headSize, vocabularySize, maxContextLength, rmsNormEps
        feedForwardLength, denseLeadingLayers
        expertCount, expertUsedCount
        expertFeedForwardLength, sharedFeedForwardLength
        expertWeightsScale
        slidingWindow, isSwa[]
        ropeDimensionCount, ropeDimensionCountSwa
        ropeTheta, ropeThetaSwa
        ropeScalingFactor, ropeOriginalContext
        ropeBetaFast, ropeBetaSlow, ropeAttentionFactor
        queryDim(layer) maxQueryDim() maxHeadCount()
        kvDim() hasSwa()
        kvCachePositions(layer, cap)
        kvCacheIndex(layer, pos)
    }
    class Weights {
        <<record>>
        tokenEmbedding
        outputNorm
        outputWeight (tied to embedding if absent)
        LayerWeights[] layers
        RoPE.Schedule ropeFull (yarn)
        RoPE.Schedule ropeSwa (plain)
    }
    class LayerWeights {
        <<record>>
        attnNorm
        query key value output
        queryNorm keyNorm
        attentionGate (heads x dim)
        ffnNorm
        DenseWeights dense (or null)
        MoeWeights moe (or null)
    }
    class DenseWeights {
        <<record>>
        gate up down
    }
    class MoeWeights {
        <<record>>
        router
        selectionBias F32
        expertGate[] expertUp[] expertDown[]
        sharedGate sharedUp sharedDown
    }
    class State {
        <<static final, extends ContextState>>
        residual normed branch logits
        query attentionGate attentionOut
        ropeCosFull ropeSinFull ropeCosSwa ropeSinSwa
        denseHidden denseHidden2
        moeRouter moeSelection moeGather moeDown
        moeHidden moeHidden2
        sharedHidden sharedHidden2 sharedOut
        batchK batchV
        keyCache[] valueCache[] F16
        moeExpertCounts moeRowTopE moeRowTopP
        moeRouting
        clearHistory() no-op
        advance(batch)
    }
    Laguna *-- Configuration
    Laguna *-- Weights
    Weights *-- LayerWeights
    LayerWeights o-- DenseWeights : layer lt denseLeadingLayers
    LayerWeights o-- MoeWeights : layer ge denseLeadingLayers
    Laguna ..> State : newState
```

### 2c. Chat template

```mermaid
classDiagram
    direction LR
    class ChatTemplate { <<interface, external>> }
    class ReplyLanguage_Spans { <<external>> }
    class PromptWriter { <<external>> }
    class ToolCallSyntax { <<external>> }
    class SpecialTokens { <<external>> }
    class LagunaChatTemplate {
        <<package-private final>>
        -promptStart (BOS)
        -thinkOpen thinkClose ids
        -spans
        encode()
        writeHeader() writeAssistant() writeCall()
        toolEnvelope() requireSupported()
    }
    ChatTemplate <|.. LagunaChatTemplate
    LagunaChatTemplate --> ReplyLanguage_Spans : think, tool_call, TURN_END marker
    LagunaChatTemplate ..> PromptWriter : chunks prompt to batchCapacity
    LagunaChatTemplate ..> ToolCallSyntax : parseTaggedXml, jinjaJson
    LagunaChatTemplate ..> SpecialTokens : require think tokens, encode
```

---

## 3. Loading sequence

```mermaid
sequenceDiagram
    participant M as Models (jinfer-chat)
    participant P as LagunaProvider
    participant L as Laguna
    participant G as GGUFTokenizerLoader
    participant ML as ModelLoader
    participant T as LagunaChatTemplate

    M->>P: loadLanguage(channel, gguf, path, arena, companions, tokenizer)
    P->>L: loadModel(channel, gguf, arena, tokenizer)
    L->>L: general.architecture must equal laguna
    opt tokenizer is null
        L->>G: build from GGUF metadata
        G-->>L: Tokenizer
    end
    L->>L: loadConfiguration (per-layer heads, isSwa pattern, rope, MoE keys)
    L->>L: validate dims, rope, FFN/MoE, then require sigmoid gating, normalised weights, 1 shared expert, rope scaling none or yarn
    L->>ML: loadTensors(channel, gguf, arena)
    ML-->>L: Map name to MemoryView
    L->>L: loadWeights (dense for leading layers, MoE after, per-head gate shape check, YaRN full + plain SWA schedules)
    L-->>P: Laguna
    P->>P: bos from tokenizer.ggml.bos_token_id, eos and eot optional
    P->>P: stops = SpecialTokens.stops(eos, assistant-close) plus eot
    P->>T: new LagunaChatTemplate(tokenizer, bos) (requires think and /think tokens)
    P-->>M: LoadedModel(model, tokenizer, chat_template string, stops, modelSeed, template, SamplingDefaults)
```

Sampling defaults come from `general.sampling.temp/top_p/top_k/min_p` (fallbacks 1, 1, 20, 0).
`isSwa[layer] = swaPeriod == 0 || layer % swaPeriod != 0` (period default 4, so layers 0, 4, 8, ... are full attention) and only when `sliding_window > 0`; the window must be a power of two.

---

## 4. Forward pass

`ingest(state, batch)` runs `state.exclusively(forward)`. Only `Batch.Input.Tokens` is accepted: `Sequences` throws "does not support packed sequences" and `Embeddings` throws "text-only". There is no recurrent state: carried state is the per-layer F16 KV cache.

### 4.0 Global architecture

```mermaid
flowchart TB
    Tok["token ids [rows]"] --> Emb["Token embedding<br/>gatherToF32 → F32 residual [rows × dim]"]
    Rope["RoPE tables filled once per batch<br/>full: YaRN, swa: plain"] -.-> Stack
    subgraph Stack["N decoder blocks (layer kinds differ)"]
        direction TB
        subgraph Block["decoderBlock (pre-norm residual ×2)"]
            direction LR
            A["RMSNorm → gated GQA attention<br/>full-causal or sliding-window (per layer)"] --> Add1(("+"))
            Add1 --> F["RMSNorm → FFN"]
            F --> Add2(("+"))
        end
        FK["FFN kind: layer &lt; denseLeadingLayers → SwiGLU dense<br/>else sigmoid top-k MoE + shared expert"]
    end
    Emb --> Stack
    Stack --> FN["Final RMSNorm (outputNorm) on selected row"]
    FN --> H["gemv outputWeight → logits [vocab]"]
    Stack -.-> St[("State: F16 KV per layer<br/>full: ctx rows, swa: ring of window rows")]
```

### 4a. `ingest` — batch validation, input kinds, rope/embedding, layer loop

```mermaid
flowchart TB
    In["ingest(state, Batch)"] --> Ex["state.exclusively"]
    Ex --> V{"0 &lt; count &lt;= batchCapacity<br/>position + count &lt;= contextCapacity"}
    V -->|no| Err["IllegalArgumentException"]
    V -->|yes| Sw{"batch.input()"}
    Sw -->|Tokens| T["ids[]"]
    Sw -->|Sequences| S["UnsupportedOperationException<br/>no packed sequences"]
    Sw -->|Embeddings| X["UnsupportedOperationException<br/>text-only"]
    T --> RT{"every id in [0, vocabularySize)"}
    RT -->|no| Err
    RT -->|yes| RF["RoPE.fill ropeCosFull/SinFull<br/>startPos, rows, ropeDimensionCount/2, ropeFull"]
    RF --> RS{"hasSwa()"}
    RS -->|yes| RS2["RoPE.fill ropeCosSwa/SinSwa<br/>ropeDimensionCountSwa/2, ropeSwa"]
    RS -->|no| Emb
    RS2 --> Emb["Views.checkAlive(tokenEmbedding)<br/>Convert.gatherToF32 → residual"]
    Emb --> Loop{"for l in 0..numberOfLayers"}
    Loop --> DB["decoderBlock(l, startPos, rows)  → see 4b"]
    DB --> Loop
    Loop -->|done| Adv["state.advance(batch)<br/>advanceContext(count, outputs)"]
    Adv -.->|later| Head["logits(state, output)  → see 5"]
```

### 4b. `decoderBlock` — one layer

```mermaid
flowchart TB
    R0["residual [rows × dim]"] --> N1["Norms.rmsnormRowsGgml → normed (attnNorm)"]
    subgraph Attn["attention(l)"]
        direction TB
        N1 --> Q["gemm query → query"]
        N1 --> K["gemm key → batchK"]
        N1 --> V["gemm value → batchV"]
        N1 --> G["gemm attentionGate → attentionGate [rows × heads]"]
        Q --> QN["Parallel per row, per head:<br/>RMSNorm (queryNorm) then RoPE.applyNeox"]
        K --> KN["Parallel per row, per kv head:<br/>RMSNorm (keyNorm) then RoPE.applyNeox"]
        QN --> FA["FlashAttention.slidingWindowPrefill<br/>cache + batchK/V, scale 1/sqrt(headSize), kvMul = heads/kvHeads<br/>swa: window, window-1; full: 0, 0"]
        KN --> FA
        V --> FA
        FA --> GT["per row, per head: attentionOut *= softplus(gate)"]
        G --> GT
        GT --> WO["gemm output → branch"]
        WO --> CK["commitKv: f32ToF16 batchK/V → keyCache/valueCache[l]<br/>row index = kvCacheIndex (swa: pos AND window-1)"]
    end
    CK --> A1["Ops.addRows: residual += branch"]
    A1 --> N2["Norms.rmsnormRowsGgml → normed (ffnNorm)"]
    N2 --> Sel{"layer &lt; denseLeadingLayers"}
    Sel -->|yes| D["denseFfn: gate, up → siluMultiplyRows → down → branch"]
    Sel -->|no| MO["moe(): routed experts + shared expert → branch  (see 5)"]
    D --> A2["Ops.addRows: residual += branch"]
    MO --> A2
    A2 --> Tr["Trace.sum l_out-layer (if Trace.ENABLED)"]
```

Details worth knowing:

- `commitKv` runs after the layer's attention, because `batchK/batchV` are single buffers shared by all layers; the attention kernel reads the new rows from `batchK/V` and older rows from the cache.
- Heads per layer come from `headCount[layer]` (`attention.head_count` may be a scalar or an array); `keyValueHeadCount` and `headSize` are global; `query` and `attentionOut` are sized for the largest layer.
- The sliding window passed to the kernel is `slidingWindow` (and `slidingWindow - 1` as the second window argument); full layers pass `0, 0`.
- The attention gate is per-head (loader requires the gate tensor shape `headCount[layer] × embeddingLength`) and applied as `softplus` to the attention output before the output projection.

---

## 5. MoE FFN and logits head

### 5a. `moe(state, w, rows)`

```mermaid
flowchart TB
    NI["normed [rows × dim]"] --> RT["gemm router → moeRouter [rows × expertCount]"]
    RT --> SG["mapInPlace sigmoid"]
    SG --> CP["copyF32 → moeSelection"]
    CP --> BI["addRowBiasInPlace selectionBias<br/>(selection only, not weights)"]
    BI --> TK["Moe.selectTopK(selection, router, expertUsedCount)<br/>→ moeRowTopE, moeRowTopP, moeExpertCounts"]
    TK --> NP["Moe.normalizeTopP; each weight *= expertWeightsScale"]
    NP --> DP["Moe.dispatch: gather rows per expert (moeGather)"]
    subgraph Ex["per active expert (count rows)"]
        direction TB
        EG["gemm expertGate[e] → moeHidden"]
        EU["gemm expertUp[e] → moeHidden2"]
        EG --> SI["siluMultiplyRows"]
        EU --> SI
        SI --> ED["gemm expertDown[e] → output (moeDown)"]
    end
    DP --> Ex
    Ex --> SC["weighted scatter-add → branch"]
    NI --> SH["shared expert: sharedGate, sharedUp → siluMultiplyRows → sharedDown → sharedOut"]
    SC --> ADD["Ops.addRows: branch += sharedOut"]
    SH --> ADD
```

### 5b. Logits head

```mermaid
flowchart LR
    Chk{"0 &lt;= output &lt; outputCount"} -->|no| Err["IllegalArgumentException"]
    Chk -->|yes| Row["row = lastBatchSize - outputCount + output"]
    Row --> FN["Norms.rmsnormGgml(outputNorm) residual row → normed"]
    FN --> GV["MatMul.gemv(outputWeight, normed) → logits [1 × vocab]"]
```

`logits` runs under `state.exclusively`. `outputWeight` falls back to `tokenEmbedding` when `output.weight` is absent (tied head).

---

## 6. State and checkpoint layout

```mermaid
flowchart TB
    subgraph State["Laguna.State (per context)"]
        direction TB
        subgraph scratch["Scratch sized by batchCapacity"]
            A["residual · normed · branch · query · attentionGate · attentionOut"]
            B["ropeCos/Sin Full and Swa · batchK · batchV"]
            C["denseHidden(2) · moeRouter · moeSelection · moeGather · moeDown<br/>moeHidden(2) · sharedHidden(2) · sharedOut"]
            D["moeExpertCounts · moeRowTopE · moeRowTopP · moeRouting"]
            E["logits [1 × vocab]"]
        end
        subgraph persist["Persistent per layer (F16)"]
            KF["full layer: keyCache/valueCache [contextCapacity × kvDim]"]
            KS["swa layer: ring [min(ctx, window) × kvDim]<br/>row = position AND window-1"]
        end
    end
    subgraph Blob["LagunaCheckpointCodec blob"]
        B1["bytesPerPosition = 2 bytes × layers × 2 × kvDim<br/>per layer in order: K then V"]
        B2["full: rows [from, to) via KvTransfer.transfer<br/>swa: KvTransfer.ringSpan(window)"]
    end
    KF -->|"transfer(capture)"| B2
    KS -->|"ringSpan(capture)"| B2
    B2 -->|restore| KF
    B2 -->|restore| KS
    B1 --- B2
```

- `clearHistory()` is a no-op: stale rows past the cursor are masked by attention and overwritten on ingest.
- `newState(ctx, batch)` allocates a cross-thread arena owned by the state (closed if construction fails); `newState(ctx, batch, arena)` borrows the caller's arena.
- `State` rejects `contextCapacity <= 0` or `> maxContextLength`. `ropeCosSwa/SinSwa` are null when there is no sliding window.
- `checkpointCodec()` returns a fresh `LagunaCheckpointCodec` over the configuration, so block caching is supported.

---

## 7. Chat template encoding

```mermaid
flowchart TB
    E["encode(conversation, batchCapacity, sink)"] --> RS["requireSupported per message<br/>system/user: Text; assistant: Text, Reasoning, ToolCall; tool: ToolResult"]
    RS -->|unsupported| UC["UnsupportedConversation"]
    RS --> SY{"first message is system?"}
    SY -->|yes| S1["system = its text"]
    SY -->|no| S2["system = DEFAULT_SYSTEM (Poolside assistant)"]
    S1 --> H["writeHeader: system and ### Tools with available_tools JSON<br/>(skipped if blank system, no tools, no thinking)"]
    S2 --> H
    H --> LP{"for each remaining message"}
    LP -->|user| U["&lt;user&gt;…&lt;/user&gt;"]
    LP -->|system| Sy["&lt;system&gt;…&lt;/system&gt;"]
    LP -->|assistant| As["&lt;assistant&gt; think block (or bare &lt;/think&gt; when off) + text + tool_call blocks &lt;/assistant&gt;"]
    LP -->|tool| TR["&lt;tool_response&gt;…&lt;/tool_response&gt;"]
    U --> LP
    Sy --> LP
    As --> LP
    TR --> LP
    LP -->|done| GP["append &lt;assistant&gt; + &lt;think&gt; (thinking) or &lt;/think&gt; (off)"]
    GP --> PW["PromptWriter: ids(BOS) + SpecialTokens.encode(prompt) → batches → sink"]
    PW --> RP["ReplyState(replyPrefix, spans.parser())<br/>thinking: think-open; off: think-open + think-close"]
```

- Tool calls are written as `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`; non-string values use `ToolCallSyntax.jinjaJson`; replies are parsed by `ToolCallSyntax.parseTaggedXml`.
- The turn end marker is `</assistant>` (also a stop token in the provider); `defaultMaxReasoningTokens` returns `-1` (no default reasoning cap).
- `constrainedReply(grammar[, calls])` and `forcedCall(tools)` delegate to `ReplyLanguage.Spans`; `forcedCall` is empty when no tools are callable.
