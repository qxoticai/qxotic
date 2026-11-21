# jinfer-llama — class architecture

Package `com.qxotic.jinfer.models.llama` (12 classes, ~2600 lines). It is the `jinfer-<model>` provider for the GGUF
architectures `llama`, `minicpm`, `mistral3`, `smollm3` (all served by the single `Llama` class, differences driven by
GGUF metadata) and `granite` (the near-identical `Granite` class). All models are **text-only, dense, generative**:
GQA attention with interleaved RoPE, SwiGLU FFN, F16 KV cache. MoE, fused QKV, packed sequences and embedding input are
rejected. Each architecture family brings its own `ChatTemplate` (reply codec: call parsing, constrained and forced calls).

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends;
classes marked `<<external>>` live outside this package.

1. [Position in the stack](#1-position-in-the-jinfer-stack)
2. [Class diagrams](#2-class-diagrams) — 2a contracts, 2b `Llama` internals, 2c `Granite` internals, 2d chat templates
3. [Loading sequence](#3-loading-sequence)
4. [Forward pass](#4-forward-pass) — 4.0 global, 4a `ingest`, 4b `decoderBlock`, 4c lazy last-layer tail and logits
5. [Variant matrix and RoPE selection](#5-variant-matrix-and-rope-selection)
6. [Chat template selection and reply grammar](#6-chat-template-selection-and-reply-grammar)
7. [State and checkpoint layout](#7-state-and-checkpoint-layout)

| Class | Role |
|---|---|
| `Llama` | `LanguageModel` for llama / minicpm / mistral3 / smollm3; nested `Configuration`, `LayerWeights`, `Weights`, `State`; forward pass with a lazy last-layer tail |
| `Granite` | `LanguageModel` for `granite`; same contract, eager full-layer evaluation, optional NoPE switch, no attention temperature |
| `LlamaProvider` | `ModelProvider` SPI for `llama`, `minicpm`, `mistral3`, `smollm3`; picks the chat template and stop tokens |
| `GraniteProvider` | `ModelProvider` SPI for `granite`; always attaches `GraniteChatTemplate` |
| `LlamaCheckpointCodec` | `CheckpointCodec<Llama.State>`: serialises F16 K/V rows of every layer |
| `GraniteCheckpointCodec` | Same for `Granite.State` |
| `Llama32ChatTemplate` | Token-exact native framing for the Llama 3.x plain conversation; bare-JSON tool-call parser; forced-call grammar |
| `MistralChatTemplate` | Reply codec only (`[TOOL_CALLS] name [ARGS] json`, close-less); `<s>` prompt start |
| `SmolLm3ChatTemplate` | Reply codec: `<think>`, `<tool_call>` JSON, `<|im_end|>` |
| `MiniCpm5ChatTemplate` | Reply codec: `<think>`, `<function ...>` XML spans, `<|im_end|>` |
| `MiniCpmToolSyntax` | Parser of MiniCPM5 XML function payload (`name="..."`, `<param>`, CDATA) |
| `GraniteChatTemplate` | Reply codec: `<think>`, `<tool_call>`; JSON envelope (4.1) or function XML (4.2) |

---

## 1. Position in the jinfer stack

```mermaid
flowchart TB
    Callers["Callers: cli / server / langchain4j / spring-ai"]
    subgraph chat["jinfer-chat"]
        Models["Models (ServiceLoader)<br/>loadLanguage"]
        MP["ModelProvider"]
        LM_["LoadedModel"]
        CT["ChatTemplate / ReplyLanguage / SpecialTokens"]
    end
    subgraph core["jinfer-core"]
        LM["LanguageModel"]
        CS["ContextState"]
        CC["CheckpointCodec / KvTransfer"]
        B["Batch (Tokens only here)"]
    end
    subgraph ll["jinfer-llama"]
        LP["LlamaProvider"]
        GP["GraniteProvider"]
        L["Llama"]
        G["Granite"]
        TPL["Llama32 / Mistral / SmolLm3 / MiniCpm5 / Granite ChatTemplate"]
        COD["LlamaCheckpointCodec / GraniteCheckpointCodec"]
    end
    subgraph kern["jinfer-kernels"]
        K["MatMul · Norms · RoPE · FlashAttention<br/>Activations · Convert · Ops · ModelLoader"]
    end
    Tok["toknroll Tokenizer (GGUFTokenizerLoader)"]

    Callers --> Models --> MP
    LP -.->|implements| MP
    GP -.->|implements| MP
    LP -->|produces| LM_
    GP -->|produces| LM_
    L -.->|implements| LM
    G -.->|implements| LM
    TPL -.->|implements| CT
    COD -.->|extends| CC
    L --> K
    G --> K
    L --> Tok
    G --> Tok
    LM_ --> L
    LM_ --> G
    LM_ --> TPL
    L --> CS
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
    class ContextConfiguration { <<interface, external>> }
    class LlamaProvider {
        <<final>>
        architectures() llama minicpm mistral3 smollm3
        loadLanguage(...) LoadedModel
    }
    class GraniteProvider {
        <<final>>
        architectures() granite
        loadLanguage(...) LoadedModel
    }
    class Llama { <<final>> }
    class Granite { <<final>> }
    class LlamaState { <<Llama.State>> }
    class GraniteState { <<Granite.State>> }
    class LlamaCheckpointCodec { <<package-private>> }
    class GraniteCheckpointCodec { <<package-private>> }

    ModelProvider <|.. LlamaProvider
    ModelProvider <|.. GraniteProvider
    LanguageModel <|.. Llama
    LanguageModel <|.. Granite
    ContextState <|-- LlamaState
    ContextState <|-- GraniteState
    CheckpointCodec <|-- LlamaCheckpointCodec
    CheckpointCodec <|-- GraniteCheckpointCodec
    LlamaProvider ..> Llama : loadModel
    GraniteProvider ..> Granite : loadModel
    Llama ..> LlamaCheckpointCodec : checkpointCodec()
    Granite ..> GraniteCheckpointCodec : checkpointCodec()
    LlamaCheckpointCodec ..> LlamaState : reads/writes KV
    GraniteCheckpointCodec ..> GraniteState : reads/writes KV
    Llama *-- LlamaState : newState
    Granite *-- GraniteState : newState
    Llama ..> ContextConfiguration : Configuration implements
```

### 2b. Model internals — `Llama` and its nested types

```mermaid
classDiagram
    direction TB
    class Llama {
        <<final>>
        -Configuration configuration
        -Tokenizer tokenizer
        -Weights weights
        +loadModel(path/channel, gguf, arena, tokenizer)$
        +ingest(State, Batch)
        +logits(State, output) MemoryView
        -forward(State, Batch)
        -forward(State, ids, startPos, seqLen)
        -attention()
        -feedForward()
        -commitKv()
        -writeKv() last layer K/V only
        -tailAt() lazy last layer
        -feedForwardRow()
        -addBias()
        ~readConfiguration()$
        ~buildRope()$
        ~loadWeights()$
    }
    class Configuration {
        <<record>>
        embeddingLength numberOfLayers
        numberOfHeads numberOfKeyValueHeads
        headSize vocabularySize maxContextLength
        rmsNormEps ropeTheta ropeDimensionCount hiddenDim
        embeddingScale residualScale logitScale
        attnTempScale attnTempFloorScale
        attentionScaleValue noRopeLayerStep
        queryDim() kvDim() ropeHalf()
        attentionScale() attnTemp(pos) useRope(layer)
    }
    class Weights {
        <<record>>
        tokenEmbeddings
        LayerWeights[] layers
        finalNorm
        RoPE.Schedule rope
        wcls (tied to embeddings if no output.weight)
    }
    class LayerWeights {
        <<record>>
        attnNorm ffnNorm
        wq wk wv wo + optional bq bk bv bo
        w1 gate, w3 up, w2 down + optional b1 b3 b2
    }
    class State {
        <<static final>>
        residual normed query attnOut
        hidden hidden2 logits
        th tailScratch
        ropeCos ropeSin
        batchK batchV
        keyCache[] valueCache[] F16
        decodeScratch
        clearHistory() no-op
        -advance(Batch)
    }
    Llama *-- Configuration
    Llama *-- Weights
    Weights *-- LayerWeights
    Llama ..> State : newState
```

### 2c. `Granite` internals — deltas from `Llama`

```mermaid
classDiagram
    direction TB
    class Granite {
        <<final>>
        +loadModel(...)$
        +ingest(State, Batch)
        +logits(State, output)
        -forward() eager: all layers
        -attention()
        -feedForward()
        -commitKv()
        ~readConfiguration()$
        ~buildRope() plain/linear/llama3 only$
        ~loadWeights()$
        -effectiveScale(v) 0 becomes 1$
    }
    class GraniteConfiguration {
        <<record, Granite.Configuration>>
        embeddingScale residualScale
        logitScale (required key)
        attentionScaleValue
        useRope (rope.scaling.finetuned)
        no attnTemp, no noRopeLayerStep
    }
    class GraniteState {
        <<Granite.State>>
        same buffers as Llama.State
        NO th and NO tailScratch
        logits read from residual row directly
    }
    Granite *-- GraniteConfiguration
    Granite ..> GraniteState : newState
```

### 2d. Chat templates and reply codecs

```mermaid
classDiagram
    direction LR
    class ChatTemplate { <<interface, external>> }
    class ReplyLanguage { <<external>> }
    class ReplyParser { <<interface, external>> }
    class Llama32ChatTemplate {
        <<final>>
        DEFAULT_DATE
        promptStart() begin_of_text
        encode() native framing, no tools
        parser() BareToolCallParser
        forcedCall() name+parameters JSON
    }
    class BareToolCallParser {
        <<nested>>
        state UNDECIDED CALL CONTENT
        holds ambiguous prefix then decides
    }
    class MistralChatTemplate {
        <<final>>
        promptStart() s
        encode() throws Unsupported
        parser() memoized language
        constrainedReply() forcedCall()
        walkCalls(payload)$
    }
    class SmolLm3ChatTemplate { <<final>> }
    class MiniCpm5ChatTemplate { <<final>> }
    class GraniteChatTemplate {
        <<final>>
        functionXml flag
    }
    class MiniCpmToolSyntax {
        <<final utility>>
        parsePayload(block)$
    }
    class ToolCallSyntax { <<external>> }
    class JsonEnvelopeReplies { <<external>> }

    ChatTemplate <|.. Llama32ChatTemplate
    ChatTemplate <|.. MistralChatTemplate
    ChatTemplate <|.. SmolLm3ChatTemplate
    ChatTemplate <|.. MiniCpm5ChatTemplate
    ChatTemplate <|.. GraniteChatTemplate
    ReplyParser <|.. BareToolCallParser
    Llama32ChatTemplate *-- BareToolCallParser
    MiniCpm5ChatTemplate ..> MiniCpmToolSyntax : parsePayload
    MiniCpm5ChatTemplate ..> ReplyLanguage : Spans think/function
    SmolLm3ChatTemplate ..> ReplyLanguage : Spans think/tool_call
    SmolLm3ChatTemplate ..> JsonEnvelopeReplies : forced
    GraniteChatTemplate ..> ReplyLanguage : Spans think/tool_call
    GraniteChatTemplate ..> ToolCallSyntax : parseBlock / parseFunctionXml
    GraniteChatTemplate ..> JsonEnvelopeReplies : forced (JSON mode)
    MistralChatTemplate ..> ToolCallSyntax : parseObject
    Llama32ChatTemplate ..> ToolCallSyntax : parseBlock
```

---

## 3. Loading sequence

`LlamaProvider.loadLanguage` and `GraniteProvider.loadLanguage` share the same shape; only the template choice and stop list differ.

```mermaid
sequenceDiagram
    participant M as Models (jinfer-chat)
    participant P as LlamaProvider / GraniteProvider
    participant L as Llama / Granite
    participant T as GGUFTokenizerLoader
    participant ML as ModelLoader
    participant R as RoPE

    M->>P: loadLanguage(channel, gguf, path, arena, companions, tokenizer)
    P->>L: loadModel(channel, gguf, arena, tokenizer)
    alt tokenizer is null
        L->>T: build().fromGGUF(gguf)
    end
    L->>L: readConfiguration(gguf, arch, vocab size)
    Note over L: validates dims, rope, scales, vocab match,<br/>expert_count == 0 (no MoE)
    L->>ML: loadTensors(channel, gguf, arena)
    L->>L: buildRope(gguf, arch, config, tensors)
    alt rope.scaling.type yarn (Llama only)
        L->>R: yarn(ropeDim, theta, factor, origCtx, betaFast, betaSlow, 1, attnFactor)
    else no factors and factor 1
        L->>R: plain(ropeDim, theta)
    else rope_freqs.weight or factor
        L->>R: withFreqFactors(ropeDim, theta, values)
    end
    L->>L: loadWeights (blk.i.* tensors, wcls = output.weight or token_embd)
    L-->>P: model
    P->>P: choose ChatTemplate by general.architecture / specials
    P->>P: SpecialTokens.stops(tok, eos, ...)
    P-->>M: LoadedModel(model, tok, chat_template source, stops, modelSeed, template, SamplingDefaults.NONE)
```

---

## 4. Forward pass

`ingest(state, batch)` → `state.exclusively(forward)`. Every layer is attention + SwiGLU FFN; the only data carried across calls is the per-layer F16 KV cache. Only token batches are accepted (`Sequences` and `Embeddings` throw `UnsupportedOperationException`).

### 4.0 Global architecture

```mermaid
flowchart TB
    Tok["token ids [rows]"] --> Emb["gatherToF32 embeddings<br/>times embeddingScale"]
    subgraph Stack["Dense trunk"]
        direction TB
        subgraph Block["decoderBlock (pre-norm, residual x2 scaled by residualScale)"]
            direction LR
            A["RMSNorm → GQA attention<br/>interleaved RoPE (NoPE-aware), attn temp"] --> Add1(("+")) --> F["RMSNorm → SwiGLU FFN"] --> Add2(("+"))
        end
        Last["Llama: layers 0..N-2 eager<br/>last layer N-1 = K/V write only (lazy tail)"]
    end
    Emb --> Stack
    Stack --> FN["finalNorm RMSNorm (per queried row)"]
    FN --> H["LM head wcls gemv → logits [vocab]<br/>divided by logitScale"]
    Stack -.-> St[("State: F16 KV cache per layer<br/>carried across ingest calls")]
```

### 4a. `ingest` — validation, inputs, embedding, layer loop, advance

```mermaid
flowchart TB
    In["ingest(state, batch)"] --> Ex["state.exclusively(...)"]
    Ex --> N{"n = batch.count"}
    N -->|"n &lt;= 0"| E1["IllegalArgument: empty batch"]
    N -->|"n &gt; batchCapacity"| E2["IllegalArgument"]
    N -->|ok| P{"position + n &gt; contextCapacity"}
    P -->|yes| E3["IllegalArgument"]
    P -->|no| K{"batch.input kind"}
    K -->|Tokens| Ids["ids = t.ids"]
    K -->|Sequences| E4["Unsupported: packed sequences"]
    K -->|Embeddings| E5["Unsupported: text-only"]
    Ids --> V{"every id in 0..vocab"}
    V -->|no| E6["IllegalArgument token id"]
    V -->|yes| Rope["RoPE.fill(ropeCos, ropeSin, startPos, seqLen, ropeHalf, schedule)<br/>once per batch (Granite: only if useRope)"]
    Rope --> Emb["checkAlive; gatherToF32 token rows into residual<br/>multiplyInPlace embeddingScale if != 1"]
    Emb --> Loop{"layer l"}
    Loop -->|"Llama l &lt; N-1, Granite all l"| Blk["decoderBlock(l)  see 4b"]
    Blk --> Tr["Trace l_out-l when enabled"]
    Tr --> Loop
    Loop -->|"Llama l = N-1"| WK["writeKv(N-1): K/V only  see 4c"]
    WK --> Adv
    Loop -->|"Granite done"| Adv["state.advance(batch)<br/>advanceContext(count, outputs)"]
```

### 4b. `decoderBlock` — one layer

```mermaid
flowchart TB
    R0["residual [rows x dim]"] --> N1["Norms.rmsnormRows attnNorm → normed"]
    N1 --> Q["gemm wq → query"]
    N1 --> Kp["gemm wk → batchK"]
    N1 --> Vp["gemm wv → batchV"]
    Q --> B1["addBias bq / bk / bv if present"]
    Kp --> B1
    Vp --> B1
    B1 --> UR{"useRope(layer)<br/>Llama: noRopeLayerStep, Granite: config.useRope"}
    UR -->|yes| Rp["Parallel.forLoop rows:<br/>RoPE.applyInterleaved on every Q head and K head"]
    UR -->|"no (NoPE layer)"| Tmp
    Rp --> Tmp["Llama only: attnTemp(startPos + s) scales query row if != 1"]
    Tmp --> Att{"seqLen == 1"}
    Att -->|yes| FD["FlashAttention.flashDecode<br/>cache 0..startPos + batchK/V, window 0"]
    Att -->|no| FP["FlashAttention.slidingWindowPrefill<br/>causal, window 0, GQA kvMul"]
    FD --> CK["commitKv(layer): f32ToF16Rows batchK/batchV → keyCache/valueCache[l] at startPos*kvDim"]
    FP --> CK
    CK --> O["gemm wo attnOut → normed, addBias bo"]
    O --> A1["Ops.addScaledRows residual += residualScale * normed"]
    A1 --> N2["rmsnormRows ffnNorm → normed"]
    N2 --> G["gemm w1 → hidden, gemm w3 → hidden2, addBias b1 b3"]
    G --> S["Activations.siluMultiplyRows hidden, hidden2"]
    S --> D["gemm w2 hidden → normed, addBias b2"]
    D --> A2["addScaledRows residual += residualScale * normed"]
    A2 --> R1["residual (next layer input)"]
```

Notes: the KV commit happens **inside** the layer because `batchK/batchV` are single buffers reused by every layer (a deferred commit would write the last layer's values everywhere). There is no MoE and no per-head Q/K norm in this module (unlike `jinfer-qwen3`).

### 4c. Lazy last-layer tail and logits (`Llama` only)

`Llama` never computes the last layer's attention and FFN during `ingest`. It writes only that layer's K and V into the cache, leaving `residual` as the last layer's *input*. Each queried output row finishes the layer on demand in `logits()`. `Granite` instead runs every layer eagerly and applies `finalNorm` straight to `residual[row]`.

```mermaid
flowchart TB
    LG["logits(state, output)"] --> Ck{"output in 0..outputCount"}
    Ck -->|no| Er["IllegalArgument"]
    Ck -->|yes| Row["row = lastBatchSize - outputCount + output"]
    Row --> TA["tailAt(state, row)"]
    subgraph Tail["tailAt: last layer for ONE row, residual never written"]
        direction TB
        T1["rmsnorm attnNorm residual[row] → tailScratch"] --> T2["gemm wq → query (1 row), addBias bq"]
        T2 --> T3{"useRope(L)"}
        T3 -->|yes| T4["RoPE.fill for pos = startPos+row (row 0), applyInterleaved on Q heads"]
        T3 -->|no| T5
        T4 --> T5["attnTemp(pos) scaling"]
        T5 --> T6["flashDecode over cache 0..pos inclusive<br/>(batchK/V null: own K/V already in F16 cache via writeKv)"]
        T6 --> T7["gemm wo → tailScratch, addBias bo"]
        T7 --> T8["th = residual[row] + residualScale * tailScratch"]
        T8 --> T9["rmsnorm ffnNorm th → tailScratch"]
        T9 --> T10["feedForwardRow: w1 w3 silu w2, in place"]
        T10 --> T11["th += residualScale * tailScratch"]
    end
    TA --> Tail
    Tail --> FN["Norms.rmsnorm finalNorm th → normed"]
    FN --> HD["MatMul.gemv wcls normed → logits"]
    HD --> LS{"logitScale != 1"}
    LS -->|yes| Dv["Ops.divideInPlace logits by logitScale"]
    LS -->|no| Out["return s.logits"]
    Dv --> Out
```

---

## 5. Variant matrix and RoPE selection

All variants run through the same `Llama` forward pass; the GGUF metadata (with architecture-specific defaults) selects the behavior.

| Architecture | Class | Distinct behavior (from code) |
|---|---|---|
| `llama` | `Llama` | Plain RoPE, or `llama3` scaling through `rope_freqs.weight`; scales default 1 |
| `minicpm` | `Llama` | Defaults when keys absent: `embedding_scale` 12.0, `residual_scale` 1.4/sqrt(layers), `logit_scale` dim/256 |
| `mistral3` | `Llama` | YaRN RoPE (`rope.scaling.type = yarn`); attention temperature from `attention.temperature_scale` and `rope.scaling.original_context_length` |
| `smollm3` | `Llama` | `noRopeLayerStep = 4`: RoPE skipped on every 4th layer (1-indexed) |
| `granite` | `Granite` | Required `logit_scale`, optional embedding/residual scale (0 means 1), `attention.scale`; `rope.scaling.finetuned=false` disables RoPE entirely; YaRN and LongRoPE rejected |

```mermaid
flowchart TB
    S["buildRope(gguf, arch, config, tensors)"] --> T{"rope.scaling.type"}
    T -->|yarn Llama only| Y{"rope_freqs.weight present"}
    Y -->|yes| Ye["IllegalArgument: cannot combine with YaRN"]
    Y -->|no| Yo["RoPE.yarn(ropeDim, theta, factor, origCtx, betaFast, betaSlow, 1, attnFactor)<br/>attnFactor = mscale1 / mscaleAll / (1 + 0.1 ln f)"]
    T -->|"empty / none / linear / llama3"| L3{"llama3 without rope_freqs.weight"}
    L3 -->|yes| Le["IllegalArgument"]
    L3 -->|no| Fc["factor = rope.scaling.factor (none or 0 → 1)"]
    Fc --> Pl{"no freq factors and factor == 1"}
    Pl -->|yes| Pp["RoPE.plain(ropeDim, theta)"]
    Pl -->|no| Wf["RoPE.withFreqFactors(ropeDim, theta, values)<br/>values = rope_freqs scaled by factor, or constant factor"]
    T -->|other| Ue["IllegalArgument: unsupported type"]
```

---

## 6. Chat template selection and reply grammar

Only `Llama32ChatTemplate` frames prompts natively (and only without tools; with tools it throws `UnsupportedConversation`). All other templates throw from `encode`, so chat falls back to rendering the GGUF `tokenizer.chat_template` Jinja whole, while the template object still provides the reply parser, constrained decoding and forced calls.

```mermaid
flowchart TB
    A["LlamaProvider: general.architecture"] --> C{"which"}
    C -->|minicpm| M5["MiniCpm5ChatTemplate"]
    C -->|smollm3| S3["SmolLm3ChatTemplate"]
    C -->|mistral3| MI["MistralChatTemplate"]
    C -->|"llama (default)"| D1{"specials function and /function present"}
    D1 -->|yes| M5
    D1 -->|no| D2{"specials begin_of_text and start_header_id"}
    D2 -->|yes| L32["Llama32ChatTemplate"]
    D2 -->|no| None["Optional.empty: Jinja path only"]
    GP["GraniteProvider"] --> GT["GraniteChatTemplate<br/>functionXml iff im_end special exists"]
    M5 --> RL["Reply language"]
    S3 --> RL
    MI --> RL
    L32 --> RL
    GT --> RL
    RL --> R1["Spans think/tool_call: SmolLM3, Granite"]
    RL --> R2["Spans think/function XML: MiniCPM5"]
    RL --> R3["call [TOOL_CALLS] name [ARGS] json, close-less: Mistral"]
    RL --> R4["bare JSON name/parameters, UNDECIDED→CALL|CONTENT: Llama 3.x"]
```

Stop tokens (`LlamaProvider`): EOS id from `tokenizer.ggml.eos_token_id` first, then `<|eot_id|>`, `<|eom_id|>` (Llama 3.1+ tool-call turns), `<|im_end|>`, `<|endoftext|>`, `<|end_of_text|>`, and last `<|im_start|>`, `<|start_header_id|>` (a reply opening a new turn has ended). Absent names are skipped. `GraniteProvider` uses EOS, `<|end_of_text|>`, `<|eot_id|>`, `<|im_end|>`, `<|endoftext|>`. Sampling defaults are `NONE` for both.

---

## 7. State and checkpoint layout

Both `State` classes allocate from a `MemoryArena`: `newState(ctx, batch, arena)` borrows it (`ownsArena=false`); `newState(ctx, batch)` creates a cross-thread arena and owns it (closed on constructor failure). Requirements: `0 < contextCapacity <= maxContextLength`, `batchCapacity > 0`. `clearHistory()` is a no-op: stale KV rows beyond the cursor are attention-masked.

```mermaid
flowchart LR
    subgraph perBatch["Per-batch F32 scratch (sized by batchCapacity)"]
        R["residual [batch x dim]"]
        NM["normed [batch x dim]"]
        BK["batchK / batchV [batch x kvDim]"]
        QQ["query / attnOut [batch x queryDim]"]
        HH["hidden / hidden2 [batch x hiddenDim]"]
        RC["ropeCos / ropeSin [batch x ropeHalf]"]
    end
    subgraph single["Single row"]
        LGT["logits [1 x vocab]"]
        TH["th, tailScratch [1 x dim] (Llama only)"]
        DS["decodeScratch (FlashAttention)"]
    end
    subgraph kv["Per layer l, F16 [contextCapacity x kvDim]"]
        KC["keyCache[l]"]
        VC["valueCache[l]"]
    end
    BK -->|"commitKv f32ToF16Rows"| KC
    BK -->|"commitKv f32ToF16Rows"| VC
```

Checkpoint codec (`LlamaCheckpointCodec`, `GraniteCheckpointCodec`, identical): uniform full attention, linear indexing, no ring, no recurrent state, so every position is a resume point. RoPE is baked into K at absolute positions, so a restored row is valid at its slot.

```mermaid
flowchart TB
    SZ["sizeOf(positions) = positions x 4 x layers x kvDim bytes"] --> TR["transfer(state, from, to, blob, capture)"]
    TR --> EL["elements = (to - from) x kvDim<br/>elementOffset = from x kvDim"]
    EL --> LP{"for layer 0..layers-1"}
    LP --> KT1["KvTransfer.transfer keyCache[layer] ↔ blob at offset"]
    KT1 --> KT2["KvTransfer.transfer valueCache[layer] ↔ blob"]
    KT2 --> LP
    CAP{"capture"} -->|true| Cp["state → blob"]
    CAP -->|false| Rs["blob → state"]
    TR --> CAP
```

Blob layout per position range: `[K layer0 | V layer0 | K layer1 | V layer1 | ...]`, each segment `(to-from) x kvDim` elements (4 bytes per element-slot per K+V pair, i.e. F16 K plus F16 V).
