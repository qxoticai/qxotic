# jinfer-mellum — class architecture

Package `com.qxotic.jinfer.models.mellum` (4 classes, ~1090 lines). It is the `jinfer-<model>` provider for GGUF
architecture `mellum` (JetBrains Mellum 2): a **text-only mixture-of-experts decoder** with grouped-query attention,
per-head Q/K RMS-norm, NeoX rotary, and a 3:1 mix of sliding-window and full-attention layers. Every block routes
softmax top-k over its experts (k weights renormalised); there is no dense or shared expert. There is no vision, audio
or speculative subsystem: `Batch.Input.Embeddings` and `Batch.Input.Sequences` are rejected.

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends;
classes marked `<<external>>` live outside this package.

1. [Position in the stack](#1-position-in-the-jinfer-stack)
2. [Class diagrams](#2-class-diagrams) — 2a contracts, 2b model internals, 2c chat template
3. [Loading sequence](#3-loading-sequence)
4. [Forward pass](#4-forward-pass) — 4.0 global, 4a `ingest`, 4b `decoderBlock`
5. [Layer kinds: sliding window vs full](#5-layer-kinds-sliding-window-vs-full)
6. [State and checkpoint layout](#6-state-and-checkpoint-layout)
7. [Chat template flow](#7-chat-template-flow)

| Class | Role |
|---|---|
| `Mellum` | The model: `LanguageModel` impl with nested `Configuration`, `LayerWeights`, `Weights`, `State`; GGUF loading and the whole forward pass (`attention` + `moe` per layer) |
| `MellumProvider` | `ModelProvider` SPI for architecture `mellum`: builds the `LoadedModel` (stops, seed, chat template, sampling defaults 0.6 / 0.95 / 20) |
| `MellumChatTemplate` | `ChatTemplate`: ChatML + Hermes JSON tool calls, token-exact with the GGUF Jinja; Instruct vs Thinking checkpoints |
| `MellumCheckpointCodec` | `CheckpointCodec<State>`: serialises FP16 K/V rows only (linear for full layers, ring slots for window layers) |

---

## 1. Position in the jinfer stack

```mermaid
flowchart TB
    Callers["Callers: cli / server / langchain4j / spring-ai"]
    subgraph chat["jinfer-chat"]
        Models["Models (ServiceLoader)"]
        MP["ModelProvider"]
        LM2["LoadedModel"]
        CT["ChatTemplate / PromptWriter / ReplyLanguage"]
    end
    subgraph core["jinfer-core"]
        LMI["LanguageModel"]
        CS["ContextState"]
        CC["CheckpointCodec"]
        B["Batch (Tokens only here)"]
    end
    subgraph mel["jinfer-mellum"]
        P["MellumProvider"]
        M["Mellum"]
        T["MellumChatTemplate"]
        C["MellumCheckpointCodec"]
    end
    subgraph kern["jinfer-kernels"]
        K["MatMul, Norms, RoPE, FlashAttention<br/>Moe, Activations, Convert, Ops, KvTransfer, ModelLoader"]
    end
    Tok["toknroll Tokenizer"]

    Callers --> Models --> P
    P -.->|implements| MP
    P -->|produces| LM2
    M -.->|implements| LMI
    T -.->|implements| CT
    C -.->|extends| CC
    M --> K
    C --> K
    M --> Tok
    T --> Tok
    LM2 --> M
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
    class MellumProvider {
        <<final>>
        ~SAMPLING_DEFAULTS
        architectures() Set
        loadLanguage(...) LoadedModel
    }
    class Mellum {
        <<final>>
        ~ARCHITECTURE = mellum
    }
    class MellumChatTemplate { <<package-private>> }
    class MellumCheckpointCodec { <<package-private>> }
    class State { <<Mellum.State>> }

    ModelProvider <|.. MellumProvider
    LanguageModel <|.. Mellum
    ChatTemplate <|.. MellumChatTemplate
    ContextState <|-- State
    CheckpointCodec <|-- MellumCheckpointCodec
    MellumProvider ..> Mellum : loadModel
    MellumProvider ..> MellumChatTemplate : new(tok, reasons)
    Mellum ..> MellumCheckpointCodec : checkpointCodec()
    Mellum ..> State : newState
    MellumCheckpointCodec ..> State : reads/writes KV
```

### 2b. Model internals — `Mellum` and its nested types

```mermaid
classDiagram
    direction TB
    class Mellum {
        <<final>>
        -Configuration configuration
        -Tokenizer tokenizer
        -Weights weights
        +loadModel(path or channel, arena)$
        ~loadConfiguration(gguf, vocab)$
        -swaLayers(gguf, layers, window)$
        ~loadWeights(tensors, c)$
        +ingest(State, Batch)
        +logits(State, output) MemoryView
        -forward(State, Batch)
        -forward(State, tokens, startPos, rows)
        -attention(State, layer, startPos, rows)
        -headNormRope()
        -moe(State, layer, rows)
        -projectLogits()
    }
    class Configuration {
        <<record>>
        embeddingLength, numberOfLayers
        numberOfHeads, numberOfKeyValueHeads, headSize
        vocabularySize, maxContextLength, rmsNormEps
        expertCount, expertUsedCount, expertFeedForwardLength
        slidingWindow, isSwa[]
        ropeTheta, ropeScalingFactor
        ropeOriginalContext, ropeAttentionFactor
        queryDim() kvDim()
        kvCachePositions(layer, cap)
        kvCacheIndex(layer, pos)
        attentionStart(layer, pos)
    }
    class Weights {
        <<record>>
        tokenEmbedding, outputNorm, outputWeight
        LayerWeights[] layers
        RoPE.Schedule ropeFull
        RoPE.Schedule ropeSwa
    }
    class LayerWeights {
        <<record>>
        attnNorm, query, key, value
        queryNorm, keyNorm, output
        ffnNorm, router
        expertGate[], expertUp[], expertDown[]
    }
    class State {
        residual, normed, branch, logits
        query, attentionOut, batchK, batchV
        ropeCos/Sin Full and Swa
        moeRouter, moeGather, moeExpertOut
        moeHidden, moeHidden2
        moeExpertCounts, moeRowTopE, moeRowTopP
        keyCache[], valueCache[] FP16
        clearHistory() no-op
        -advance(Batch)
    }
    class RoPE_Schedule { <<external>> }
    class Moe_Routing { <<external>> }
    class DecodeScratch { <<external FlashAttention>> }

    Mellum *-- Configuration
    Mellum *-- Weights
    Weights *-- LayerWeights : layers
    Weights --> RoPE_Schedule : full (YaRN or plain), swa (plain)
    Mellum ..> State : newState
    State --> Configuration : sizes buffers
    State *-- Moe_Routing
    State *-- DecodeScratch
```

### 2c. Chat template internals

```mermaid
classDiagram
    direction LR
    class ChatTemplate { <<interface, external>> }
    class MellumChatTemplate {
        <<package-private>>
        ~TOOLS_HEAD
        ~TOOLS_TAIL
        -boolean reasons
        -int imStart, imEnd
        -IntSequence closedThink
        -Spans spans
        thinkingPolicy() OPTIONAL or NONE
        encode(conversation, cap, sink) ReplyState
        parser(tokenizer) ReplyParser
        constrainedReply(grammar)
        forcedCall(tools)
        -lastQuery(msgs)
        -writeTurn()
        -writeAssistant()
        -requireShapes()
    }
    class PromptWriter { <<external>> }
    class ReplyLanguage_Spans { <<external>> }
    class ToolCallSyntax { <<external>> }
    class JsonEnvelopeReplies { <<external>> }
    class SpecialTokens { <<external>> }

    ChatTemplate <|.. MellumChatTemplate
    MellumChatTemplate ..> PromptWriter : writes ids and text
    MellumChatTemplate *-- ReplyLanguage_Spans : think and tool_call spans
    MellumChatTemplate ..> ToolCallSyntax : jinjaJson, parseBlock
    MellumChatTemplate ..> JsonEnvelopeReplies : forced calls
    MellumChatTemplate ..> SpecialTokens : require markers
```

---

## 3. Loading sequence

```mermaid
sequenceDiagram
    participant Ms as Models
    participant P as MellumProvider
    participant M as Mellum
    participant ML as ModelLoader
    participant TL as GGUFTokenizerLoader
    participant R as RoPE
    participant T as MellumChatTemplate

    Ms->>P: loadLanguage(channel, gguf, path, arena, companions, tokenizer)
    P->>M: loadModel(channel, gguf, arena, tokenizer)
    M->>M: check general.architecture == mellum
    alt tokenizer is null
        M->>TL: fromGGUF(gguf)
        TL-->>M: Tokenizer
    end
    M->>M: loadConfiguration(gguf, vocab size)
    Note over M: reads block_count, head counts, key_length, expert_*, sliding_window,<br/>sliding_window_pattern (bool per layer, or period default 4), rope.*<br/>validates dims, MoE counts, window power of two, scaling none or yarn
    M->>ML: loadTensors(channel, gguf, arena)
    ML-->>M: Map name to MemoryView (mmap)
    M->>M: loadWeights(tensors, c)
    Note over M: per layer: norms, q/k/v/o, q/k norm, router,<br/>stacked experts sliced on leading axis<br/>output.weight falls back to token_embd
    M->>R: yarn(...) if scaling factor != 1 else plain, for full layers
    M->>R: plain(...) for window layers
    M-->>P: Mellum
    P->>P: eos id, chat_template string
    P->>T: new MellumChatTemplate(tok, template has enable_thinking)
    P-->>Ms: LoadedModel(model, tok, template, stops, seed, chatTemplate, sampling 0.6/0.95/20)
```

---

## 4. Forward pass

### 4.0 Global architecture

```mermaid
flowchart TB
    In["Batch of token ids"]
    Ingest["Mellum.ingest (State.exclusively)"]
    Embed["Embedding gather to F32 residual<br/>RoPE tables: full and swa"]
    subgraph Loop["for each layer"]
        Att["attention: pre-norm GQA<br/>window or full, KV commit"]
        Moe["moe: pre-norm softmax top-k experts, SwiGLU"]
    end
    Adv["State.advance: cursor and outputs"]
    Logits["logits(state, output)<br/>output rmsnorm, gemv to vocab"]
    Out["float logits view"]

    In --> Ingest --> Embed --> Loop --> Adv
    Adv -.-> Logits --> Out
    Att --> Moe
```

### 4a. `ingest` — batch validation to advance

```mermaid
flowchart TB
    S["ingest(state, batch)"] --> X["state.exclusively"]
    X --> V1{"rows in 1..batchCapacity?"}
    V1 -- no --> E1["IllegalArgumentException"]
    V1 -- yes --> V2{"position + rows &lt;= contextCapacity?"}
    V2 -- no --> E1
    V2 -- yes --> K{"batch.input kind"}
    K -- Tokens --> Ids["ids"]
    K -- Sequences --> E2["UnsupportedOperationException: no packed sequences"]
    K -- Embeddings --> E3["UnsupportedOperationException: text-only"]
    Ids --> V3{"each id in 0..vocab-1?"}
    V3 -- no --> E1
    V3 -- yes --> R1["RoPE.fill full tables (startPos, rows, headSize/2)"]
    R1 --> R2["RoPE.fill swa tables"]
    R2 --> G["Views.checkAlive; Convert.gatherToF32 token rows into residual"]
    G --> L{"layer &lt; numberOfLayers"}
    L -- yes --> A["attention(state, layer, startPos, rows)"]
    A --> F["moe(state, layer, rows)"]
    F --> TR["optional Trace.sum l_out-layer"]
    TR --> L
    L -- done --> Adv["state.advance(batch): advanceContext(count, outputs)"]
    Adv --> Fence["Reference.reachabilityFence(this)"]
```

### 4b. `decoderBlock` — one layer (attention then MoE)

```mermaid
flowchart TB
    subgraph ATT["attention(state, layer, startPos, rows)"]
        direction TB
        N1["rmsnormRowsGgml: residual to normed (attnNorm)"]
        Q["gemm Wq to query<br/>gemm Wk to batchK, gemm Wv to batchV"]
        HN["per row, Parallel.forLoop:<br/>headNormRope on query heads and K heads<br/>RMS-norm per head (queryNorm, keyNorm) then RoPE.applyNeox"]
        Tab{"isSwa[layer]?"}
        TabS["cos/sin = swa tables"]
        TabF["cos/sin = full tables (YaRN)"]
        Rows{"rows &gt; 1?"}
        Pre["FlashAttention.slidingWindowPrefill<br/>scale 1/sqrt(headSize)<br/>window = slidingWindow or 0"]
        Dec["FlashAttention.flashDecode<br/>start = attentionStart(layer, pos), decodeScratch"]
        O["gemm Wo to branch; Ops.addRows into residual"]
        KV["KV commit per row: slot = kvCacheIndex(layer, pos) * kvDim<br/>Convert.f32ToF16 into keyCache and valueCache"]
        N1 --> Q --> Tab
        Tab -- yes --> TabS --> HN
        Tab -- no --> TabF --> HN
        HN --> Rows
        Rows -- yes --> Pre --> O
        Rows -- no --> Dec --> O
        O --> KV
    end
    subgraph FFN["moe(state, layer, rows)"]
        direction TB
        N2["rmsnormRowsGgml: residual to normed (ffnNorm)"]
        Rt["gemm router to moeRouter (rows x expertCount)"]
        Sel["Moe.softmaxSelectTopK: softmax, top expertUsedCount,<br/>weights renormalised, per-expert counts"]
        Dp["Moe.dispatch: gather rows per expert"]
        Ex["per expert: gemm gate and up, siluMultiplyRows, gemm down"]
        Sc["scatter-add weighted outputs into branch"]
        Ad["Ops.addRows branch into residual"]
        N2 --> Rt --> Sel --> Dp --> Ex --> Sc --> Ad
    end
    ATT --> FFN
```

Notes: KV is committed inside the layer because `batchK` / `batchV` are reused by every layer. All layers are the same kind
of block (attention then MoE); they differ only in window, cache size and RoPE schedule (section 5).

---

## 5. Layer kinds: sliding window vs full

```mermaid
flowchart LR
    Cfg["attention.sliding_window_pattern<br/>(bool per layer, or period n, default 4)<br/>window = attention.sliding_window (power of two)"]
    Cfg --> Q{"window &gt; 0?"}
    Q -- no --> AllFull["no layer slides"]
    Q -- yes --> Per{"period form: layer % n != 0 (n = 0: all slide)"}
    Per -- slides --> SW["Window layer<br/>cache positions = min(capacity, window)<br/>index = pos &amp; (window-1) (ring)<br/>attention start = max(0, pos-window+1)<br/>RoPE: plain schedule"]
    Per -- not --> FL["Full layer<br/>cache positions = capacity<br/>index = pos<br/>attention start = 0<br/>RoPE: YaRN if scaling factor != 1 else plain"]
```

---

## 6. State and checkpoint layout

```mermaid
flowchart TB
    subgraph St["Mellum.State (allocated from MemoryArena)"]
        Act["F32 activations: residual, normed, branch (rows x dim)<br/>query, attentionOut (rows x queryDim), batchK, batchV (rows x kvDim)<br/>logits (1 x vocab)"]
        Rp["RoPE tables: cos/sin full and swa (rows x headSize/2)"]
        Mo["MoE scratch: moeRouter, moeGather, moeExpertOut,<br/>moeHidden, moeHidden2, counts, topE, topP, Routing"]
        Kc["Per layer FP16 keyCache / valueCache<br/>positions x kvDim: full = contextCapacity, window = min(capacity, window)"]
    end
    subgraph Cd["MellumCheckpointCodec"]
        Sz["sizeOf(n) = n * 2 * 2 bytes * layers * kvDim"]
        Tr["transfer(state, from, to, blob, capture)"]
        Lp{"for each layer: isSwa?"}
        Ring["KvTransfer.ringSpan on K then V<br/>(rows through ring slots pos &amp; (W-1))"]
        Lin["KvTransfer.transfer on K then V<br/>offset from * kvDim, (to-from) * kvDim elements"]
        Tr --> Lp
        Lp -- window --> Ring
        Lp -- full --> Lin
    end
    Kc --> Tr
    Note["No recurrent state: clearHistory is a no-op,<br/>rows beyond the cursor are masked, so every block is a resume point"]
    St --- Note
```

The checkpoint blob layout is, per layer in order, the K span followed by the V span; `capture = true` copies cache to blob,
`false` restores blob to cache. `newState(ctx, batch)` makes its own cross-thread arena (and closes it if allocation
fails); `newState(ctx, batch, arena)` borrows the caller's. `contextCapacity` must be in `(0, maxContextLength]`.

---

## 7. Chat template flow

```mermaid
flowchart TB
    Enc["encode(conversation, batchCapacity, sink)"]
    Chk{"messages empty or unsupported part?"}
    Enc --> Chk
    Chk -- yes --> UC["UnsupportedConversation"]
    Chk -- no --> Tools{"tools declared?"}
    Tools -- yes --> SysT["system turn: system text, TOOLS_HEAD,<br/>one jinjaJson per tool, TOOLS_TAIL"]
    Tools -- no --> Sys{"first message system?"}
    Sys -- yes --> SysP["plain system turn"]
    Sys -- no --> Loop
    SysT --> Loop
    SysP --> Loop
    Loop["for each non-system message"] --> Role{"role"}
    Role -- TOOL --> TR["fold consecutive results into one user turn<br/>of tool_response blocks"]
    Role -- ASSISTANT --> AS["writeAssistant: split reasoning from content<br/>echo think span only if Thinking and after last query<br/>then one tool_call JSON envelope per call"]
    Role -- other --> WT["writeTurn: im_start role, content, im_end"]
    TR --> Gen
    AS --> Gen
    WT --> Gen
    Gen["generation prompt: im_start assistant newline"] --> Th{"conversation.thinking?"}
    Th -- yes --> P0["empty reply prefix"]
    Th -- no --> P1["reply prefix = closedThink<br/>(empty for Instruct checkpoint)"]
    P0 --> Ret["ReplyState(prefix, spans.parser())"]
    P1 --> Ret
```

Policy: `thinkingPolicy()` is `OPTIONAL` for the Thinking checkpoint (template contains `enable_thinking`), else `NONE`.
Reply parsing, grammar-constrained replies and forced calls (`JsonEnvelopeReplies.forced(tools, "<|im_end|>")`) are
delegated to `ReplyLanguage.Spans` / `Selection`. Stop tokens come from the provider: GGUF eos, `<|im_end|>`, `<|endoftext|>`.
