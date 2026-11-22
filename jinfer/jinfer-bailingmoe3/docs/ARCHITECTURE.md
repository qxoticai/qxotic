# jinfer-bailingmoe3 — class architecture

Package `com.qxotic.jinfer.models.bailingmoe3` (5 classes, ~1860 lines). It is the `jinfer-<model>` provider for GGUF
architecture `bailingmoe3` (Ling / Bailing V3): a **text-only** hybrid decoder whose layers are either **KDA**
(Kimi Delta Attention, a gated delta-rule recurrent mixer with causal depthwise convolutions) or **MLA** (compressed
latent attention with an F16 latent cache), followed by a dense SwiGLU FFN (leading layers) or a sigmoid-gated grouped
MoE with a shared expert. An optional embedded single **NextN / MTP** layer drives native speculative decoding.
There is no vision, audio or packed-sequence input.

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends;
classes marked `<<external>>` live outside this package.

1. [Position in the stack](#1-position-in-the-jinfer-stack)
2. [Class diagrams](#2-class-diagrams) — 2a contracts, 2b model internals, 2c weights, 2d speculative and chat subsystems
3. [Loading sequence](#3-loading-sequence)
4. [Forward pass](#4-forward-pass) — 4.0 global, 4a `ingest`, 4b `decoderBlock` (+ KDA, MLA, MoE, MTP detail)
5. [Heads and logits](#5-heads-and-logits)
6. [Speculative decoding flow](#6-speculative-decoding-flow)
7. [Chat template flow](#7-chat-template-flow)
8. [State and checkpoint layout](#8-state-and-checkpoint-layout)

| Class | Role |
|---|---|
| `BailingMoe3` | The model: `LanguageModel` + `SpeculativeDecoding` impl with nested records `Configuration`, `KdaWeights`, `MlaWeights`, `FfnWeights`, `MoeWeights`, `NextNWeights`, `Weights` and the class `State`; GGUF loading, whole forward pass, logits, NextN drafting |
| `BailingMoe3Provider` | `ModelProvider` SPI for architecture `bailingmoe3`; builds the `LoadedModel` (stops, seed, chat template) |
| `BailingMoe3ChatTemplate` | `ChatTemplate`: native encoder of the Bailing V3 role format (`&lt;role&gt;HUMAN&lt;/role&gt;` ... `&lt;|role_end|&gt;`), tools as XML, reply parser with think/tool-call spans |
| `BailingMoe3CheckpointCodec` | `CheckpointCodec<State>`: MLA latent rows (per position) plus fixed KDA recurrent/conv state and the MTP pending hidden |
| `BailingMoe3Speculative` | Package-private draft-and-verify loop using the embedded NextN layer; owns the `Scratch` buffers |

---

## 1. Position in the jinfer stack

```mermaid
flowchart TB
    Callers["Callers: cli / server / langchain4j / spring-ai"]
    subgraph chat["jinfer-chat"]
        Models["Models (ServiceLoader)"]
        MP["ModelProvider"]
        LM1["LoadedModel"]
        CT["ChatTemplate / PromptWriter / ReplyLanguage"]
        Gen["Generator"]
    end
    subgraph core["jinfer-core"]
        LM["LanguageModel"]
        SD["SpeculativeDecoding"]
        CS["ContextState"]
        CC["CheckpointCodec"]
        B["Batch (Tokens only here)"]
    end
    subgraph bm["jinfer-bailingmoe3"]
        P["BailingMoe3Provider"]
        M["BailingMoe3"]
        T["BailingMoe3ChatTemplate"]
        C["BailingMoe3CheckpointCodec"]
        S["BailingMoe3Speculative"]
    end
    subgraph kern["jinfer-kernels"]
        K["MatMul, Norms, RoPE, FlashAttention, Convert, Ops<br/>Convolutions, KimiDeltaAttention, Moe, Activations, KvTransfer, ModelLoader"]
    end
    Tok["toknroll Tokenizer"]

    Callers --> Models --> P
    P -.->|implements| MP
    P -->|produces| LM1
    M -.->|implements| LM
    M -.->|implements| SD
    T -.->|implements| CT
    C -.->|extends| CC
    M --> K
    M --> Tok
    S --> M
    Gen --> LM1
    LM1 --> M
    LM1 --> T
```

---

## 2. Class diagrams

### 2a. Contracts and relations

```mermaid
classDiagram
    direction LR
    class ModelProvider { <<interface, external>> }
    class LanguageModel~C,W,S~ { <<interface, external>> }
    class SpeculativeDecoding~S~ { <<interface, external>> }
    class ChatTemplate { <<interface, external>> }
    class ContextState { <<abstract, external>> }
    class CheckpointCodec~S~ { <<abstract, external>> }
    class BailingMoe3Provider {
        <<final>>
        architectures() Set
        loadLanguage(...) LoadedModel
    }
    class BailingMoe3 {
        <<final>>
        loadModel(...)$
        newState(ctx, batch)
        ingest(State, Batch)
        logits(State, output)
        speculationReady()
        speculate(...)
        checkpointCodec()
    }
    class BailingMoe3ChatTemplate {
        <<package-private>>
        encode(Conversation, batchCapacity, sink)
        parser() constrainedReply() forcedCall()
    }
    class BailingMoe3CheckpointCodec { <<package-private>> }
    class BailingMoe3Speculative { <<package-private>> }
    class State { <<BailingMoe3.State>> }

    ModelProvider <|.. BailingMoe3Provider
    LanguageModel <|.. BailingMoe3
    SpeculativeDecoding <|.. BailingMoe3
    ChatTemplate <|.. BailingMoe3ChatTemplate
    ContextState <|-- State
    CheckpointCodec <|-- BailingMoe3CheckpointCodec
    BailingMoe3Provider ..> BailingMoe3 : loadModel
    BailingMoe3Provider ..> BailingMoe3ChatTemplate : new(tokenizer)
    BailingMoe3 ..> BailingMoe3CheckpointCodec : checkpointCodec()
    BailingMoe3 ..> BailingMoe3Speculative : speculate()
    BailingMoe3Speculative --> BailingMoe3 : draft / ingest / logitsAll
    BailingMoe3Speculative --> BailingMoe3CheckpointCodec : capture / restore
    BailingMoe3CheckpointCodec ..> State : reads/writes
    BailingMoe3 *-- State : newState
```

### 2b. Model internals

```mermaid
classDiagram
    direction TB
    class BailingMoe3 {
        <<final>>
        -Configuration configuration
        -Tokenizer tokenizer
        -Weights weights
        -forward(State, Batch)
        -forward(State, tokens, startPos, rows)
        -decoderBlock(s, layer, startPos, rows)
        -kda(s, layer, rows)
        -mla(s, layer, startPos, rows)
        -denseFfn(s, layer, rows)
        -moe(s, layer, rows)
        -swiglu(gate, up, rows, dim, clamp)$
        -mtpInputs() synchronizeMtp()
        draft(state, depth, candidates)
        -draftOne()
        logitsAll(state, destination)
        -loadConfiguration() validate() buildRope() loadWeights()$
    }
    class Configuration {
        <<record>>
        embeddingLength numberOfLayers nextnPredictLayers
        numberOfHeads vocabularySize maxContextLength
        rmsNormEps ropeTheta ropeDimensionCount
        boolean[] isAttention
        kdaHeadDim convKernel safeKdaGate kdaGateLowerBound
        qLoraRank kvLoraRank qkNopeDim mlaValueHeadDim
        feedForwardLength denseLeadingLayers
        expertCount expertUsedCount expertGroupCount expertGroupCountUsed
        expertFeedForwardLength sharedFeedForwardLength
        normalizeExpertWeights expertWeightsScale
        float[] expertSwiGluClamp sharedSwiGluClamp
        kdaInnerSize() mlaQkDim() mlaCacheDim() mlaPackedDim()
        storedLayers() hasMtp() mtpLayer()
    }
    class State {
        <<static final>>
        residual normed branch logits ropeCos ropeSin
        kda* scratch buffers
        mla* scratch buffers
        denseHidden denseHidden2
        moeRouter moeSelection moeGather moeDown moeHidden shared*
        targetHidden mtpConcat pendingHidden
        attentionCache[] F16
        qConvState[] kConvState[] vConvState[] recurrentState[] F32
        moeExpertCounts moeRowTopE moeRowTopP moeGroupScores moeGroupMask
        Moe.Routing moeRouting
        Scratch specScratch
        clearHistory() zeroes recurrent state
        advance(Batch)
    }
    BailingMoe3 *-- Configuration
    BailingMoe3 *-- State
    Configuration ..|> ContextConfiguration
```

### 2c. Weights

```mermaid
classDiagram
    direction LR
    class Weights {
        <<record>>
        tokenEmbedding
        outputNorm
        outputWeight (tied to embedding if absent)
        attnNorm[] ffnNorm[]
        RoPE.Schedule rope
        int ropeHalf
    }
    class KdaWeights {
        <<record>>
        q k v
        qConv kConv vConv
        f beta a dtBias
        outputGate outputNorm output
    }
    class MlaWeights {
        <<record>>
        q or (qA qANorm qB)
        kvA kvANorm
        kB[] per head
        vB[] per head
        gate output
    }
    class FfnWeights {
        <<record>>
        gate up down
    }
    class MoeWeights {
        <<record>>
        router selectionBias
        expertGate[] expertUp[] expertDown[]
        sharedGate sharedUp sharedDown
    }
    class NextNWeights {
        <<record>>
        embeddingNorm hiddenNorm
        inputProjection outputNorm
    }
    Weights *-- KdaWeights : kda[layer] if not attention
    Weights *-- MlaWeights : mla[layer] if attention
    Weights *-- FfnWeights : ffn[layer] if layer lt denseLeadingLayers
    Weights *-- MoeWeights : moe[layer] otherwise
    Weights *-- NextNWeights : nextn (null without MTP)
```

### 2d. Speculative and chat subsystems

```mermaid
classDiagram
    direction LR
    class BailingMoe3Speculative {
        <<package-private>>
        generate(model, state, maxTokens, timeout, stops, depth, sampler, listener, audit)$
        -walk(...)$
        -sample() row() result()$
    }
    class Scratch {
        verificationLogits F32 [depth+1 x vocab]
        codec BailingMoe3CheckpointCodec
        checkpoint MemorySegment
        depth
    }
    class BailingMoe3ChatTemplate {
        <<package-private>>
        -thinkOpen thinkClose token ids
        -ReplyLanguage.Spans spans
        encode() writeSystem() writeAssistant()
        writeCall() writeToolResults()
        assistantText() requireSupported()
        constrainedReply() forcedCall()
    }
    class PromptWriter { <<external>> }
    class ReplyLanguage { <<external>> }
    class ToolCallSyntax { <<external>> }
    class SpeculationAudit { <<external>> }
    class SpeculationResult { <<external>> }
    BailingMoe3Speculative *-- Scratch
    BailingMoe3Speculative ..> SpeculationResult : returns
    BailingMoe3Speculative ..> SpeculationAudit : onEmit
    BailingMoe3ChatTemplate ..> PromptWriter : writes prompt
    BailingMoe3ChatTemplate --> ReplyLanguage : Spans parser
    BailingMoe3ChatTemplate ..> ToolCallSyntax : parseTaggedXml, jinjaJson
```

---

## 3. Loading sequence

```mermaid
sequenceDiagram
    participant M as Models (jinfer-chat)
    participant P as BailingMoe3Provider
    participant B as BailingMoe3
    participant L as ModelLoader
    participant G as GGUFTokenizerLoader

    M->>P: loadLanguage(channel, gguf, path, arena, companions, tokenizer)
    P->>B: loadModel(channel, gguf, arena, tokenizer)
    B->>B: check general.architecture equals bailingmoe3
    opt tokenizer is null
        B->>G: fromGGUF(gguf)
    end
    B->>B: loadConfiguration (block_count minus nextn layers, head_count_kv per layer gives isAttention, sigmoid gating required)
    B->>B: validate (dims, KDA, MLA, MoE group constraints)
    B->>L: loadTensors(channel, gguf, arena)
    L-->>B: Map name to MemoryView
    B->>B: buildRope (plain, yarn or rope_freqs factors)
    B->>B: loadWeights per stored layer (KDA or MLA, dense FFN or MoE, optional NextN, output tied to embedding)
    B-->>P: BailingMoe3
    P->>P: eos from tokenizer.ggml.eos_token_id
    P-->>M: LoadedModel(model, tokenizer, chat_template, stops eos + role_end + endoftext, modelSeed, BailingMoe3ChatTemplate, SamplingDefaults.NONE)
```

---

## 4. Forward pass

`ingest(state, batch)` runs `state.exclusively(forward)`. Only `Batch.Input.Tokens` is accepted. Per-layer carried state is the F16 MLA latent cache (attention layers) or the KDA recurrent matrix plus three conv tails (KDA layers).

### 4.0 Global architecture

```mermaid
flowchart TB
    Tok["token ids [rows]"] --> Emb["Token embedding gather<br/>F32 residual stream [rows x dim]"]
    Emb -.->|"if MTP"| MI["mtpInputs: enorm of embeddings into mtpConcat"]
    Emb --> Stack
    subgraph Stack["numberOfLayers decoder blocks"]
        direction TB
        subgraph Block["decoderBlock (pre-norm residual x2)"]
            direction LR
            A["RMSNorm, then mixer:<br/>MLA if isAttention else KDA"] --> Add1(("+")) --> F["RMSNorm, then FFN:<br/>dense SwiGLU if layer lt denseLeadingLayers else MoE"] --> Add2(("+"))
        end
    end
    Stack --> Sync["synchronizeMtp (only if hasMtp)<br/>outputNorm into targetHidden, run NextN block on mtpLayer"]
    Stack --> FN["Final RMSNorm (outputNorm) at logits time"]
    Sync --> TH["targetHidden already normed"]
    FN --> LG["LM head gemv: logits [vocab]"]
    TH --> LG
    Stack -.-> St[("State: MLA F16 cache and KDA recurrent state<br/>carried across ingest calls")]
```

### 4a. `ingest` — validation, input kinds, rope/embedding, layer loop

```mermaid
flowchart TB
    In["ingest(state, Batch)"] --> Ex["state.exclusively"]
    Ex --> V{"rows in 1..batchCapacity<br/>and position + rows lte contextCapacity"}
    V -->|no| Err["IllegalArgumentException"]
    V -->|yes| Sw{"batch.input()"}
    Sw -->|Tokens| T["ids"]
    Sw -->|Sequences| S1["UnsupportedOperationException: packed sequences"]
    Sw -->|Embeddings| S2["UnsupportedOperationException: embedding input"]
    T --> RT{"each id in 0..vocab"}
    RT -->|no| Err2["IllegalArgumentException"]
    RT -->|yes| Emb["Views.checkAlive; Convert.gatherToF32<br/>tokenEmbedding rows into residual"]
    Emb --> M1{"hasMtp?"}
    M1 -->|yes| MI["mtpInputs: rmsnorm(embeddingNorm) per row into mtpConcat first half"]
    M1 -->|no| Rope
    MI --> Rope["RoPE.fill(ropeCos, ropeSin, startPos, rows)<br/>once per batch, consecutive positions"]
    Rope --> Loop{"for layer in 0..numberOfLayers"}
    Loop --> DB["decoderBlock(layer, startPos, rows)  see 4b"]
    DB --> Loop
    Loop -->|done| M2{"hasMtp?"}
    M2 -->|yes| SM["synchronizeMtp(startPos, rows)  see 4d"]
    M2 -->|no| Adv
    SM --> Adv["state.advance(batch): advanceContext(count, outputs)"]
    Adv -.->|later| Heads["logits / logitsAll  see 5"]
```

### 4b. `decoderBlock` — one layer

```mermaid
flowchart TB
    R0["residual [rows x dim]"] --> N1["Norms.rmsnormRows: normed (attnNorm[layer])"]
    N1 --> Kind{"isAttention[layer]?"}
    Kind -->|yes| MLA["mla(s, layer, startPos, rows)<br/>latent attention + KV commit  see 4c-i"]
    Kind -->|no| KDA["kda(s, layer, rows)<br/>conv + delta-rule scan  see 4c-ii"]
    MLA --> BR["branch [rows x dim]"]
    KDA --> BR
    BR --> A1["Ops.addRows: residual += branch"]
    A1 --> N2["Norms.rmsnormRows: normed (ffnNorm[layer])"]
    N2 --> Ffn{"layer lt denseLeadingLayers?"}
    Ffn -->|yes| D["denseFfn  see 4c-iii"]
    Ffn -->|no| E["moe  see 4c-iv"]
    D --> BR2["branch"]
    E --> BR2
    BR2 --> A2["Ops.addRows: residual += branch"]
    A2 --> Tr["Trace.sum l_out-layer when Trace.ENABLED"]
```

Layers differ in kind, so the mixer and FFN get their own sub-diagrams.

#### 4c-i. MLA mixer (attention layers) with KV commit

```mermaid
flowchart TB
    N["normed"] --> Qp{"w.q != null?"}
    Qp -->|"yes (qLoraRank 0)"| Q1["gemm q: mlaQAll"]
    Qp -->|"no"| Q2["gemm qA: mlaQA; rmsnormRows qANorm; gemm qB: mlaQAll"]
    N --> KV["gemm kvA: mlaKvAll [rows x (kvLoraRank + ropeDim)]"]
    KV --> Row["per row: rmsnorm kvANorm of latent part into mlaLatent, copy back<br/>RoPE.applyNeox on the rope tail (shared K rope)"]
    Q1 --> QR["per row, per head: RoPE.applyNeox on the rope tail of mlaQAll"]
    Q2 --> QR
    QR --> Absorb["per head: copy q nope part; gemm kB[head] into mlaQLatent<br/>pack [latent | rope] into mlaQPacked (absorbed K projection)"]
    Row --> Commit["Convert.f32ToF16Rows: mlaKvAll into attentionCache[layer] at startPos<br/>(commit KV inside the layer, one latent row per position)"]
    Absorb --> FA["FlashAttention.causalPrefill<br/>q=mlaQPacked, K=V=attentionCache[layer]<br/>kvHeads 1 view, scale 1/sqrt(qkNope+ropeDim)"]
    Commit --> FA
    FA --> VB["per head: copy latent part of mlaAttentionOut; gemm vB[head] into mlaVHead<br/>concat into mlaProjected"]
    N --> G["gemm gate: mlaGate [rows x heads]"]
    VB --> HS["KimiDeltaAttention.headSigmoidMultiply(mlaProjected, mlaGate)"]
    G --> HS
    HS --> O["gemm output: branch"]
```

#### 4c-ii. KDA mixer (recurrent layers)

```mermaid
flowchart TB
    N["normed"] --> Q["gemm q: kdaQ"]
    N --> K["gemm k: kdaK"]
    N --> V["gemm v: kdaV"]
    Q --> CQ["Convolutions.causalDepthwiseSilu with qConv and qConvState[layer]: kdaQConv"]
    K --> CK["causalDepthwiseSilu kConv, kConvState[layer]: kdaKConv"]
    V --> CV["causalDepthwiseSilu vConv, vConvState[layer]: kdaVConv"]
    N --> F["gemm f: kdaGateProjection"]
    N --> Be["gemm beta: kdaBetaProjection"]
    F --> Ga["KimiDeltaAttention.gates (dtBias, a, safeKdaGate, lower bound): kdaGate, kdaBeta"]
    Be --> Ga
    CQ --> Nq["normalizeQk (L2 per head)"]
    CK --> Nq
    Nq --> Sc["KimiDeltaAttention.scan<br/>updates recurrentState[layer] [heads x D x D], writes kdaOutput"]
    CV --> Sc
    Ga --> Sc
    N --> OG["gemm outputGate: kdaOutputGate"]
    Sc --> PN["postNorm(kdaOutput, kdaOutputGate, outputNorm): kdaNormed"]
    OG --> PN
    PN --> O["gemm output: branch"]
```

#### 4c-iii. Dense SwiGLU FFN and 4c-iv. MoE FFN

```mermaid
flowchart TB
    subgraph Dense["denseFfn (leading layers)"]
        direction TB
        DN["normed"] --> DG["gemm gate: denseHidden"]
        DN --> DU["gemm up: denseHidden2"]
        DG --> DS["Activations.siluMultiplyRows"]
        DU --> DS
        DS --> DD["gemm down: branch"]
    end
    subgraph Moe["moe (later layers)"]
        direction TB
        MN["normed"] --> Rt["gemm router: moeRouter; sigmoid in place"]
        Rt --> Sel["copy to moeSelection; add selectionBias row-wise"]
        Sel --> Top["Moe.selectTopKGrouped (expertGroupCount, expertGroupCountUsed, expertUsedCount)<br/>top experts from selection scores, weights from sigmoid scores"]
        Top --> Nm{"normalizeExpertWeights?"}
        Nm -->|yes| Nt["Moe.normalizeTopP"]
        Nm -->|no| Sk["multiply weights by expertWeightsScale"]
        Nt --> Sk
        Sk --> Dsp["Moe.dispatch: gather tokens per expert, callback per expert:<br/>gemm expertGate, gemm expertUp, swiglu(clamp), gemm expertDown"]
        MN --> SH["shared expert: gemm sharedGate, sharedUp, swiglu(clamp), gemm sharedDown: sharedOut"]
        Dsp --> Sum["branch += sharedOut"]
        SH --> Sum
    end
```

`swiglu` with `clamp` greater than 1e-6: SiLU on gate, gate clamped to at most `clamp`, up clamped to `[-clamp, clamp]`, then multiply; otherwise plain `siluMultiplyRows`. Clamps are per stored layer (`swiglu_clamp_exp`, `swiglu_clamp_shexp`).

#### 4d. MTP synchronisation

```mermaid
flowchart TB
    Res["residual after the main stack"] --> TH["rmsnormRows(outputNorm): targetHidden"]
    TH --> Loop["per row r: hidden = (r == 0 ? pendingHidden : targetHidden row r-1)<br/>rmsnorm(hiddenNorm) into mtpConcat second half"]
    Loop --> Keep["copy targetHidden last row into pendingHidden"]
    Keep --> Proj["gemm inputProjection(mtpConcat [enorm(emb) | hnorm(hidden)]): residual"]
    Proj --> Blk["decoderBlock(layer = numberOfLayers = mtpLayer)<br/>an MLA + MoE block with its own cache slot"]
```

---

## 5. Heads and logits

```mermaid
flowchart LR
    Out["logits(state, i)"] --> Row["row = lastBatchSize - outputCount + i"]
    Row --> Mtp{"hasMtp?"}
    Mtp -->|yes| T1["copy targetHidden row (already outputNorm-ed) into normed"]
    Mtp -->|no| T2["Norms.rmsnorm(outputNorm) of residual row into normed"]
    T1 --> Gv["MatMul.gemv(outputWeight, normed): logits [1 x vocab]"]
    T2 --> Gv
    LA["logitsAll(state, destination)"] --> Rows["same normalisation for all outputCount rows"]
    Rows --> Gm["MatMul.gemm(outputWeight): destination [rows x vocab]<br/>used for speculative verification"]
```

---

## 6. Speculative decoding flow

`speculationReady()` is true iff `weights.nextn != null`. `speculate` computes the token budget from the context room and `maxOutputTokens`, then calls `BailingMoe3Speculative.generate` (depth 1..8, capped at `batchCapacity - 1`). `draft` fills candidates greedily (argmax) by running the NextN block on one row at a time.

```mermaid
sequenceDiagram
    participant G as caller (Generator)
    participant M as BailingMoe3
    participant S as BailingMoe3Speculative
    participant C as BailingMoe3CheckpointCodec
    participant St as State

    G->>M: speculate(state, sampler, constraints, depth, listener, audit)
    M->>S: generate(...) lazily allocating Scratch (verificationLogits, checkpoint)
    S->>M: logits(state, last output), then sample next
    loop until maxTokens, context full or deadline
        S->>C: capture(state, base, base, checkpoint) when drafts gt 0
        S->>M: draft(state, drafts, candidates) with candidates[0] = next
        Note over M: per draft: rope at position, enorm(embedding) + hnorm(hidden), gemv inputProjection, decoderBlock(mtpLayer, 1 row), outputNorm, gemv logits, argmax
        S->>M: ingest(Batch.score(candidates))
        S->>M: logitsAll into verificationLogits
        S->>S: accept while sample(row i) equals candidate i+1, else take the target token
        S->>S: stop token inside accepted prefix gives keep = stop + 1
        alt keep lt drafts + 1
            S->>C: restore(state, base, base, checkpoint)
            S->>St: resumeAt(base)
            S->>M: ingest(Batch.score(candidates[0..keep)))
        end
        S->>G: listener.onIngested / onToken, audit.onEmit
    end
    S-->>M: SpeculationResult(emitted, committed, stop, finish, elapsed, drafted, accepted, forwards)
    Note over S: emits SpeculationEvent (JFR) when enabled
```

The checkpoint is needed because KDA state is not positional: rejecting drafts cannot simply rewind the cursor, so the recurrent state is restored and the accepted prefix is re-ingested.

---

## 7. Chat template flow

```mermaid
flowchart TB
    E["encode(Conversation, batchCapacity, sink)"] --> Chk["requireSupported per message<br/>text only for system/user, text+reasoning+toolcall for assistant, toolresult for tool"]
    Chk --> Sys["writeSystem: role SYSTEM, optional system text,<br/>tools block (tools XML json + call instructions),<br/>'detailed thinking on/off' unless declared; role_end"]
    Sys --> Fl1["out.flush"]
    Fl1 --> Loop{"each remaining message"}
    Loop -->|USER| U["role HUMAN + text + role_end"]
    Loop -->|SYSTEM| Sy["role SYSTEM + text + role_end"]
    Loop -->|ASSISTANT| As["role ASSISTANT, think open, reasoning, think close, visible text,<br/>tool_call blocks with arg_key / arg_value, role_end"]
    Loop -->|TOOL| To["role OBSERVATION once per run, tool_response blocks, role_end after the run"]
    U --> Loop
    Sy --> Loop
    As --> Loop
    To --> Loop
    Loop -->|done| Gen["role ASSISTANT newline + think open id"]
    Gen --> Th{"conversation.thinking()?"}
    Th -->|yes| P1["replyPrefix = think open"]
    Th -->|no| P2["add think close; replyPrefix = think open, think close"]
    P1 --> Fin["out.finish; return ReplyState(replyPrefix, spans.parser())"]
    P2 --> Fin
```

Reply parsing: `ReplyLanguage.Spans` over `&lt;think&gt;...&lt;/think&gt;`, `&lt;tool_call&gt;...&lt;/tool_call&gt;` with `ToolCallSyntax::parseTaggedXml`, turn end mark `&lt;|role_end|&gt;`. `constrainedReply` and `forcedCall` delegate to the same spans. Provider stop tokens: EOS id, `&lt;|role_end|&gt;`, `&lt;|endoftext|&gt;`.

---

## 8. State and checkpoint layout

```mermaid
flowchart TB
    subgraph State["BailingMoe3.State (per context)"]
        direction TB
        subgraph scratch["Scratch sized by batchCapacity"]
            A["residual, normed, branch, ropeCos/Sin, logits [1 x vocab]"]
            B["kda* buffers, mla* buffers, denseHidden*, moe* and shared* buffers, MoE routing int/float arrays"]
            C["MTP only: targetHidden, mtpConcat [b x 2 x dim], pendingHidden [1 x dim]"]
        end
        subgraph persist["Persistent per stored layer (storedLayers = numberOfLayers + nextnPredictLayers)"]
            KV["attention layer: attentionCache[l] F16 [contextCapacity x mlaCacheDim]<br/>mlaCacheDim = kvLoraRank + ropeDimensionCount"]
            KD["KDA layer: recurrentState[l] F32 [heads x D x D]<br/>q/k/vConvState[l] F32 [convKernel-1 x inner] each"]
        end
    end
    subgraph Blob["BailingMoe3CheckpointCodec blob"]
        direction TB
        B1["1 per attention layer in order: F16 latent rows for positions [from, to)"]
        B2["2 per KDA layer in order: recurrentState, qConv, kConv, vConv (FP32, fixed size)"]
        B3["3 if MTP: pendingHidden FP32 [dim]"]
    end
    KV -->|"transfer capture"| B1
    KD -->|"transfer capture"| B2
    C -->|"pendingHidden"| B3
    B1 -->|"transfer restore"| KV
    B2 -->|"transfer restore"| KD
```

- `sizeOf(positions) = positions x attentionLayers x 2 bytes x mlaCacheDim + residueBytes`; `residueBytes = recurrentLayers x (heads x D x D + convFloats) x 4` (plus `embeddingLength x 4` with MTP), where `convFloats = 3 x (convKernel - 1) x kdaInnerSize` as computed in the codec.
- The KDA part is position-independent: capture/restore always moves the whole recurrent state, whatever `from`/`to` are.
- `clearHistory()` zeroes all recurrent and conv state and `pendingHidden`; the MLA cache needs no clearing (rows beyond the cursor are masked).
- `newState(ctx, batch)` owns a cross-thread arena (closed on constructor failure); `newState(ctx, batch, arena)` borrows the caller's arena. `State` rejects `contextCapacity` above `maxContextLength`.
