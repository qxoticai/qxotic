# jinfer-nemotronh — class architecture

Package `com.qxotic.jinfer.models.nemotronh` (5 classes, ~1350 lines). It is the `jinfer-<model>`
provider for GGUF architectures `nemotron_h` and `nemotron_h_moe` (hybrid **Mamba2 SSM / attention / MoE**,
text-only). Only the MoE variants run: a dense `nemotron_h` (no experts) and `expert_group_count > 1` are rejected at load.

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends;
classes marked `<<external>>` live outside this package.

1. [Position in the stack](#1-position-in-the-jinfer-stack)
2. [Class diagram](#2-class-diagram)
3. [Loading sequence](#3-loading-sequence)
4. [Forward pass](#4-forward-pass)
5. [MoE layer](#5-moe-layer)
6. [Chat template and tool flow](#6-chat-template-and-tool-flow)
7. [State and checkpoint layout](#7-state-and-checkpoint-layout)

| Class | Role |
|---|---|
| `NemotronH` | The model: `LanguageModel` impl + nested `Configuration`, `Weights`, `State`, `LayerType`, loaders and the three layer kernels' orchestration |
| `NemotronHProvider` | `ModelProvider` SPI entry (ServiceLoader), arch dispatch, builds `LoadedModel` |
| `NemotronHCheckpointCodec` | `CheckpointCodec<State>`: serialises KV rows + SSM conv/recurrent state |
| `NemotronHChatTemplate` | `ChatTemplate`: ChatML framing, think/direct seeds, tool flow, reply grammar (`Dialect` = CASCADE / LIGHTNING) |
| `NemotronToolSyntax` | Pure-text rendering of tool declarations (`<tools>` XML) and `<function=...>` call bodies |

---

## 1. Position in the jinfer stack

```mermaid
flowchart TB
    Callers["Callers: cli / server / langchain4j / spring-ai"]
    subgraph chat["jinfer-chat"]
        Models["Models (ServiceLoader discovery)"]
        MP["ModelProvider"]
        LM["LoadedModel"]
        CT["ChatTemplate / ReplyLanguage<br/>PromptWriter / ToolCallSyntax"]
    end
    subgraph core["jinfer-core"]
        LangModel["LanguageModel"]
        CS["ContextState"]
        CC["CheckpointCodec"]
        B["Batch"]
    end
    subgraph nemo["jinfer-nemotronh"]
        NP["NemotronHProvider"]
        NH["NemotronH"]
        NCT["NemotronHChatTemplate"]
        NTS["NemotronToolSyntax"]
        NCC["NemotronHCheckpointCodec"]
    end
    subgraph kern["jinfer-kernels"]
        K["MatMul · Norms · Mamba2 · Convolutions<br/>FlashAttention · Moe · Ops · Convert<br/>Activations · ModelLoader · KvTransfer"]
    end
    Tok["toknroll Tokenizer / GGUFTokenizerLoader"]

    Callers --> Models
    Models -->|"ServiceLoader"| NP
    NP -.->|"implements"| MP
    NP --> NH
    NP --> NCT
    NP -->|"produces"| LM
    NH -.->|"implements"| LangModel
    NH --> K
    NH --> Tok
    NCT -.->|"implements"| CT
    NCT --> NTS
    NCC -.->|"extends"| CC
    NH --> NCC
    NH -->|"State extends"| CS
```

---

## 2. Class diagram

```mermaid
classDiagram
    direction LR
    class LanguageModel~C,W,S~ {
        <<interface, external>>
        configuration() C
        weights() W
        newState(ctx, batch) S
        ingest(S, Batch)
        logits(S, int) MemoryView
        checkpointCodec() Optional
    }
    class ContextState {
        <<abstract, external>>
        exclusively(Runnable)
        advanceContext(count, outputs)
        clearHistory()*
    }
    class CheckpointCodec~S~ {
        <<abstract, external>>
        sizeOf(positions)*
        transfer(S, from, to, blob, capture)*
    }
    class ModelProvider {
        <<interface, external>>
        architectures() Set
        loadLanguage(...) LoadedModel
    }
    class ChatTemplate {
        <<interface, external>>
        encode(Conversation, batchCap, sink) ReplyState
        parser(Tokenizer) ReplyParser
        constrainedReply(grammar) Optional
        forcedCall(tools) Optional
    }

    class NemotronH {
        <<final>>
        -Configuration configuration
        -Tokenizer tokenizer
        -Weights weights
        +loadModel(Path, Arena)$ NemotronH
        +loadModel(channel, gguf, arena, tokenizer)$ NemotronH
        ~loadWeights(tensors, Configuration)$ Weights
        -forward(State, tokens, startPos, rows)
        -ssm(State, layer, rows)
        -attention(State, layer, startPos, rows)
        -moe(State, layer, rows)
        -projectLogits(State, output)
    }
    class Configuration {
        <<record>>
        embeddingLength, numberOfLayers
        numberOfHeads, numberOfKeyValueHeads, headSize
        LayerType[] layerTypes
        ssmInnerSize, ssmGroupCount, ssmTimeStepRank
        ssmStateSize, ssmConvKernel
        expertCount, expertUsedCount
        expertFeedForwardLength, expertSharedFeedForwardLength
        expertWeightsNorm, expertWeightsScale
        queryDim() kvDim()
        ssmConvChannels() ssmInProjSize()
    }
    class LayerType {
        <<enumeration>>
        SSM
        ATTENTION
        MOE
    }
    class Weights {
        <<record>>
        tokenEmbedding, outputNorm, outputWeight
        attnNorm[]
        AttentionWeights[] attention
        SsmWeights[] ssm
        MoeWeights[] moe
    }
    class AttentionWeights {
        <<record>>
        wq, wk, wv, wo
    }
    class SsmWeights {
        <<record>>
        inProj, conv1d, conv1dBias
        a, d, dtBias, norm, outProj
    }
    class MoeWeights {
        <<record>>
        router, expProbsB
        upExps[], downExps[]
        upShared, downShared
    }
    class State {
        <<static final>>
        residual, normed, branch, logits
        q k v attentionOut
        ssm* scratch views
        moe* scratch + Moe.Routing
        keyCache[] valueCache[] F16
        convState[] recurrentState[] F32
        clearHistory()
    }

    class NemotronHProvider {
        <<final>>
        architectures() Set
        loadLanguage(...) LoadedModel
    }
    class NemotronHCheckpointCodec {
        <<package-private>>
        -recurrentFloats, convFloats
        -bytesPerPosition, residueBytes
        sizeOf(positions)
        transfer(State, from, to, blob, capture)
    }
    class NemotronHChatTemplate {
        <<package-private>>
        +DEFAULT_SYSTEM$
        ~TOOL_INSTRUCTIONS$
        -Dialect dialect
        -imStart imEnd think endThink
        -toolCall toolResponse markers
        encode(...) ReplyState
        parser / constrainedReply / forcedCall
    }
    class Dialect {
        <<record>>
        defaultSystem
        reasoningNewlines
        trimsTruncatedCallTail
        CASCADE$
        LIGHTNING$
        of(templateSource)$
    }
    class NemotronToolSyntax {
        <<utility>>
        declarations(List~Tool~)$ String
        call(Content.ToolCall)$ String
    }

    LanguageModel <|.. NemotronH
    ContextState <|-- State
    CheckpointCodec <|-- NemotronHCheckpointCodec
    ModelProvider <|.. NemotronHProvider
    ChatTemplate <|.. NemotronHChatTemplate

    NemotronH *-- Configuration
    NemotronH *-- Weights
    NemotronH *-- State
    Configuration --> LayerType
    Weights *-- AttentionWeights
    Weights *-- SsmWeights
    Weights *-- MoeWeights
    NemotronH ..> NemotronHCheckpointCodec : creates
    NemotronHCheckpointCodec --> Configuration
    NemotronHCheckpointCodec ..> State : reads/writes
    NemotronHProvider ..> NemotronH : loadModel
    NemotronHProvider ..> NemotronHChatTemplate : creates
    NemotronHProvider ..> Dialect : Dialect.of(jinja)
    NemotronHChatTemplate *-- Dialect
    NemotronHChatTemplate ..> NemotronToolSyntax : uses
```

---

## 3. Loading sequence

```mermaid
sequenceDiagram
    participant M as Models (jinfer-chat)
    participant P as NemotronHProvider
    participant N as NemotronH
    participant L as ModelLoader
    participant T as GGUFTokenizerLoader
    participant C as NemotronHChatTemplate

    M->>P: loadLanguage(channel, gguf, path, arena, companions, tokenizer)
    P->>N: loadModel(channel, gguf, arena, tokenizer)
    N->>N: check arch in {nemotron_h, nemotron_h_moe}
    opt tokenizer == null
        N->>T: fromGGUF(gguf)
    end
    N->>N: read hparams, layers = block_count - nextn_predict_layers
    N->>N: per layer: ffn>0 → MOE, kv_heads>0 → ATTENTION, else SSM
    N->>N: validate (experts>0, groups<=1, divisibility)
    N->>L: loadTensors(channel, gguf, arena)
    L-->>N: Map name → MemoryView
    N->>N: loadWeights: blk.N.* per LayerType
    N-->>P: NemotronH(config, tokenizer, weights)
    P->>P: eos id, SpecialTokens.stops(im_end, endoftext)
    P->>C: new(tok, Dialect.of(chat_template))
    P-->>M: LoadedModel(model, tok, template, stops, seed, codec, NONE)
```

Layer-type inference has no explicit per-layer tag in the GGUF: it derives from the two per-layer arrays
`feed_forward_length` (>0 → MoE) and `attention.head_count_kv` (>0 → attention), with the remainder being Mamba2.

---

## 4. Forward pass

`ingest` → `state.exclusively(forward)` → validates batch (tokens only; no packed sequences, no embeddings),
then runs the hybrid stack. Every layer is a **pre-norm residual block** of exactly one mixer kind.

### 4.0 Global architecture

```mermaid
flowchart TB
    Tok["token ids [rows]"] --> Emb["Token embedding<br/>gather → F32 residual stream [rows × dim]"]
    subgraph Stack["Hybrid trunk: N layers, one mixer each, order given by GGUF layerTypes[]"]
        direction TB
        subgraph Block["Residual block (every layer)"]
            direction LR
            RN["RMSNorm<br/>attnNorm[layer]"] --> Mixer{{"mixer"}} --> Add(("+ residual"))
        end
        SSMB["SSM: Mamba2<br/>conv + selective scan<br/>state: conv ring + recurrent"]
        ATTB["ATTENTION: GQA causal<br/>no RoPE<br/>state: F16 KV cache"]
        MOEB["MOE: top-k routed ReLU² experts<br/>+ shared expert<br/>state: none"]
        Mixer -.- SSMB
        Mixer -.- ATTB
        Mixer -.- MOEB
    end
    Emb --> Stack
    Stack --> FN["Final RMSNorm<br/>outputNorm"]
    FN --> Head["LM head (gemv outputWeight, tied to embedding if absent)"]
    Head --> Logits["logits [vocab] for the requested output row"]
    Stack -.-> St[("State: KV caches + SSM states<br/>carried across ingest calls")]
```

### 4.1 Layer loop

```mermaid
flowchart TB
    In["token ids"] --> Emb["Convert.gatherToF32 → residual"]
    Emb --> Loop{"for layer in 0..N"}
    Loop --> Norm["Norms.rmsnormRows(residual → normed, attnNorm[layer])"]
    Norm --> Sw{"layerTypes[layer]"}
    Sw -->|SSM| S["ssm()"]
    Sw -->|ATTENTION| A["attention()"]
    Sw -->|MOE| E["moe()"]
    S --> Add["Ops.addRows(residual += branch)"]
    A --> Add
    E --> Add
    Add --> Loop
    Loop -->|done| Adv["state.advance(batch)"]
    Adv -.->|"later: logits(state, i)"| Out["rmsnorm(outputNorm) → MatMul.gemv(outputWeight) → logits"]
```

### 4a. SSM layer (Mamba2)

```mermaid
flowchart LR
    X["normed"] --> IP["gemm inProj"] --> Proj["ssmProjection<br/>[z | xBC | dt]"]
    Proj --> Z["ssmZ (gate)"]
    Proj --> XBC["ssmXbc"]
    Proj --> DT["dt = softplus(dt + dtBias)"]
    XBC --> Conv["Convolutions.causalDepthwiseSilu<br/>(convState[layer])"]
    Conv --> Scan["Mamba2.scan<br/>(A, D, recurrentState[layer])"]
    Z --> Scan
    DT --> Scan
    Scan --> GN["Mamba2.groupedRmsNorm"]
    GN --> OP["gemm outProj → branch"]
```

### 4b. Attention layer

```mermaid
flowchart LR
    X["normed"] --> Q["gemm wq → q"]
    X --> K["gemm wk → k"]
    X --> V["gemm wv → v"]
    K --> KC["f32ToF16Rows → keyCache[layer] @ startPos"]
    V --> VC["f32ToF16Rows → valueCache[layer] @ startPos"]
    Q --> FA["FlashAttention.causalPrefill (GQA)"]
    KC --> FA
    VC --> FA
    FA --> WO["gemm wo → branch"]
```

No RoPE or positional embedding is applied in the attention layers; position enters only through the KV cache index and the SSM recurrence.

---

## 5. MoE layer

Routed experts use sigmoid gating, a selection-only bias, and **ReLU²** FFNs (no gate projection: `up → relu² → down`).

```mermaid
flowchart TB
    X["normed"] --> R["gemm router → moeRouter"]
    R --> Sig["sigmoid in place"]
    Sig --> Bias{"expProbsB != null?"}
    Bias -->|yes| Sel["selection = probs + expProbsB (scratch copy)"]
    Bias -->|no| Sel2["selection = probs"]
    Sel --> TopK["Moe.selectTopK(selection, probs)<br/>→ moeRowTopE / moeRowTopP / expertCounts"]
    Sel2 --> TopK
    TopK --> Norm["if expertWeightsNorm: w /= max(Σw, 6.1e-5)<br/>then w *= expertWeightsScale"]
    Norm --> Disp["Moe.dispatch(routing, ...)"]
    Disp --> Ex["per expert: gather → gemm upExps[e] → reluSqr → gemm downExps[e]"]
    Ex --> Br["weighted scatter-add → branch"]
    X --> Sh{"upShared != null?"}
    Sh -->|yes| SE["gemm upShared → reluSqr → gemm downShared"]
    SE --> Br
```

Routed weights stay **unbiased**; `exp_probs_b` only influences which experts are picked (as in llama.cpp `build_moe_ffn`).

---

## 6. Chat template and tool flow

```mermaid
flowchart TB
    Enc["encode(Conversation, batchCapacity, sink)"] --> Norm["normalize: inject Dialect.defaultSystem if first msg is not SYSTEM"]
    Norm --> Shape{"no tools AND all Text parts AND no TOOL turns?"}
    Shape -->|yes| Plain["Plain shape: writePlainTurn per message, flushed per turn (turn-stable cached prefix)"]
    Shape -->|no| Check["requireToolShapes; needs &lt;tool_call&gt; and &lt;tool_response&gt; ids"]
    Check --> TC["writeToolConversation (whole flow)"]
    TC --> Sys["system turn + NemotronToolSyntax.declarations + TOOL_INSTRUCTIONS (trusted)"]
    TC --> Turns["user / assistant(+calls via NemotronToolSyntax.call) / folded tool responses"]
    Plain --> Gen
    Turns --> Gen["generation prompt: assistant + think(+newline) or think + endThink"]
    Sys --> Turns
    Gen --> RS["ReplyState(seedThinking | seedDirect, spans().parser())"]
```

Reply-side faces (all built from one lazily created `ReplyLanguage.Spans`):

```mermaid
flowchart LR
    Spans["ReplyLanguage.Spans<br/>think / tool_call markers<br/>ToolCallSyntax::parseFunctionXml<br/>stop = im_end"]
    Spans --> Parser["parser() / ReplyParser"]
    Spans --> Con["constrainedReply(grammar[, calls])"]
    Spans --> Forced["forcedCall(tools)<br/>header = &lt;function=NAME"]
```

Key rules:

| Concern | Behaviour |
|---|---|
| History thinking | Assistant turns before the last user are truncated to `<think></think>` + text after the last `</think>` |
| Dialect `CASCADE` | Default system injected, reasoning framed `<think>\n{R}\n</think>\n`, truncated call tails trimmed |
| Dialect `LIGHTNING` | Empty system, `<think>\n{R}</think>`, tails kept as-is. Chosen by `Dialect.of(template)` (looks for "You are a helpful and harmless assistant") |
| Injection safety | Tool names/descriptions are plain-encoded via `out.text`; only the fixed `TOOL_INSTRUCTIONS` goes through `PromptWriter.trusted`, and marker ids come from `out.id` |
| Tool results | Consecutive TOOL messages fold into one `user` turn of `<tool_response>` blocks; both `ToolResult` and `Text` parts accepted |

---

## 7. State, memory and checkpoint layout

```mermaid
flowchart TB
    subgraph State["NemotronH.State (per context)"]
        direction TB
        subgraph scratch["Scratch, sized by batchCapacity"]
            R["residual · normed · branch · logits"]
            AT["q · k · v · attentionOut"]
            SS["ssmProjection · ssmZ · ssmXbc · ssmDt · ssmConvOut · ssmOutput · ssmTmp"]
            MO["moeRouter · moeSelection · moeGather · moeDown · moeHidden · sharedHidden · sharedOut<br/>moeRowTopE/P · expertCounts · Moe.Routing"]
        end
        subgraph persist["Per layer, persistent"]
            KV["ATTENTION: keyCache / valueCache F16 [contextCapacity × kvDim]"]
            SM["SSM: convState F32 [(kernel-1) × channels]<br/>recurrentState F32 [inner × stateSize]"]
            NO["MOE: nothing"]
        end
    end
    subgraph Blob["NemotronHCheckpointCodec blob"]
        direction TB
        B1["attention layers in order: K rows then V rows for positions [from, to)<br/>= positions × bytesPerPosition"]
        B2["SSM layers in order: recurrentState + convState (full, position independent)<br/>= residueBytes"]
        B1 --> B2
    end
    KV -->|"transfer(capture)"| B1
    SM -->|"transfer(capture)"| B2
    B1 -->|"transfer(restore)"| KV
    B2 -->|"transfer(restore)"| SM
```

- `sizeOf(positions) = positions × Σ_attention(2 × 2 B × kvDim) + #ssmLayers × (inner×stateSize + (kernel-1)×channels) × 4 B`.
- The SSM share is a **fixed overhead**, independent of position count; restoring a prefix checkpoint therefore always rewrites the full recurrent state. MoE routing is per-token and carries no cross-token state.
- `clearHistory()` zero-fills `convState` and `recurrentState`; KV caches need no clearing (position-indexed, rewritten on ingest).
- `newState(ctx, batch)` creates a cross-thread arena owned by the state; `newState(ctx, batch, arena)` borrows the caller's arena.
