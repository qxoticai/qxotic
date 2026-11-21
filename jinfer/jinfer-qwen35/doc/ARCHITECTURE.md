# jinfer-qwen35 — class architecture

Package `com.qxotic.jinfer.models.qwen35` (9 classes, ~4000 lines). It is the `jinfer-<model>`
provider for GGUF architectures `qwen35` (dense) and `qwen35moe` (MoE).

Format: **Mermaid** (renders natively on GitHub / GitLab / IntelliJ / VS Code). Each diagram answers one question:

1. [Where does the module sit?](#1-position-in-the-jinfer-stack) — layering
2. [Who implements what?](#2-class-diagrams) — types and relations (5 sub-diagrams)
3. [How is a model loaded?](#3-loading-sequence)
4. [What happens in one forward pass?](#4-forward-pass) — 4a `ingest`, 4b `decoderBlock` (+ attention / delta / FFN)
5. [How does speculative decoding work?](#5-speculative-decoding-mtp)
6. [How does the decision (d3) path work?](#6-decision-path-vllm-d3)
7. [What is in `State` and in a checkpoint?](#7-state-memory-and-checkpoint-layout)

Legend for class diagrams: `<<…>>` = stereotype; `..|>` implements; `--|>` extends; `-->` uses/holds;
`..>` creates/depends; classes with a grey tint are **outside** this package (contracts / kernels).

---

## 1. Position in the jinfer stack

```mermaid
flowchart TB
    subgraph apps["Callers"]
        CLI[jinfer-cli / jinfer-server<br/>langchain4j / spring-ai]
    end
    subgraph chat["jinfer-chat"]
        Models["Models.load<br/>(ServiceLoader)"]
        LM_["LoadedModel"]
        CT_["ChatTemplate / ReplyLanguage<br/>PromptWriter / ToolCallSyntax"]
    end
    subgraph decision["jinfer-decision"]
        DM_["DecisionModel<br/>Question: Predicate / Choice / Rating"]
    end
    subgraph core["jinfer-core"]
        Ctr["LanguageModel · SpeculativeDecoding<br/>Multimodal · MediaProjector<br/>ContextState · CheckpointCodec<br/>Batch · Media · Content"]
    end
    subgraph kern["jinfer-kernels (CPU, MemoryView)"]
        K["MatMul · Norms · RoPE · FlashAttention<br/>GatedDeltaNet · Convolutions · Moe<br/>Convert · Ops · ModelLoader · Resample · KvTransfer"]
    end
    subgraph q35["jinfer-qwen35  ◀── this module"]
        Q["Qwen35Provider · Qwen35 · Qwen35Vision<br/>Qwen35ChatTemplate · Qwen35Decision<br/>Qwen35Speculative · Qwen35CheckpointCodec<br/>Hadamard · Qwen35VisionPreprocess"]
    end
    ext["jota (MemoryView, Arena, DataType)<br/>gguf (GGUF) · toknroll (Tokenizer)"]

    CLI --> Models --> Q
    Q -. "implements" .-> Ctr
    Q -. "implements" .-> CT_
    Q -. "implements" .-> DM_
    Q --> K
    Q --> ext
    K --> ext
    Models --> LM_
```

Discovery: `META-INF/services/…ModelProvider` → `Qwen35Provider`.

---

## 2. Class diagrams

The class diagram is split into five focused views. 2a is the overview; 2b–2e zoom into one area each.
Classes shaded `<<interface>>` / `<<abstract>>` come from other modules (core, chat, decision).

### 2a. Overview — the nine classes and who creates / uses whom

```mermaid
classDiagram
    direction LR
    class Qwen35Provider { <<public, ServiceLoader>> }
    class Qwen35 { <<public final, the model>> }
    class Qwen35Vision { <<final>> }
    class Qwen35VisionPreprocess { <<util>> }
    class Qwen35ChatTemplate { <<final>> }
    class Qwen35Decision { <<final>> }
    class Qwen35Speculative { <<util>> }
    class Qwen35CheckpointCodec { <<final>> }
    class Hadamard { <<final>> }

    Qwen35Provider ..> Qwen35 : loadModel + withMedia
    Qwen35Provider ..> Qwen35ChatTemplate : new(model)
    Qwen35Provider ..> Qwen35Decision : new(model, gguf)
    Qwen35 o-- Qwen35Vision : vision (via withMedia)
    Qwen35 o-- Hadamard : nullable
    Qwen35 ..> Qwen35CheckpointCodec : checkpointCodec()
    Qwen35 ..> Qwen35Speculative : speculate()
    Qwen35Vision ..> Qwen35VisionPreprocess : smartResize
    Qwen35ChatTemplate --> Qwen35 : media (as Multimodal)
    Qwen35Decision --> Qwen35 : model
    Qwen35Speculative ..> Qwen35 : draft / ingest / logitsAll
    Qwen35Speculative ..> Qwen35CheckpointCodec : snapshot / rollback
```

### 2b. Contracts — which external interfaces each class implements

```mermaid
classDiagram
    direction LR
    class ModelProvider { <<interface>> }
    class LanguageModel~C,W,S~ { <<interface>> }
    class SpeculativeDecoding~S~ { <<interface>> }
    class Multimodal { <<interface>> }
    class MediaProjector~Image~ { <<interface>> }
    class ChatTemplate { <<interface>> }
    class DecisionModel { <<interface>> }
    class ContextState { <<abstract>> }
    class CheckpointCodec~S~ { <<abstract>> }

    ModelProvider <|.. Qwen35Provider
    LanguageModel <|.. Qwen35
    SpeculativeDecoding <|.. Qwen35
    Multimodal <|.. Qwen35
    Multimodal <|.. Qwen35Decision
    MediaProjector <|.. Qwen35Vision
    ChatTemplate <|.. Qwen35ChatTemplate
    DecisionModel <|.. Qwen35Decision
    ContextState <|-- State
    CheckpointCodec <|-- Qwen35CheckpointCodec
    class State { <<Qwen35.State>> }
```

### 2c. Core model — `Qwen35` and its nested types

```mermaid
classDiagram
    direction TB
    class Qwen35 {
        <<public final>>
        -Configuration configuration
        -Tokenizer tokenizer
        -Weights weights
        -MediaProjector vision
        -Hadamard hadamard (nullable)
        +loadModel(...)$
        +withMedia(mmproj, arena)
        +ingest(state, Batch)
        +logits(state, output)
        +speculate(...)
        ~ingestPrompt(state, tokens, Placed[])
        ~logits(state, row, tokens[], out)
        ~draft(state, depth, candidates)
        ~logitsAll(state, dest)
        -decoderBlock() attention() delta()
        -denseFfn() moe() mtpInputs() synchronizeMtp()
    }
    class Configuration {
        <<record>>
        dims, heads, ssm*, expert*
        isFullAttention[]
        nextnPredictLayers
        causalAttention
    }
    class Weights {
        <<record>>
        per-layer MemoryView[] (attn / ssm / ffn / moe)
        RoPE.Schedule rope
    }
    class NextNWeights { <<record, MTP head>> }
    class State {
        <<public static final>>
        residual normed branch logits
        keyCache valueCache (F16)
        convState recurrentState (F32)
        moe* scratch, ropeDelta
        specScratch
    }
    class Placed {
        <<record>>
        at, rows, count, positions
    }
    class Hadamard {
        <<final>>
        read(gguf)$ / inverse(rows)
    }

    Qwen35 *-- Configuration
    Qwen35 *-- Weights
    Weights *-- NextNWeights
    Qwen35 *-- State : nested type
    Qwen35 *-- Placed : nested type
    Qwen35 o-- Hadamard : nullable
    Hadamard ..> Configuration : read()
    State ..> Configuration : sizes buffers
```

### 2d. Multimodal and chat — vision tower and chat template

```mermaid
classDiagram
    direction LR
    class Qwen35 { <<public final>> }
    class Multimodal { <<interface>> }
    class MediaProjector~Image~ { <<interface>> }
    class ChatTemplate { <<interface>> }
    class Qwen35Vision {
        <<final>>
        ViT tower + 2x2 merger
        project() decoderPositions()
        positions() planId()
        Layer[] / Linear mm0, mm2
    }
    class Qwen35VisionPreprocess {
        <<util>>
        smartResize()
    }
    class Qwen35ChatTemplate {
        <<final>>
        ChatML framing, tool XML
        think seeds
        parser() constrainedReply() forcedCall()
    }

    Multimodal <|.. Qwen35
    MediaProjector <|.. Qwen35Vision
    ChatTemplate <|.. Qwen35ChatTemplate
    Qwen35 o-- Qwen35Vision : withMedia()
    Qwen35Vision ..> Qwen35VisionPreprocess : smartResize
    Qwen35ChatTemplate --> Multimodal : media (= Qwen35)
    Qwen35 ..> Qwen35ChatTemplate : created by Provider
```

### 2e. Decision, speculation and checkpointing

```mermaid
classDiagram
    direction LR
    class Qwen35 { <<public final>> }
    class State { <<Qwen35.State>> }
    class Placed { <<record>> }
    class Qwen35Decision {
        <<final>>
        vLLM d3 prompt + code-logit readout
        codes A..Z, AA..ZZ
        decide() decideIndependently()
    }
    class Qwen35Speculative {
        <<util>>
        generate() walk()
    }
    class Scratch {
        <<static final>>
        verificationLogits
        checkpoint segment
    }
    class Qwen35CheckpointCodec {
        <<final>>
        capture() restore()
        sizeOf() transfer()
    }
    class CheckpointCodec~S~ { <<abstract>> }
    class DecisionModel { <<interface>> }

    DecisionModel <|.. Qwen35Decision
    CheckpointCodec <|-- Qwen35CheckpointCodec
    Qwen35Decision --> Qwen35 : model
    Qwen35Decision ..> Placed : builds
    Qwen35Decision ..> State : state instanceof
    Qwen35 ..> Qwen35Speculative : speculate()
    Qwen35 ..> Qwen35CheckpointCodec : checkpointCodec()
    Qwen35Speculative *-- Scratch
    Scratch --> Qwen35CheckpointCodec
    State o-- Scratch : specScratch
    Qwen35Speculative ..> Qwen35 : draft / ingest / logitsAll
    Qwen35CheckpointCodec ..> State : transfer()
```

### Relationship summary

| From | To | Kind | Meaning |
|---|---|---|---|
| `Qwen35Provider` | `Qwen35` | creates | `Qwen35.loadModel(...)`, then `.withMedia(mmproj)` when a `media` companion exists |
| `Qwen35Provider` | `Qwen35ChatTemplate` | creates | wrapped in `LoadedModel` as the reply codec (tool calls, constrained decoding, forced calls) |
| `Qwen35Provider` | `Qwen35Decision` | creates | only when GGUF has `<arch>.decision.type` (d3); otherwise language path. The two paths are mutually exclusive (`IncompatibleModelException`) |
| `Qwen35` | `Qwen35Vision` | holds | as `MediaProjector<Media.Image>`; `withMedia` returns a **new** `Qwen35` sharing config/weights/hadamard |
| `Qwen35` | `Hadamard` | holds (nullable) | only for Prism-folded checkpoints; rotates embeddings and projection inputs |
| `Qwen35` | `Qwen35CheckpointCodec` | creates | on each `checkpointCodec()` call; prompt-cache save/restore |
| `Qwen35` | `Qwen35Speculative` | delegates | `speculate(...)` → `Qwen35Speculative.generate` |
| `Qwen35Speculative` | `Qwen35CheckpointCodec` | uses | snapshot before drafting, rollback on rejection (recurrent state cannot be "un-advanced") |
| `Qwen35Decision` | `Qwen35` | uses | package-private `ingestPrompt` + `logits(state,row,tokens,out)` |
| `Qwen35Decision` / `Qwen35ChatTemplate` | `Multimodal` | via `Qwen35` | image positions and rows come from the vision tower |
| `Qwen35Vision` | `Qwen35VisionPreprocess` | uses | `smartResize` for grid size; also used by `positions()` so token count is known without encoding |

---

## 3. Loading sequence

```mermaid
sequenceDiagram
    autonumber
    participant M as Models.load (jinfer-chat)
    participant P as Qwen35Provider
    participant Q as Qwen35
    participant H as Hadamard
    participant V as Qwen35Vision
    participant L as ModelLoader (kernels)

    M->>P: architectures() ∋ gguf arch ?
    M->>P: loadLanguage(channel, gguf, path, arena, companions, tokenizer)
    P->>P: refuse if decision.type set or attention.causal=false
    P->>Q: loadModel(channel, gguf, arena, tokenizer)
    Q->>Q: readConfiguration(gguf)
    Q->>L: loadTensors(channel, gguf, arena)
    Q->>H: Hadamard.read(gguf, config, tensorNames)
    H-->>Q: Hadamard or null
    Q->>Q: loadWeights + validateWeights (incl. nextn MTP block)
    opt companions["media"] (mmproj)
        P->>Q: withMedia(mmproj, arena)
        Q->>L: readGguf + loadTensors(mmproj)
        Q->>Q: check clip.vision.projection_dim == embeddingLength
        Q->>V: loadModel(path, gguf, tensors)
        Q-->>P: new Qwen35(.., vision=tower)
    end
    P->>P: new Qwen35ChatTemplate(model)
    P-->>M: LoadedModel(model, tokenizer, chat_template, stops, seed, template, SamplingDefaults)
```

`loadDecisionModel` follows the same steps, then `new Qwen35Decision(model, gguf)` instead of the template.

---

## 4. Forward pass

The forward pass is two nested levels. **4a** is the outer one, `Qwen35.ingest` (one batch → embeddings →
N layers → MTP sync → advance). **4b** is the inner one, `decoderBlock` (one layer: norm → token mixer →
residual → norm → FFN → residual). Everything runs on `State` scratch buffers; no per-call allocation.

### 4-0. Global view

```mermaid
flowchart LR
    IN["Batch<br/>Tokens or media Embeddings"] --> EMB["Embedding<br/>token lookup, or media rows as-is<br/>+ RoPE angles"]
    EMB --> STACK
    subgraph STACK["N decoder blocks (isFullAttention[] picks the mixer per layer)"]
        direction LR
        L0["block 0"] --> L1["block 1"] --> LD["..."] --> LN["block N-1"]
    end
    subgraph BLOCK["one decoderBlock (4b)"]
        direction TB
        B1["rmsnorm"] --> B2{"layer type"}
        B2 -- "full attention" --> B3["gated GQA attention<br/>+ RoPE + KV cache"]
        B2 -- "linear" --> B4["Gated DeltaNet<br/>conv + recurrent state"]
        B3 --> B5["+ residual"]
        B4 --> B5
        B5 --> B6["rmsnorm"] --> B7{"model type"}
        B7 -- "qwen35" --> B8["dense FFN (SwiGLU)"]
        B7 -- "qwen35moe" --> B9["MoE: router, top-k experts, shared expert"]
        B8 --> B10["+ residual"]
        B9 --> B10
    end
    STACK -.-> BLOCK
    LN --> OUTN["output rmsnorm"]
    OUTN --> HEAD["lm head gemv/gemm<br/>logits"]
    OUTN --> MTP["MTP layer (optional)<br/>drafts next tokens"]
    STATE[("State<br/>KV cache / conv + recurrent state<br/>scratch buffers, position")] -.- STACK
    STATE -.- MTP
    HEAD --> SMP["Sampler / Decision / Speculative"]
    MTP --> SMP
```

### 4a. `ingest` — one batch through the whole network

Entry: `ingest(state, batch)` → `state.exclusively(() -> forward(state, batch))` + `Reference.reachabilityFence(this)`.

```mermaid
flowchart TD
    A["ingest(state, Batch)<br/>state.exclusively(...)"] --> V1["validate<br/>rows = batch.count() &gt; 0<br/>rows ≤ batchCapacity<br/>start + rows ≤ contextCapacity<br/>requireWholePrompt (two-way checkpoint: start must be 0)"]
    V1 --> B{"batch.input()"}

    B -- "Sequences" --> X["UnsupportedOperationException<br/>(no packed sequences)"]

    B -- "Tokens (ids[])" --> T0["check every id in [0, vocab)"]
    T0 --> T1["RoPE.fill(ropeCos, ropeSin,<br/>pos = start + ropeDelta, rows)<br/>1-D positions: the same angle on all 3 axes"]
    T1 --> T2["Convert.gatherToF32(tokenEmbedding, ids) → residual<br/>(dequantises just the rows needed)"]
    T2 --> T3{"hadamard != null?"}
    T3 -- yes --> T4["hadamard.inverse(residual)<br/>embedding stores rotated rows: D·(H·z)"]
    T3 -- no --> T5
    T4 --> T5{"hasMtp()?"}
    T5 -- yes --> T6["gatherToF32(nextn.tokenEmbedding, ids) → normed<br/>mtpInputs: rmsnorm(embeddingNorm)<br/>→ mtpConcat[row, 0..dim)"]
    T5 -- no --> LAYERS
    T6 --> LAYERS

    B -- "Embeddings (media rows)" --> E0["check width == embeddingLength<br/>cast to segment-backed view"]
    E0 --> E1{"batch.positions()"}
    E1 -- "null (plain rows)" --> E2["RoPE.fill(pos = start + ropeDelta)"]
    E1 -- "3-D positions" --> E3["fillMediaRope:<br/>per row, per axis d∈{t,y,x}:<br/>rope.angles(base + positions.value(row,d))<br/>lanes picked by ropeDimensionSections"]
    E2 --> E4["copyF32(rows → residual)<br/>(each row IS its own embedding)"]
    E3 --> E4
    E4 --> E5{"hasMtp()?"}
    E5 -- yes --> E6["mtpInputs(residual) → mtpConcat[.., 0..dim)"]
    E5 -- no --> LAYERS
    E6 --> LAYERS

    LAYERS["for layer = 0 .. numberOfLayers-1<br/>decoderBlock(state, layer, start, rows)  — see 4b"]
    LAYERS --> S{"hasMtp()?"}
    S -- yes --> Y["synchronizeMtp (below)"]
    S -- no --> P
    Y --> P{"media batch with positions?"}
    P -- yes --> P1["ropeDelta += positions.advance − rows<br/>(a picture occupies `rows` cache slots but only<br/>max(w,h) rope positions)"]
    P -- no --> Z
    P1 --> Z["state.advance(batch)<br/>position += rows, record outputs"]
```

**Notes on 4a**

- **Three input kinds, one body.** `Tokens` and `Embeddings` differ only in how `residual` is filled and
  which RoPE angles are written; after that both run the same layer loop. `Sequences` (packed
  multi-sequence) is rejected.
- **`ropeDelta`.** Cache index and RoPE position diverge after a picture: the tower's grid spans
  `max(w,h)` positions but takes `w·h` cache rows. `ropeDelta` accumulates the difference so every later
  text row gets `cachePosition + ropeDelta`. It is saved in checkpoints (see 7) and reset by `clearHistory`.
- **Hadamard.** Only for Prism-folded checkpoints: the embedding is rotated back after lookup (here), and
  every projection input is rotated forward inside `folded(...)` (see 4b).
- **MTP rows.** The MTP block needs `[norm(embedding(token_i)) ‖ norm(hidden_{i-1})]` per row. The
  embedding half is written *before* the layers (they reuse the scratch it was computed in); the hidden half
  is written *after* them:

```mermaid
flowchart LR
    subgraph before["before the layers — mtpInputs"]
        m1["embedding(token_i)"] --> m2["rmsnorm(nextn.embeddingNorm)"] --> m3["mtpConcat[i, 0..dim)"]
    end
    subgraph after["after the layers — synchronizeMtp"]
        h1["rmsnorm(outputNorm, residual)<br/>→ targetHidden (all rows)"]
        h1 --> h2["row 0: pendingHidden (last row of previous batch)<br/>row i&gt;0: targetHidden[i-1]"]
        h2 --> h3["rmsnorm(nextn.hiddenNorm)<br/>→ mtpConcat[i, dim..2·dim)"]
        h1 --> h4["pendingHidden := targetHidden[rows-1]"]
        h3 --> h5["gemm(nextn.inputProjection, mtpConcat) → residual"]
        h5 --> h6["decoderBlock(layer = numberOfLayers)<br/>writes the MTP layer's own KV for these rows"]
    end
    before --> after
```

After `ingest`, `logits(state, output)` / `logitsAll` read the final hidden row(s): with MTP from
`targetHidden` (already output-normed), otherwise `rmsnorm(outputNorm)` over `residual`; then
`gemv/gemm(outputWeight)` (through `folded` if rotated).

### 4b. `decoderBlock` — one layer (pre-norm residual block)

Same function for the 0..N-1 main layers and for the single MTP layer (index `numberOfLayers`).

```mermaid
flowchart TD
    IN["residual [rows × dim]"] --> N1["Norms.rmsnormRows(attnNorm[layer]) → normed"]
    N1 --> D{"isFullAttention[layer]?"}
    D -- yes --> ATT["attention(layer, startPos, rows)<br/>→ branch"]
    D -- no --> DEL["delta(layer, rows)<br/>→ branch"]
    ATT --> R1["Ops.addRows(residual += branch)"]
    DEL --> R1
    R1 --> N2["rmsnormRows(postAttentionNorm[layer]) → normed"]
    N2 --> F{"isMoE()?"}
    F -- no --> DF["denseFfn → branch"]
    F -- yes --> MF["moe → branch"]
    DF --> R2["residual += branch"]
    MF --> R2
    R2 --> OUT["residual (next layer input)<br/>Trace.sum('l_out-layer') if tracing"]
```

#### 4b-1. `attention` — full-attention layers (gated, GQA, RoPE, KV cache)

```mermaid
flowchart TD
    x["x = folded(normed)"] --> q["gemm attnQ → packedQ [rows × 2·qDim]<br/>(query and gate interleaved)"]
    x --> k["gemm attnK → k [rows × kvDim]"]
    x --> v["gemm attnV → v [rows × kvDim]"]
    q --> u["unpackAttentionQGate<br/>→ q [rows × qDim] + attentionGate"]
    u --> qn["per row, per head (Parallel.forLoop):<br/>rmsnorm(attnQNorm, headSize)<br/>RoPE.applyNeox(q)"]
    k --> kn["per row, per KV head:<br/>rmsnorm(attnKNorm)<br/>RoPE.applyNeox(k)"]
    kn --> w["f32 → F16 rows into<br/>keyCache[layer] / valueCache[layer]<br/>at offset startPos·kvDim"]
    v --> w
    qn --> fa{"causalAttention?"}
    w --> fa
    fa -- "yes (chat)" --> c["FlashAttention.causalPrefill<br/>attends cache[0 .. startPos+row]<br/>GQA ratio = heads / kvHeads"]
    fa -- "no (d3)" --> b["FlashAttention.bidirectionalPrefill<br/>every row sees the whole prompt"]
    c --> g["GatedDeltaNet.sigmoidMultiply:<br/>attentionOut ×= sigmoid(attentionGate)"]
    b --> g
    g --> o["gemm attnOutput(folded(attentionOut)) → branch"]
```

#### 4b-2. `delta` — Gated DeltaNet layers (linear attention with recurrent state)

```mermaid
flowchart TD
    x["x = folded(normed)"] --> p1["gemm attnQkv → ssmQkv [rows × channels]<br/>channels = inner + 2·groups·stateSize"]
    x --> p2["gemm attnGate → z (output gate)"]
    n["normed (unrotated)"] --> p3["gemm ssmAlpha → alpha"]
    n --> p4["gemm ssmBeta → betaProjection"]
    p1 --> cv["Convolutions.causalDepthwiseSilu<br/>kernel K, history in convState[layer] (K-1 rows)<br/>→ convOut"]
    cv --> pq["GatedDeltaNet.prepareQkv<br/>split q/k/v, L2-normalise q,k per group,<br/>expand groups → heads: ssmQ, ssmK, ssmV"]
    p3 --> gt["GatedDeltaNet.gates<br/>(alpha, beta, ssmDtBias, ssmA) → ssmGate (decay), ssmBeta"]
    p4 --> gt
    pq --> sc["GatedDeltaNet.scan  (sequential over rows)<br/>S[h] ← decay·S[h] + β·(v − S·k) ⊗ k<br/>out = S·q   — state in recurrentState[layer]<br/>[heads × hd × hd] F32"]
    gt --> sc
    sc --> pn["GatedDeltaNet.postNorm<br/>rmsnorm(ssmNorm) gated by silu(z) → ssmTmp"]
    p2 --> pn
    pn --> out["gemm ssmOut(ssmTmp) → branch<br/>(Hadamard: forwardDeltaOut rotates input)"]
```

Unlike attention, this layer reads **no KV cache**: its entire history is `convState` + `recurrentState`,
fixed-size whatever the context length. That is also why a rejected speculation cannot just truncate
position — the state must be restored from a checkpoint (section 5).

#### 4b-3. FFN — dense or MoE

```mermaid
flowchart LR
    subgraph dense["denseFfn (qwen35)"]
        d0["x = folded(normed)"] --> d1["gemm ffnGate → hidden"]
        d0 --> d2["gemm ffnUp → hidden2"]
        d1 --> d3["siluMultiplyRows(hidden, hidden2)"]
        d2 --> d3
        d3 --> d4["gemm ffnDown(folded(hidden)) → branch"]
    end
    subgraph moe["moe (qwen35moe)"]
        m0["gemm moeRouter(normed) → logits per expert"] --> m1["Moe.softmaxSelectTopK<br/>→ moeRowTopE / moeRowTopP / expertCounts"]
        m1 --> m2["Moe.dispatch: gather rows per expert<br/>for each active expert:<br/>gemm gate, gemm up, silu×, gemm down<br/>scatter-add weighted by top-k prob → branch"]
        m0 --> s0
        s0["shared expert (if present):<br/>gate, up, silu×, down → sharedOut"] --> s1["× sigmoid(gemm moeSharedInputGate(normed))<br/>saxpy into branch"]
        m2 --> s1
    end
```

**Notes on 4b**

- **Layer pattern.** `Configuration.isFullAttention[]` decides per layer; the same array also decides what `State`
  allocates for it (KV cache *or* conv+recurrent state) and what the checkpoint codec serialises.
- **`branch` scratch.** Mixer and FFN each write into `branch`, then it is added to `residual`: the only two
  residual updates in a block.
- **`folded(...)`.** Returns `x` itself, or `hadamard.forward(x, state.rotated, …)` for Prism checkpoints. It is applied
  to the input of every quantised projection except `ssmAlpha`/`ssmBeta` and the MoE router/experts, which read `normed` directly.
- **Parallelism.** Matmuls parallelise inside `MatMul`; the per-row/per-head norm+RoPE loops use `Parallel.forLoop`.
  Scalars read from `State` views are read on the owning thread (confined-arena rule).

## 5. Speculative decoding (MTP)

The checkpoint embeds a **single-layer MTP head** (`nextnPredictLayers == 1`, layer index `numberOfLayers`). It drafts tokens; the full model verifies them in one batched forward. The Gated DeltaNet state is recurrent, so rejected drafts cannot be truncated like a KV cache: **snapshot → verify → restore & replay** via `Qwen35CheckpointCodec`.

```mermaid
sequenceDiagram
    autonumber
    participant W as Qwen35Speculative.walk
    participant C as Qwen35CheckpointCodec
    participant Q as Qwen35
    participant S as Sampler

    W->>Q: logits(state, last)  → sample next
    loop until maxTokens / stop / context / timeout
        W->>C: capture(state, base, base, scratch.checkpoint)
        W->>Q: draft(state, k, candidates)<br/>MTP greedy chain from pendingHidden (draftOne ×k)
        W->>Q: ingest(Batch.score(candidates[0..k]))  (1 verify forward)
        W->>Q: logitsAll(state, verificationLogits)
        loop accepted < k
            W->>S: sample(row[accepted])
            Note over W: accept while candidate == sampled target
        end
        alt some rejected / stop token inside
            W->>C: restore(state, base, ...)
            W->>Q: ingest(accepted prefix)  (replay)
        end
        W->>W: emit tokens → listener.onToken, audit.onEmit
    end
    W-->>W: SpeculationResult(drafted, accepted, forwards) + JFR SpeculationEvent
```

`Scratch` (cached in `State.specScratch`, sized by depth ≤ 8) owns the `(depth+1) × vocab` verification logits and the checkpoint segment.

---

## 6. Decision path (vLLM d3)

Not a chat model: typed questions in → probability per option out. Bidirectional prompt read in **one** batch.

```mermaid
sequenceDiagram
    autonumber
    participant Cl as caller (jinfer-decision)
    participant D as Qwen35Decision
    participant V as Qwen35Vision (via model.projector)
    participant Q as Qwen35
    Cl->>D: decideIndependently(state, input[], questions[])
    D->>D: split Content → text + Media[]
    loop each picture (once per call)
        D->>V: positions(image) / project(image, count, sink)
        V-->>D: Encoded(rows, count, decoderPositions)
    end
    loop each question  (one prompt each)
        D->>D: HEAD + "State/Question/Options A:.." + TAIL<br/>tokens, picture slots → Placed(at, rows, count, positions)
        D->>Q: ingestPrompt(state, tokens, placed)<br/>reset · embed · splice media rows · decoderBlock×N (bidirectional)
        D->>Q: logits(state, lastRow, codeIds[0..n), out)<br/>only n option-code rows of the head
        D->>D: softmax(logits, temperature by kind: predicate/choice/rating)
    end
    D-->>Cl: float[][] probabilities
```

`Qwen35.promptPositions` lays out 3-axis M-RoPE positions: text rows advance all three axes; picture rows use the tower's (t,y,x) from the cursor; the cursor then jumps by `positions.advance()`.

---

## 7. State memory and checkpoint layout

```mermaid
flowchart LR
    subgraph State["Qwen35.State  (extends ContextState; one arena)"]
        direction TB
        s1["activations (batchCapacity rows, F32)<br/>residual · normed · branch · hidden/hidden2<br/>q/k/v/packedQ · ssm* · rotated"]
        s2["logits (1 × vocab)"]
        s3["per layer, full-attention:<br/>keyCache / valueCache  [ctx × kvDim] F16"]
        s4["per layer, DeltaNet:<br/>convState [(K-1) × channels] F32<br/>recurrentState [heads × hd × hd] F32"]
        s5["MoE scratch + routing arrays<br/>(only if expertCount &gt; 0)"]
        s6["MTP: targetHidden · mtpConcat · pendingHidden<br/>(only if hasMtp)"]
        s7["ropeDelta · mediaRopeAngles · specScratch"]
    end
    subgraph Blob["Qwen35CheckpointCodec blob  (sizeOf(n))"]
        direction TB
        b1["n × (K,V F16 rows of every full-attention layer incl. MTP layer)<br/>= O(n) per position"]
        b2["fixed residue: Σ linear layers (S matrix + conv history) F32<br/>≈ 66 MB at 35B-A3B"]
        b3["pendingHidden (if MTP) + ropeDelta (int)"]
    end
    s3 --> b1
    s4 --> b2
    s6 --> b3
    s7 --> b3
```

Everything else in `State` is per-batch scratch and is **not** checkpointed (MoE routing is per-token, stateless across tokens).

---

## File index

| File | Lines | Role |
|---|---:|---|
| `Qwen35.java` | 1610 | Model: config/weights/state records, loading, forward pass, attention / Gated DeltaNet / dense+MoE FFN, MTP draft, logits, `withMedia` |
| `Qwen35Vision.java` | 840 | Qwen3-VL ViT tower + 2×2 merger (`MediaProjector<Image>`); fused-QKV, dual patch conv, vision M-RoPE |
| `Qwen35ChatTemplate.java` | 437 | ChatML framing, tool-call XML (render + parse), thinking seeds, constrained/forced replies |
| `Qwen35Decision.java` | 327 | vLLM d3 decision model over a bidirectional Qwen3.5 |
| `Qwen35Speculative.java` | 278 | MTP draft-and-verify loop with checkpoint rollback |
| `Hadamard.java` | 247 | Prism folded Walsh-Hadamard rotation (`prism.hadamard.*`) |
| `Qwen35CheckpointCodec.java` | 119 | Prompt-cache / rollback (de)serialisation of `State` |
| `Qwen35Provider.java` | 91 | `ModelProvider` SPI entry; routes language vs decision, attaches mmproj |
| `Qwen35VisionPreprocess.java` | 74 | `smartResize` (token-bounded, aligned, bicubic target size) |

## Notable design points

- **Hybrid stack.** Layers alternate Gated DeltaNet (linear, recurrent state) and full attention; both write into the same `State`, selected per layer by `isFullAttention[]`.
- **Same weights, three faces.** One `Qwen35` backs chat (`LoadedModel`), decision (`Qwen35Decision`) and speculation. `causalAttention=false` is what makes a checkpoint a d3 decision model.
- **Immutable model, `withMedia` copies.** The vision sidecar borrows the caller's arena; the backbone weights are shared.
- **M-RoPE for media.** Image rows enter as embedding batches carrying 3-D `Batch.Positions`; `ropeDelta` accumulates the gap between cache position and RoPE position so later text stays consistent.
- **Concurrency.** All mutating entry points go through `state.exclusively(...)`; `Reference.reachabilityFence(this)` pins the model while kernels read weights through raw segments.
- **Package-private seams.** `ingestPrompt`, `logits(...,tokens,out)`, `draft`, `logitsAll`, `Placed` exist for `Qwen35Decision` / `Qwen35Speculative`, which are therefore tightly coupled to `Qwen35` by design.
