# jinfer-kernels — class architecture

Package `com.qxotic.jinfer.kernels` (one package, 25 top-level types + `package-info`, ~10 800 lines). It is the
**compute layer** shared by every `jinfer-<model>` provider: static, dtype-checked kernels over
`MemoryView<MemorySegment>` (matmul dispatch, norms, RoPE, flash attention, MoE routing, recurrent scans,
convolutions, conversions) plus the GGUF tensor loader that produces those views. The `package-info` states it is
*internal to jinfer*: published so the dependency graph resolves, with no API or compatibility promise.

Maven dependencies (`pom.xml`): `jota-core`, `jota-memory`, `jinfer-core`, `gguf`, `jam-core` (API + ServiceLoader
discovery only — `jam-native` / `jam-vector` / `jam-scalar` come from the *consumer's* classpath; `jam-scalar` and
`jam-vector` are test-scoped here). `graalvm-annotations` is `provided` (build-time `@AlwaysInline` / `@NeverInline`
stubs). JPMS automatic name `com.qxotic.jinfer.kernels`.

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends;
classes marked `<<external>>` live outside this package.

1. [Position in the stack](#1-position-in-the-jinfer-stack)
2. [Class diagrams](#2-class-diagrams) — 2a loading & dtypes, 2b matmul & JAM, 2c kernel families, 2d shared infrastructure
3. [Loading sequence](#3-loading-sequence-modelloaderloadtensors)
4. [Matmul dispatch](#4-matmul-dispatch-matmulmm)
5. [Attention](#5-attention-flashattention)
6. [Recurrent scans: vector vs scalar selection](#6-recurrent-scans)
7. [MoE dispatch](#7-moe-dispatch)
8. [Run-time selection switches, SPI and native-image config](#8-run-time-selection-switches-spi-and-native-image)

| Class | Role |
|---|---|
| `ModelLoader` | GGUF entry: `readGguf`, `loadTensors` (mmap read-only → one jota `Memory`, one `MemoryView` per tensor, then `JamPack.apply` + `MatMul.shareWeight`), `requireComplete`, `requireVectorApi`, lookup helpers (`require`, `requireF32`, `find`, `findF32`, `findFirst`, `ropeFreqFactors`) |
| `GGMLDataTypes` | package-private `GGMLType` ↔ jota `DataType` table (19 pairs), geometry-checked at class init |
| `JamPack` | package-private load-time repacker: Q4_0/MXFP4/Q4_K/Q5_K/Q6_K weights → jam's packed layout in one page-aligned private mmap slab; `PackPolicy` seam |
| `JamPacked` | package-private `DataType` impl: a packed weight (one block = one row), keeps the `canonical()` view it replaced |
| `MatMul` | `c = w · aᵀ` (ggml convention): `gemm`/`gemv` façades, `mm` router over JAM rungs (native → vector → scalar) and the Java floor `mmFloor`; all per-dtype dot kernels; nested `JamMm` |
| `Raw` | record `(vseg, vbase)`: checked extraction of a raw segment + absolute base from a view (kernel-only access) |
| `FlashAttention` | causal / sliding-window / bidirectional prefill, `flashDecode` (partitioned) and `rollingDecode`; nested `Src`, `Buffers`, public `DecodeScratch` |
| `Norms` | RMSNorm (plain, ggml-ordering, rows, no-weight), `sumOfSquares`, `scaleByWeight`, LayerNorm variants |
| `RoPE` | `Schedule` functional interface + `plain` / `withFreqFactors` / `yarn` factories; `fill`, `applyInterleaved`, `applyNeox`, `rotatePairs` |
| `Activations` | scalar activations (sigmoid, silu, gelu, softplus, clamped SwiGLU, tanh-sigmoid) and fused span forms (`geluMultiply`, `siluMultiply`, `glu`, `softcap`, …) |
| `Ops` | elementwise FP32 span kernels (`addInPlace`, `saxpyInPlace`, `softmaxInPlace`, `argmax`, `dot`, transposes, pools, `MapFunction`) |
| `FastMath` | polynomial `expNeg`, `sigmoid`, `tanh` + exported vector constants (`EXP_*`) |
| `Convert` | dtype converters: `f16ToF32Vector`, `copyF32`, f16/bf16↔f32, `dequantQ8_0`, `dequantMxfp4`, `dequantLegacy`, `copyToF32`, `gatherToF32` (JamPacked-aware) |
| `Moe` | MoE routing: `Routing` buffers, top-k / grouped top-k / softmax-top-k selection, `normalizeTopP`, CSR `dispatch` with an `ExpertKernel` callback |
| `GatedDeltaNet` | Qwen3.5 gated-delta preparation (`prepareQkv`, `gates`, `postNorm`, …) and `scan` dispatcher |
| `VectorGatedDeltaNet` | package-private Vector API recurrence for `GatedDeltaNet.scan` |
| `KimiDeltaAttention` | Kimi Delta Attention gates, Q/K normalisation, recurrent `scan`, `postNorm`, `headSigmoidMultiply` |
| `Mamba2` | Nemotron-H Mamba2 selective `scan` and `groupedRmsNorm` dispatchers + scalar oracles |
| `VectorMamba2` | package-private Vector API forms of both |
| `Convolutions` | `conv2dStride2Pad1`, causal depthwise 1-D (+SiLU), `conv1dRows` (register-tiled, `jinfer.convTile`), `shortConvScan`, `segmentedShortConv` |
| `LogMel` | log-mel spectrogram front end for audio models; nested `Spec` record |
| `Resample` | Pillow-compatible bicubic RGB resize (`bicubic`) for vision towers; private `Filter` record |
| `KvTransfer` | direction-flagged bulk copy between state views and a checkpoint blob (`transfer`, `ringSpan`) |
| `Timer` | `AutoCloseable` interface; `Timer.log(label)` logs elapsed load phases on `jinfer.load` |
| `Trace` | `-Djinfer.trace` per-layer residual checksum printer |

---

## 1. Position in the jinfer stack

```mermaid
flowchart TB
    subgraph models["jinfer-&lt;model&gt; providers"]
        M["llama · qwen3 · qwen35 · gemma4 · gptoss · lfm2 · lfm3 · nemotronh<br/>laguna · bailingmoe3 · mellum · maple · inflect2 · parakeet · kokoro · ..."]
    end
    Other["jinfer-chat · jinfer-bench · jinfer-langchain4j"]
    subgraph kern["jinfer-kernels"]
        ML["ModelLoader / JamPack"]
        MM["MatMul"]
        K["FlashAttention · Norms · RoPE · Ops · Activations<br/>Convert · Moe · GatedDeltaNet · Mamba2 · KimiDeltaAttention<br/>Convolutions · LogMel · Resample · KvTransfer"]
    end
    subgraph core["jinfer-core"]
        C["Segments · Parallel · Views · Arenas<br/>RuntimeFlags · PerformanceCliff · Media"]
    end
    subgraph jota["jota-core / jota-memory"]
        J["DataType · Shape · Layout<br/>Memory · MemoryView · Memories"]
    end
    G["gguf: GGUF · TensorEntry · GGMLType"]
    subgraph jam["jam"]
        JC["jam-core: JAM / JAM.Provider"]
        JB["jam-native · jam-vector · jam-scalar<br/>consumer classpath only"]
    end

    M --> ML
    M --> MM
    M --> K
    Other --> kern
    ML --> G
    ML --> J
    kern --> C
    kern --> J
    MM --> JC
    JB -.->|"ServiceLoader JAM$Provider"| JC
```

The poms of `jinfer-bench`, `jinfer-chat`, `jinfer-langchain4j` and every model module (list above) depend on
`jinfer-kernels`. Nothing in this module depends on a model.

---

## 2. Class diagrams

### 2a. Loading and dtypes

```mermaid
classDiagram
    direction LR
    class GGUF { <<external>> }
    class TensorEntry { <<external>> }
    class GGMLType { <<enum, external>> }
    class DataType { <<interface, external>> }
    class MemoryView~MemorySegment~ { <<interface, external>> }
    class Memories { <<external>> }
    class Arenas { <<external>> }
    class ModelLoader {
        <<final, static>>
        +readGguf(FileChannel, label) GGUF
        +requireVectorApi()
        +loadTensors(FileChannel, GGUF, Arena) Map
        +loadTensors(FileChannel, offset, tensors, Arena) Map
        +requireComplete(GGUF, byteSize, source)
        +ropeFreqFactors(views) Optional
        +require(views, name) MemoryView
        +requireF32(views, name) MemoryView
        +find(views, name) Optional
        +findF32(views, name) Optional
        +findFirst(views, names) Optional
    }
    class GGMLDataTypes {
        <<package-private>>
        -SUPPORTED Pair[]
        +toDataType(GGMLType) DataType
        +toGGMLType(DataType) GGMLType
    }
    class JamPack {
        <<package-private>>
        -ENABLED jinfer.pack
        apply(views, Arena) Map
        apply(views, Arena, PackPolicy) Map
        mappedSlab(Path, bytes, Arena) MemorySegment
        -packGroup(...)
    }
    class PackPolicy {
        <<interface>>
        size(DataType, rows, k) long
    }
    class JamPacked {
        <<package-private>>
        -base DataType
        -k long
        -rowBytes long
        -canonical MemoryView
        of(base, k, rowBytes, canonical)$ JamPacked
        base() DataType
        canonical() MemoryView
    }
    class Timer {
        <<interface>>
        +log(label)$ Timer
        +close()
    }

    ModelLoader ..> GGUF : GGUF.read
    ModelLoader ..> TensorEntry : iterates
    ModelLoader ..> GGMLDataTypes : toDataType
    ModelLoader ..> Memories : Memories.of(mmap)
    ModelLoader ..> MemoryView : MemoryView.of per tensor
    ModelLoader ..> Arenas : requireCrossThread
    ModelLoader ..> JamPack : apply
    ModelLoader ..> MatMul : shareWeight
    ModelLoader ..> Timer : log
    GGMLDataTypes ..> GGMLType
    GGMLDataTypes ..> DataType
    DataType <|.. JamPacked
    JamPack *-- PackPolicy : nested
    JamPack ..> JamPacked : of
    JamPack ..> MatMul : nativePackSize = PackPolicy
    JamPack ..> Timer
    JamPacked --> MemoryView : canonical
```

`JamPacked` is never a wire format — `GGMLDataTypes` does not know it, so it cannot be serialized. `MatMul`
reads the packed slab; `Convert.dequantLegacy` / `copyToF32` / `gatherToF32` detect `instanceof JamPacked` and
read `canonical()` instead.

### 2b. Matmul and the JAM backends

```mermaid
classDiagram
    direction LR
    class JAM { <<interface, external>> }
    class Provider { <<interface, external>>
        JAM.Provider
        id() String
        create(JAM.Parallel) JAM
    }
    class JAMParallel { <<interface, external>>
        JAM.Parallel
    }
    class Parallel { <<external>> }
    class PerformanceCliff { <<enum, external>> }
    class RuntimeFlags { <<external>> }
    class MatMul {
        <<final, static>>
        -HOST JAM.Parallel
        -NATIVE JamMm
        -VECTOR JamMm
        -SCALAR JamMm
        -SHARING boolean
        +gemm(w, a, aStride, c, cStride, m, n, k)
        +gemm(w, wOff, a, aStride, c, cStride, m, n, k)
        +gemm(w, a, c, n)
        +gemv(w, a, c, m, k)
        +gemv(w, a, c)
        +mm(w, wOff, wStride, a, aOff, aStride, c, cOff, cStride, m, n, k)
        mmFloor(...)
        jamApplies(DataType, k, wOff)$ boolean
        jamTag(DataType)$ int
        nativePackSize(DataType, rows, k)$ long
        shareWeight(MemoryView)$
        jamRungs()$ List
        getLegacy(...)$ float
    }
    class JamMm {
        <<private static>>
        -jam JAM
        mm(ws, wBase, wOff, dt, ...) boolean
    }
    class Raw {
        <<record>>
        +vseg MemorySegment
        +vbase long
        +f32(view, name)$ Raw
        +of(view, DataType, name)$ Raw
    }

    MatMul *-- JamMm : NATIVE / VECTOR / SCALAR
    JamMm --> JAM
    MatMul ..> Provider : JAM.providers + create HOST
    MatMul *-- JAMParallel : HOST adapter
    JAMParallel ..> Parallel : shared run / forLoop / threads
    MatMul ..> Raw : f32 / of
    MatMul ..> JamPacked : instanceof route
    MatMul ..> Convert
    MatMul ..> PerformanceCliff : JAM_ABSENT / JAM_DECLINE
    MatMul ..> RuntimeFlags : THREADS
```

### 2c. Kernel families and their internal dependencies

```mermaid
flowchart LR
    subgraph attn["Attention"]
        FA["FlashAttention"]
        FAS["Src · Buffers · DecodeScratch"]
    end
    subgraph rec["Recurrences"]
        GDN["GatedDeltaNet"] --> VGDN["VectorGatedDeltaNet"]
        M2["Mamba2"] --> VM2["VectorMamba2"]
        KDA["KimiDeltaAttention"]
    end
    subgraph elem["Elementwise / norms"]
        OPS["Ops"]
        ACT["Activations"]
        NORM["Norms"]
        FM["FastMath"]
    end
    subgraph misc["Other"]
        MOE["Moe"]
        CONV["Convolutions"]
        CVT["Convert"]
        ROPE["RoPE"]
        LM["LogMel"]
        RS["Resample"]
        KV["KvTransfer"]
    end
    RAW["Raw"]

    FA --- FAS
    FA --> FM
    FA --> CVT
    ACT --> FM
    ACT --> OPS
    OPS --> FM
    NORM --> OPS
    GDN --> ACT
    GDN --> OPS
    GDN --> NORM
    M2 --> ACT
    VM2 --> ACT
    CONV --> ACT
    CONV --> OPS
    MOE --> OPS
    MOE --> CVT
    attn --> RAW
    rec --> RAW
    elem --> RAW
    MOE --> RAW
    CONV --> RAW
    ROPE --> RAW
```

Edges are verified `import`/call references between files of this package. Every kernel validates its view
operands at entry through `Raw.f32` / `Raw.of` (dense, row-major, expected dtype, native-backed) and then works on
raw `(vseg, vbase)` addresses.

### 2d. Nested types and shared infrastructure

```mermaid
classDiagram
    direction LR
    class Segments {
        <<external>>
        F_SPECIES
        USE_VECTOR_API
        FAST_VECTOR_JIT
        readFloat()
        writeFloat()
    }
    class Views { <<external>> }
    class MemoryAllocator~MemorySegment~ { <<interface, external>> }
    class Media { <<external>> }
    class FlashAttention { <<final>> }
    class Src {
        <<record>>
        raw Raw
        f16 boolean
    }
    class Buffers {
        <<static final>>
        s float[]
        m float[]
        l double[]
        kvOff int[]
        kDec vDec MemorySegment
    }
    class DecodeScratch {
        <<public static final>>
        -allocator MemoryAllocator
        +DecodeScratch(MemoryAllocator)
    }
    class Moe { <<final>> }
    class Routing {
        <<public static final>>
        rowTopE counts offsets cursor rowByExpert int[]
        rowTopP probByExpert float[]
        topK int
    }
    class ExpertKernel {
        <<interface>>
        apply(e, n, gather, out)
    }
    class RoPE { <<final>> }
    class Schedule {
        <<interface>>
        angles(position, out)
        amplitude() float
    }
    class LogMel {
        +LogMel(Spec)
        +frames(pcm, from, length, leftPad, frames) float[]
        +normalizePerFeature(...)$
    }
    class Spec {
        <<record>>
        nFft hop nMels int
        window filterbank float[]
        preemphasis magPower melFloor logZeroGuard float
        bins() int
    }
    class Ops { <<final>> }
    class MapFunction { <<interface>> }
    class Resample {
        +bicubic(...)$ int[]
    }

    FlashAttention *-- Src : nested
    FlashAttention *-- Buffers : nested
    FlashAttention *-- DecodeScratch : nested
    DecodeScratch --> MemoryAllocator
    Moe *-- Routing : nested
    Moe *-- ExpertKernel : nested
    RoPE *-- Schedule : nested
    LogMel *-- Spec : nested
    Ops *-- MapFunction : nested
    Resample ..> Media
    FlashAttention ..> Segments
    FlashAttention ..> Views
```

`FlashAttention.Buffers` are per pool *slot* (`BUFFERS = new Buffers[Parallel.threads()]`), sized lazily to the
largest head seen. `DecodeScratch` is per model state and draws from the state's `MemoryAllocator`, so it dies with
the state.

---

## 3. Loading sequence (`ModelLoader.loadTensors`)

```mermaid
sequenceDiagram
    autonumber
    participant P as Model provider
    participant ML as ModelLoader
    participant G as GGUF «external»
    participant GT as GGMLDataTypes
    participant JP as JamPack
    participant MM as MatMul
    P->>ML: readGguf(channel, label)
    ML->>G: GGUF.read(buffered channel)
    G-->>P: GGUF metadata
    P->>ML: loadTensors(channel, gguf, arena)
    ML->>ML: requireVectorApi, Arenas.requireCrossThread(arena)
    ML->>ML: requiredTensorBytes (truncation check)
    ML->>ML: channel.map(READ_ONLY, dataOffset, required, arena)
    ML->>ML: Memories.of(mapping) - one Memory
    loop each TensorEntry
        ML->>GT: toDataType(ggmlType) (throws if unsupported)
        ML->>ML: reverse GGUF dims, physicalShape, size check
        ML->>ML: MemoryView.of(memory, offset, dtype, rowMajor)
    end
    ML->>JP: apply(views, arena)
    alt jinfer.pack=true and NATIVE jam reports packSize > 0
        JP->>MM: nativePackSize(dt, rows, k)
        JP->>JP: mappedSlab (private mmap of temp file)
        JP->>JP: Parallel.forLoop packGroup per 4-row group
        JP->>JP: replace view: dtype = JamPacked(base, canonical)
    end
    JP-->>ML: views
    loop each view
        ML->>MM: shareWeight(view) (no-op unless SHARING)
    end
    ML-->>P: Map name to MemoryView
```

Notes verified in code: a confined arena is refused; the tensor data offset must be 4-byte aligned;
`token_embd.weight` and `per_layer_token_embd.weight` are never packed (row-read outside matmul); only tensors whose
second-to-last dim is a multiple of 4 and whose `packSize` divides into `rows/4` groups are packed.

---

## 4. Matmul dispatch (`MatMul.mm`)

Contract: `C[s*cStride + cOff + row] = dot(W[row], A[s])` — ggml's `ggml_mul_mat(w, a)`, with `m` = weight rows
(output width), `n` = activation rows, `k` = contraction. `a`/`c` are dense FP32.

```mermaid
flowchart TD
    E["gemm / gemv / shaped gemm(w,a,c,n)"] --> MM["mm(...)"]
    MM --> R0{"Parallel.inRegion()?"}
    R0 -->|yes| X0["IllegalStateException"]
    R0 -->|no| SH["Raw.f32 a,c; inPlace = same region<br/>SHARING: shareOnce a,c"]
    SH --> P{"dt instanceof JamPacked?"}
    P -->|yes| PN{"NATIVE.mm ok?"}
    PN -->|yes| DONE["return"]
    PN -->|no| X1["IllegalStateException: packed weight rejected"]
    P -->|no| D{"n > 1 or jamDecode,<br/>and !inPlace and jamApplies?"}
    D -->|no| F["mmFloor: Java dots by dtype<br/>Parallel region over n x m cells"]
    D -->|yes| N{"NATIVE.mm ok?<br/>skipped first for NVFP4 prefill on x86"}
    N -->|yes| DONE
    N -->|no| V{"VECTOR.mm ok?"}
    V -->|yes| DONE
    V -->|no| S{"SCALAR.mm ok?"}
    S -->|yes| DONE
    S -->|no| F
```

`jamDecode` (n == 1) is true when the Vector API is off, or for the C2 `slowDot` k-quants (Q4_K/Q5_K/Q6_K without
`FAST_VECTOR_JIT`), or when a native decode switch is on for the dtype: `jinfer.q4.nativeDecode`,
`jinfer.q8.nativeDecode` (default: AArch64 or ≤4 threads), `jinfer.kq.nativeDecode` (AArch64 or ≤8 threads),
`jinfer.mxfp4.nativeDecode` and `jinfer.lowbit.nativeDecode` (default AArch64), `jinfer.nvfp4.nativeDecode`
(default true). `jamApplies` accepts Q8_0, Q4_0, Q5_0, Q4_K, Q5_K, Q6_K, MXFP4, NVFP4, Q1_0, Q2_0, PQ2_0, PTQ1_0,
FP32, FP16, BF16 with block-aligned `k` and `wOff`. `JamMm.mm` returns `false` on any non-OK status and reports
`PerformanceCliff.JAM_DECLINE` unless the status is `EUNSUPPORTED`. If no rung loaded at all,
`PerformanceCliff.JAM_ABSENT` is reported from `MatMul`'s static initializer.

`mmFloor` covers every block dtype (adds Q4_1, Q5_1, TQ1_0, TQ2_0 which jam does not take) plus BF16/FP16/FP32 and
throws `UnsupportedOperationException` otherwise.

### Backend loading (class init)

```mermaid
sequenceDiagram
    participant MM as MatMul clinit
    participant J as JAM «external»
    participant SL as ServiceLoader
    participant PR as JAM.Provider «jam-native / jam-vector / jam-scalar»
    MM->>MM: HOST = JAM.Parallel adapter over jinfer Parallel
    loop id in native, vector, scalar
        MM->>J: providers()
        J->>SL: load(Provider.class), drop jam.id.disabled, filter available
        SL-->>J: providers sorted by priority
        MM->>PR: create(HOST) where id matches
        PR-->>MM: JAM, wrapped in JamMm (null on absence or Throwable)
    end
    MM->>MM: all null: PerformanceCliff.JAM_ABSENT.report
    MM->>MM: SHARING = NATIVE != null and probeSharing
```

---

## 5. Attention (`FlashAttention`)

```mermaid
flowchart LR
    subgraph prefill["Prefill - n query rows"]
        CP["causalPrefill"]
        SW["slidingWindowPrefill"]
        BI["bidirectionalPrefill"]
    end
    subgraph decode["Decode - one query"]
        FD["flashDecode"]
        RD["rollingDecode"]
    end
    CP --> T["Br x Bc = 64 x 64 tiles<br/>qkTile / pvTile, QT = 4 query rows"]
    SW --> T
    BI --> T
    T --> DF["decodeF16Run: F16 K/V block to F32 scratch once"]
    T --> BUF["Buffers per Parallel slot"]
    FD -->|"range below DECODE_BLOCK_SIZE: 1 partition"| RD
    FD -->|"else"| PART["nParts slices, online softmax partials<br/>in DecodeScratch, merged per head"]
    RD --> ACC["accumulate / normalize<br/>ring addressing via ringMask, optional sinks"]
    PART --> ACC
```

K/V sources are FP32 or FP16 (`Src`); Q/out are FP32 `Raw`. Softmax exponentials use `FastMath.expNeg` and its
vector constants. `nParts = max(1, min(RuntimeFlags.THREADS, range / RuntimeFlags.DECODE_BLOCK_SIZE + 1))`.

---

## 6. Recurrent scans

```mermaid
flowchart TD
    G["GatedDeltaNet.scan"] --> GA{"VectorGatedDeltaNet.applies(headDim)?<br/>jinfer.vector.gdn and FAST_VECTOR_JIT<br/>and headDim multiple of lane count"}
    GA -->|yes| GV["VectorGatedDeltaNet.scan<br/>work split by head x 16-row block"]
    GA -->|no| GC{"geometryUncovered?"}
    GC -->|yes| GP["PerformanceCliff.GDN_SCALAR.report"] --> GS["scanScalar - oracle"]
    GC -->|no| GS
    M["Mamba2.scan"] --> MA{"VectorMamba2.appliesScan(stateSize)?<br/>jinfer.vector.mamba2"}
    MA -->|yes| MV["VectorMamba2.scan"]
    MA -->|no| MC{"geometryUncovered?"}
    MC -->|yes| MP["PerformanceCliff.MAMBA2_SCALAR.report"] --> MS["scanScalar"]
    MC -->|no| MS
    N["Mamba2.groupedRmsNorm"] --> NA{"VectorMamba2.appliesGroupedRmsNorm?<br/>jinfer.vector.mamba2Norm"}
    NA -->|yes| NV["VectorMamba2.groupedRmsNorm"]
    NA -->|no| NS["groupedRmsNormScalar"]
```

`KimiDeltaAttention.scan` has its own single implementation (no separate vector class). The scalar forms are
package-private so the parity tests can compare against them.

---

## 7. MoE dispatch

```mermaid
sequenceDiagram
    participant Model as Model layer «external»
    participant Moe
    participant Ops
    participant Convert
    participant K as ExpertKernel «model lambda»
    Model->>Moe: selectTopK / selectTopKGrouped / softmaxSelectTopK into Routing
    Model->>Moe: dispatch(routing, rows, dim, input, gather, expertOut, out, expertScale, kernel)
    Moe->>Moe: buildCsr (bucket rows per expert, fold expertScale)
    Moe->>Ops: fillInPlace(out, 0)
    loop each expert e with n rows
        Moe->>Convert: copyF32 rows into gather (Parallel.forLoop)
        Moe->>K: apply(e, n, gather, expertOut)
        Moe->>Ops: saxpyInPlace(out, expertOut, weight) (Parallel.forLoop)
    end
```

The `ExpertKernel` is where a model calls `MatMul.gemm` with a weight offset (`gemm(w, wOff, …)` — "a MoE expert
slice").

---

## 8. Run-time selection switches, SPI and native image

| Switch | Read in | Effect |
|---|---|---|
| `jinfer.pack` (default true) | `JamPack` | load-time packing on/off |
| `jinfer.{q4,q8,kq,mxfp4,nvfp4,lowbit}.nativeDecode` | `MatMul` | force decode through native jam either way |
| `jam.<id>.disabled` | `JAM.providers` (jam-core) | drop a JAM rung (`native`, `vector`, `scalar`) |
| `jinfer.vector.gdn` | `VectorGatedDeltaNet` | vector GDN recurrence |
| `jinfer.vector.mamba2`, `jinfer.vector.mamba2Norm` | `VectorMamba2` | vector Mamba2 scan / norm |
| `jinfer.convTile` (`auto`, `4x1`, `4x2`, `4x4`) | `Convolutions` | `conv1dRows` register tile (auto: 4x4 on AArch64, else 4x1) |
| `jinfer.trace` | `Trace` | per-layer checksum prints |
| `jinfer.vector.jit`, `jinfer.vector.width` | `Segments` «external» | `FAST_VECTOR_JIT`, species width (exercised by the surefire matrix) |

**SPI.** This module registers no `META-INF/services` entries. It is a *consumer* of the `com.qxotic.jam.JAM$Provider`
SPI: `jam-native`, `jam-vector` and `jam-scalar` each ship that service file, and `MatMul.load(id)` picks the
provider by `id()`.

**Native image.** `META-INF/native-image/com.qxotic/jinfer-kernels/native-image.properties` sets
`--initialize-at-run-time` for `MatMul` (JAM backend resolved at `<clinit>`), `FlashAttention` (per-slot scratch sized
by the host thread budget) and `JamPack` (reads `-Djinfer.pack`). Hot Vector-API callees carry `@AlwaysInline`
(e.g. `FlashAttention.srcFloat`); some tiles are `@NeverInline` to stay their own expansion unit. `Trace.ENABLED` and
`Convolutions`' tile code are deliberately build-time constants in an image.

**Test matrix (pom).** Surefire runs the suite once per selection (`conv-tile-4x2/4x4`, `vector-jit-fast/slow`,
`java-recurrences`, `vector-128/256`, `scalar`, `jam-parity`); ordinary executions exclude `jam-vector`/`jam-scalar`
from the classpath, and `jam-parity` compares `MatMul.mm` against `MatMul.mmFloor` with the pure-Java rungs present.
