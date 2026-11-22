# jam-vector — Architecture

`jam-vector` (JPMS module `com.qxotic.jam.vector`, provider id **`vector`**, priority **500**) is
the **Java Vector API** backend (`jdk.incubator.vector`). It handles **prefill only** (`n > 1`)
with F32 activations/result and contiguous weights (`ldw == k`); anything else returns
`EUNSUPPORTED` so the caller falls back (typically to `scalar`). It also runs inside a GraalVM
native image.

| Class | Role |
|---|---|
| `VectorJAMProvider` | `JAM.Provider`; available when `jdk.incubator.vector` is in the boot layer and native access is allowed |
| `VectorJAM` | `JAM`: validation, bounds checks, per-dtype dispatch switch, `mmLock`, reachability fences |
| `VectorSupport` | Shared constants and helpers: `F_SPECIES`, `GLOBAL` absolute-address segment, fp16/byte readers, decode stores, tile/JIT selection, `jamProp` knobs, `parallelChunks` |
| `BandGemm` | Generic dequant-to-scratch F32 GEMM: blocking plan, activation packing, panel dequant, 3x3 / 4x4 register sweeps |
| `BandGemm.RowDequant` | Functional interface each dtype kernel provides: decode a run of one weight row to F32 |
| `Scratch` | Per-instance native buffers: one packed-activation buffer, one dequant panel per slot |
| `Q8Kernel` | Q8_0: band route by default; legacy register tiles (`-Djam.vector.q8=tile`) |
| `Q4Kernel`, `Q5Kernel` | Q4_0, Q5_0 dequant rows |
| `Q4KKernel`, `Q5KKernel`, `Q6KKernel` | K-quant dequant rows (6-bit scale/min unpack) |
| `Mxfp4Kernel`, `Nvfp4Kernel` | FP4 formats (e8m0 / ue4m3 scales, LUT decode) |
| `Q1Kernel`, `Q2Kernel`, `Ptq1Kernel` | Low-bit formats: Q1_0, Q2_0/PQ2_0, PTQ1_0 (ternary) |
| `VectorWidth` | GraalVM native-image `Feature`: pins `jam.vector.width` to the target ISA at build time |

## 1. Entry point and dispatch

```mermaid
classDiagram
    direction LR
    class JAM {
        <<interface>>
    }
    class Provider {
        <<interface JAM.Provider>>
    }
    class VectorJAMProvider {
        +id() "vector"
        +priority() 500
        +isAvailable() boolean
        +create(Parallel) JAM
    }
    class VectorJAM {
        -Scratch scratch
        -ReentrantLock mmLock
        +VectorJAM(Parallel)
        +isAvailable()$ boolean
        +mm(...) int
    }
    class Scratch {
        -Parallel parallel
        -MemorySegment[] perSlot
        -MemorySegment packed
        ~parallel() Parallel
        ~width() int
        ~local(slot, need) MemorySegment
        ~packed(need) MemorySegment
    }
    class VectorSupport {
        <<utility>>
    }
    class MemoryChecks {
        <<jam-core internal>>
    }
    Provider <|.. VectorJAMProvider
    JAM <|.. VectorJAM
    VectorJAMProvider ..> VectorJAM
    VectorJAM *-- Scratch
    VectorJAM ..> MemoryChecks
    VectorJAM ..> VectorSupport : GLOBAL, F_SPECIES
    VectorJAM ..> Q8Kernel : Q8_0
    VectorJAM ..> Q4Kernel : Q4_0
    VectorJAM ..> Q5Kernel : Q5_0
    VectorJAM ..> Q4KKernel : Q4_K
    VectorJAM ..> Q5KKernel : Q5_K
    VectorJAM ..> Q6KKernel : Q6_K
    VectorJAM ..> Mxfp4Kernel : MXFP4
    VectorJAM ..> Nvfp4Kernel : NVFP4
    VectorJAM ..> Q1Kernel : Q1_0
    VectorJAM ..> Q2Kernel : Q2_0 / PQ2_0
    VectorJAM ..> Ptq1Kernel : PTQ1_0
```

```mermaid
flowchart TD
    A[VectorJAM.mm] --> B[requireNative W, A, R]
    B --> C{"m,n,k > 0 and strides >= k/m"}
    C -- no --> EI[EINVAL]
    C -- yes --> D{"n == 1 or at/rt != F32 or SIMD < 128 bit"}
    D -- yes --> EU[EUNSUPPORTED]
    D -- no --> E{"known dtype?"}
    E -- no --> EU
    E -- yes --> F{"k, ldw multiple of block"}
    F -- no --> EI
    F -- yes --> G{"ldw == k"}
    G -- no --> EU
    G -- yes --> H[checkSegment W, A, R]
    H --> I[mmLock, switch on wt -> XxxKernel.gemm]
    I --> J[OK + reachabilityFence w, a, r, this]
```

Activations and results are addressed by **absolute address** through `VectorSupport.GLOBAL`
(`MemorySegment.NULL.reinterpret(Long.MAX_VALUE)`): one constant segment type, so bounds and
liveness checks fold, and Graal native-image vector loads compile to correct absolute addressing.
That is why the bounds are checked once up front and the operands are fenced after the call.

## 2. The band GEMM and the per-dtype kernels (strategy pattern)

Every dtype kernel is just a `RowDequant`: "decode `count` elements of a weight row into F32".
`BandGemm` owns the blocking, packing, threading and register sweeps.

```mermaid
classDiagram
    direction TB
    class RowDequant {
        <<functional interface BandGemm.RowDequant>>
        +dequantize(w, rowElemOffset, count, dst, dstBase)
    }
    class BandGemm {
        <<final, static>>
        BAND : "3x3" | "4x4"
        MR, NR
        KC = 512
        PANEL_BYTES = 256 KiB
        PANEL_GRANULE
        gemm(w, a, aBase, o, oBase, aStride, oStride, n, m, k, wOff, scratch, deq)$
        packTile(...)$
        interleave(...)$
        sweepPanel33(...)$
        sweepPanel44(...)$
        sweepEdge33(...)$
        sweepEdge44(...)$
    }
    class Plan {
        <<BandGemm.Plan>>
        n, m, k, kc
        kBlocks, tiles
        panel, tailPanel, bulkPanels, panels
        splits, tilesPerSplit
        packItems, tasks
        start(p) int
        rows(p) int
    }
    class GemmRun {
        <<BandGemm.Gemm>>
        run()
        pack(i)
        sweep(j, slot)
        zeroRows(...)
        dequantPanel(...)
        sweepTiles(...)
    }
    class Q8Kernel {
        gemm(...)$
        dequantizeRow(...)$
        tileCode()$ int
        gemm512Tile3x2F32 ... gemm512Tile8x2F32
        gemm256Tile2x4F32 ... gemm256Tile4x3F32
        gemm128Tile1x1/2x4/4x4F32
        gemmScalar1x1F32, gemmScalarTile4x1F32
    }
    class Q4Kernel {
        gemm()$
        dequantizeRow()$
    }
    class Q5Kernel {
        gemm()$
        dequantizeRow()$
    }
    class Q4KKernel {
        gemm()$
        dequantizeRow()$
        getScaleMinK4()$
        packedScales()$
        packedMins()$
    }
    class Q5KKernel {
        gemm()$
        dequantizeRow()$
    }
    class Q6KKernel {
        gemm()$
        dequantizeRow()$
    }
    class Mxfp4Kernel {
        gemm()$
        dequantizeRow()$
    }
    class Nvfp4Kernel {
        gemm()$
        dequantizeRow()$
        dequantizeRow512()$
    }
    class Q1Kernel {
        gemm()$
        dequantizeRow()$
    }
    class Q2Kernel {
        gemm(qk)$
        dequantizeQ2_0()$
        dequantizePQ2_0()$
    }
    class Ptq1Kernel {
        gemm()$
        dequantizeRow()$
        trits()$
    }
    BandGemm *-- RowDequant : nested
    BandGemm *-- Plan : nested
    BandGemm *-- GemmRun : nested
    GemmRun --> Plan
    GemmRun --> RowDequant : deq
    GemmRun --> Scratch : local(slot), packed()
    Q8Kernel ..> BandGemm : default band route
    Q4Kernel ..> BandGemm
    Q5Kernel ..> BandGemm
    Q4KKernel ..> BandGemm
    Q5KKernel ..> BandGemm
    Q5KKernel ..> Q4KKernel : getScaleMinK4
    Q6KKernel ..> BandGemm
    Mxfp4Kernel ..> BandGemm
    Nvfp4Kernel ..> BandGemm
    Q1Kernel ..> BandGemm
    Q2Kernel ..> BandGemm
    Ptq1Kernel ..> BandGemm
```

(Each `XxxKernel::dequantizeRow` method reference is passed as the `RowDequant`.)

### How one `BandGemm.gemm` runs

```mermaid
sequenceDiagram
    participant K as XxxKernel.gemm
    participant BG as BandGemm.gemm
    participant P as Plan
    participant S as Scratch
    participant Pool as JAM.Parallel
    K->>BG: gemm(..., scratch, Xxx::dequantizeRow)
    BG->>P: new Plan(n, m, k, kc, width)
    BG->>S: packed(tiles * NR * k)
    BG->>Pool: run(packItems) — pack activation tiles [k/F_LEN][NR][F_LEN]
    BG->>Pool: run(panels * splits) — sweep(j, slot)
    Note over Pool: per task: S.local(slot) panel buffer,<br/>zero output rows, then per k-block:<br/>dequantPanel (RowDequant into [k/F_LEN][MR][F_LEN])<br/>sweepTiles (MR x NR FMA accumulators)
```

Key blocking choices (see `BandGemm` Javadoc):

* **Panel**: up to `PANEL_BYTES` (256 KiB) of dequantized bands per task, kept in that core's L2;
  the slot's buffer is reused by the next task on the same slot.
* **k-blocks** of `KC` (512, multiple of 256) so an NR-column tile + a band stay in L1.
* **Scheduling**: pack items first, then full panels, then quarter-panel tail tasks; small `m`
  also splits the tokens. `PANEL_GRANULE` keeps panels on separate 64-byte output lines.
* **Tile**: 4x4 under C2 / GraalVM ≥ 25.4 JIT at 512 bit, otherwise 3x3 (Graal before 25.4
  only allocates zmm0-15); override `-Djam.vector.band=3x3|4x4`.

## 3. `VectorSupport`: species, addressing and tuning

```mermaid
classDiagram
    direction LR
    class VectorSupport {
        <<final utility>>
        +VectorSpecies~Float~ F_SPECIES$
        +boolean IS_512$
        +int F_LEN$
        +int DECODE_PARTS$
        +boolean IN_NATIVE_IMAGE$
        +boolean WIDE_TILES_COMPILABLE$
        +boolean GRAAL_JIT$
        +boolean WIDE_TILE$
        +int SEQ_TILE$
        +int ROW_TILE$
        +String TILE$
        +int TILE_CODE$
        +MemorySegment GLOBAL$
        jamProp(name, def)$ String
        jamPropInt(name, def)$ int
        vectorWidth(requested, preferred)$ int
        tileCode(tile)$ int
        wideBand(graalJit, vendorVersion)$ boolean
        storeAffine(q, scale, neg, dst, off)$
        storeScaled(q, scale, dst, off)$
        vectorSegment(seg)$ MemorySegment
        vectorBase(seg)$ long
        readByte / readShort / readInt / readLong / readFloat16$
        putFloat(address, value)$
        parallelChunks(parallel, count, ChunkConsumer)$
    }
    class ChunkConsumer {
        <<functional VectorSupport.ChunkConsumer>>
        accept(lo, hi)
    }
    class VectorWidth {
        <<GraalVM Feature>>
        +beforeAnalysis(access)
        -targetFeatures()$ Set~String~
    }
    class Feature {
        <<interface org.graalvm.nativeimage.hosted>>
    }
    VectorSupport *-- ChunkConsumer : nested
    Feature <|.. VectorWidth
    VectorWidth ..> VectorSupport : sets jam.vector.width BEFORE its class init
```

* `F_SPECIES` is the float species, the JVM's preferred width or `jam.vector.width` (128/256/512).
  The decoders fan one 16-byte `ByteVector.SPECIES_128` chunk into `DECODE_PARTS` F32 stores, so
  every kernel runs at 128, 256 or 512 bit.
* `TILE_CODE` (used only by the legacy Q8_0 tile route) is picked from arch, width and JIT:
  NEON 4x4 on ARM, 4x4 on C2 at 512-bit, 3x2 on Graal / native image, AVX2 2x4 at 256-bit,
  scalar otherwise.
* In a native image the package is initialized at build time
  (`META-INF/native-image/.../native-image.properties`); `VectorWidth` sets `jam.vector.width`
  from the target CPU features (AVX512F → 512, AVX2 → 256, else 128) before the species freeze.

## 4. Configuration knobs (`-D` or `JAM_VECTOR_*` env)

| Property | Default | Effect |
|---|---|---|
| `jam.vector.width` | preferred | 128 / 256 / 512 |
| `jam.vector.band` | auto | `3x3` or `4x4` band sweep |
| `jam.vector.kc` | 512 | k-block length (rounded to 256) |
| `jam.vector.panelKb` | 256 | dequant panel size per task |
| `jam.vector.q8` | `band` | `tile` = legacy Q8_0 register tiles |
| `jam.vector.tile` | `auto` | legacy Q8 tile shape (3x2, 4x4, avx256-…, neon, scalar…) |
| `jam.vector.seqTile` / `rowTile` | 32 / 128 | legacy Q8 tile blocking |
| `jam.vector.wideTiles` | false | native-image build opt-in for 512-bit wide tiles |
| `jam.vector.disabled` | false | drop the provider |
