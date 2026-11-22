# jam — Architecture (summary)

jam ("JVM Accelerated Math", jokingly "Just A Matmul") computes `R = W @ Aᵀ` with quantized
weights on the CPU (and Apple GPU), behind **one interface**, `JAM`, and several interchangeable
backends discovered at run time. This page sums up the per-module documents:

| Module | JPMS module | Provider id / priority | Details |
|---|---|---|---|
| `jam-core` | `com.qxotic.jam` | — (API + SPI) | [jam-core/docs/ARCHITECTURE.md](../jam-core/docs/ARCHITECTURE.md) |
| `jam-native` | `com.qxotic.jam.libjam` | `native` / 1000 | [jam-native/docs/ARCHITECTURE.md](../jam-native/docs/ARCHITECTURE.md) |
| `jam-vector` | `com.qxotic.jam.vector` | `vector` / 500 | [jam-vector/docs/ARCHITECTURE.md](../jam-vector/docs/ARCHITECTURE.md) |
| `jam-scalar` | `com.qxotic.jam.scalar` | `scalar` / 0 | [jam-scalar/docs/ARCHITECTURE.md](../jam-scalar/docs/ARCHITECTURE.md) |
| `graalvm-annotations` | `com.qxotic.graalvm.annotations` | — (compile-only) | [graalvm-annotations/docs/ARCHITECTURE.md](../graalvm-annotations/docs/ARCHITECTURE.md) |

For the high-level picture see [ARCHITECTURE-overview.md](ARCHITECTURE-overview.md); for native
kernel internals see [design.md](design.md).

## 1. Module dependencies

```mermaid
classDiagram
    direction BT
    class jam_core {
        <<module com.qxotic.jam>>
        exports com.qxotic.jam
        exports internal to libjam, scalar, vector
        uses JAM.Provider
    }
    class graalvm_annotations {
        <<module com.qxotic.graalvm.annotations>>
        AlwaysInline, NeverInline, Uninterruptible
    }
    class jam_native {
        <<module com.qxotic.jam.libjam>>
        provides JAM.Provider
        + libjam (C / Metal)
    }
    class jam_scalar {
        <<module com.qxotic.jam.scalar>>
        provides JAM.Provider
    }
    class jam_vector {
        <<module com.qxotic.jam.vector>>
        provides JAM.Provider
        requires jdk.incubator.vector
    }
    jam_native --> jam_core : requires
    jam_scalar --> jam_core : requires
    jam_vector --> jam_core : requires
    jam_scalar ..> graalvm_annotations : requires static
    jam_vector ..> graalvm_annotations : requires static
    jam_vector ..> jam_scalar : test scope (reference)
    jam_vector ..> jam_native : test scope (parity)
```

## 2. The API and its three implementations

```mermaid
classDiagram
    direction TB
    class JAM {
        <<interface>>
        +providers()$ List~Provider~
        +mm(w, wOff, wt, ldw, a, aOff, at, lda, r, rOff, rt, ldr, m, n, k) int
        +packSize(dtype, m, k) long
        +share(segment, immutable) boolean
    }
    class Provider {
        <<interface JAM.Provider>>
        +id() String
        +priority() int
        +isAvailable() boolean
        +create(Parallel) JAM
    }
    class Parallel {
        <<interface JAM.Parallel>>
        +run(jobs, Job)
        +width() int
        +forLoop(count, Job)
    }
    class NativeJAMProvider
    class VectorJAMProvider
    class ScalarJAMProvider
    class NativeJAM {
        jam_context* via JNI / FFM
        packSize, share supported
    }
    class VectorJAM {
        Vector API, prefill only
    }
    class ScalarJAM {
        pure Java, reference, all shapes
    }
    Provider <|.. NativeJAMProvider
    Provider <|.. VectorJAMProvider
    Provider <|.. ScalarJAMProvider
    JAM <|.. NativeJAM
    JAM <|.. VectorJAM
    JAM <|.. ScalarJAM
    NativeJAMProvider ..> NativeJAM : create
    VectorJAMProvider ..> VectorJAM : create
    ScalarJAMProvider ..> ScalarJAM : create
    NativeJAM --> Parallel : fan-out via upcall
    VectorJAM --> Parallel : via Scratch
    ScalarJAM --> Parallel : via kernels
```

| | `native` | `vector` | `scalar` |
|---|---|---|---|
| Compute | libjam C kernels (SSE3 … AVX-512-VNNI, NEON/DotProd/I8MM) + Metal | Java Vector API, 128/256/512-bit | plain Java, JIT auto-vectorized |
| Shapes | all | `n > 1`, `ldw == k`, F32 A/R | all, F32 A/R |
| Operands | native segments only | native segments only | native or heap |
| Packed weights / `share` | yes | no (defaults) | no (defaults) |
| Concurrency | FIFO RW lock (or `EBUSY`) | fair lock | lock |
| Extras | ISA cap, JNI/FFM binding | native-image `Feature`, JIT-aware tiles | correctness reference |

## 3. Shared internal layer

```mermaid
classDiagram
    direction LR
    class GGMLType {
        <<enum, jam-core internal>>
        code, elementsPerBlock, bytesPerBlock
        rowBytes(elements) long
        spanBytes(rows, stride, rowElems) long
        byCode(code)$ GGMLType
    }
    class MemoryChecks {
        <<jam-core internal>>
        requireNative(segment, name)$
        checkSegment(...)$
        checkSpan(...)$
    }
    class NativeJAM
    class VectorJAM
    class ScalarJAM
    MemoryChecks ..> GGMLType
    NativeJAM ..> MemoryChecks
    VectorJAM ..> MemoryChecks
    VectorJAM ..> GGMLType
    ScalarJAM ..> MemoryChecks
    ScalarJAM ..> GGMLType
```

Every backend follows the same `mm` preamble: reject dead / (for native & vector) heap segments,
validate the shape (`EINVAL`), refuse what it cannot run (`EUNSUPPORTED`), compute the exact byte
span of each operand with `GGMLType.spanBytes` and bounds-check it, then run the kernel and fence
the operands with `Reference.reachabilityFence`.

## 4. Backend internals side by side

```mermaid
classDiagram
    direction TB
    namespace scalar {
        class ScalarJAM_
        class Gemv {
            n == 1
        }
        class RowGemm {
            2 <= n < 256
        }
        class Gemm {
            n >= 256
        }
        class Tile {
            3 x 4 FMA sweep
        }
        class Decode {
            row -> float[]
        }
        class ScalarScratch {
            Slot per width
        }
    }
    namespace vector {
        class VectorJAM_
        class XxxKernel {
            11 dtype kernels
        }
        class BandGemm {
            plan, pack, panel, sweep
        }
        class RowDequant {
            dequantize row
        }
        class VectorScratch {
            panel per slot
        }
        class VectorSupport {
            species, GLOBAL, knobs
        }
    }
    namespace native {
        class NativeJAM_
        class NativeLoader {
            extract + load
        }
        class libjam {
            jam.h C API
        }
    }
    ScalarJAM_ --> Gemv
    ScalarJAM_ --> RowGemm
    ScalarJAM_ --> Gemm
    RowGemm --> Tile
    Gemm --> Tile
    Gemv --> Decode
    RowGemm --> Decode
    Gemm --> Decode
    ScalarJAM_ *-- ScalarScratch
    VectorJAM_ --> XxxKernel
    XxxKernel --> BandGemm
    XxxKernel ..|> RowDequant : provides
    BandGemm --> VectorScratch
    BandGemm --> VectorSupport
    VectorJAM_ *-- VectorScratch
    NativeJAM_ --> NativeLoader
    NativeJAM_ --> libjam : JNI / FFM
```

* **scalar**: three kernels chosen by `n`, built on one row decoder and one register tile shape
  both JITs vectorize; all buffers per instance (`Scratch` + one `Slot` per parallel slot).
* **vector**: a strategy pattern, each dtype provides a `RowDequant` and the generic `BandGemm`
  does blocking (L2 panels, L1 k-blocks), packing, scheduling and 3x3/4x4 register sweeps.
* **native**: a thin JNI/FFM handle to a `jam_context`; parallel work comes back to the Java host
  pool through a `jam_parallel_fn` upcall.

## 5. Threading model (all backends)

```mermaid
sequenceDiagram
    participant Host as Host (e.g. jinfer)
    participant B as JAM backend
    participant P as JAM.Parallel (host pool)
    Host->>B: provider.create(parallel)
    Note over B: sizes per-slot scratch to parallel.width()
    Host->>B: mm(...)
    B->>P: run(jobs, (index, slot) -> work)
    P-->>B: all jobs done (slot 0 = calling thread)
    B-->>Host: status
```

No backend owns threads: the host's `JAM.Parallel` runs every task and `width()` is the thread
budget. Per-worker scratch is indexed by `slot`, never by `ThreadLocal`.
