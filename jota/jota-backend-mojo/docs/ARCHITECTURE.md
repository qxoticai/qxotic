# jota-backend-mojo — class architecture

Packages `com.qxotic.jota.runtime.mojo` (+ `.bridge`, `.codegen`, `.codegen.lir`): 16 classes plus `package-info`, about 2.4k lines.
This is the **experimental Mojo backend** for Jota.
It registers a distinct device type `DeviceType.MOJO`.
Kernels are generated as **Mojo source** from the LIR graph, compiled by the `mojo` CLI to an AMDGPU ELF, and then **loaded and launched through the HIP backend** (`jota-backend-hip`).
Memory, streams and kernel launches are therefore HIP's.
The Mojo layer adds three things: a separate device identity, Mojo code generation, and the Mojo→ELF build step.

Format: **Mermaid**.
Legend: `..|>` implements, `--|>` extends, `-->` holds/uses, `..>` creates/depends, `o--` wraps (decorator).
Classes marked `<<external>>` live outside this module.

1. [Position in the jota stack](#1-position-in-the-jota-stack)
2. [Class diagrams](#2-class-diagrams): 2a runtime and SPI, 2b memory decorators, 2c kernel compilation, 2d code generation, 2e native bridge and tools
3. [Execution sequence](#3-execution-sequence-mojocomputeengineexecute)
4. [Mojo→GPU binary compilation](#4-mojo--gpu-binary-compilation)
5. [Code generation: LIR → Mojo source](#5-code-generation-lir--mojo-source)
6. [On-disk cache layout](#6-on-disk-cache-layout)
7. [Configuration knobs](#7-configuration-knobs)

| Class | Package | Role |
|---|---|---|
| `MojoRuntimeProvider` | `mojo` | `DeviceRuntimeProvider` SPI (ServiceLoader) for `DeviceType.MOJO`, priority 50. Its probe checks the HIP probe, then `mojo --version`, then that the JNI lib loads |
| `MojoDeviceRuntime` | `mojo` | `DeviceRuntime`: wires `HipMojoExecutionEngine` and `MojoComputeEngine` together |
| `MojoExecutionEngine<T>` | `mojo` | Package-private interface: memory domain, pointer→address, `KernelService`, `getOrCompile` |
| `HipMojoExecutionEngine` | `mojo` | The only implementation, built on `HipDeviceRuntime`. Wraps the HIP domain and kernel backend in Mojo decorators |
| `MojoComputeEngine<T>` | `mojo` | `ComputeEngine`: TIR→LIR→Mojo source, cache key, registration, arg building, launch config, launch |
| `MojoKernelBackend` | `mojo` | `KernelBackend` decorator over HIP's backend. Turns Mojo `SOURCE` programs into HIP `BINARY` (ELF) programs, with a disk cache and stats |
| `MojoMemoryDomain<T>` | `mojo` | `MemoryDomain` decorator. Changes the device identity to MOJO and unwraps `MojoMemory` before `copy` |
| `MojoMemoryAllocator<T>` | `mojo` | `MemoryAllocator` decorator that wraps every allocation in a `MojoMemory` |
| `MojoMemory<T>` | `mojo` | `Memory` decorator: HIP device pointer with a MOJO `Device` |
| `MojoCachePaths` | `mojo` | Static helpers for the on-disk cache layout (`<version>/mojo/sources/lir/<key>/…`) |
| `MojoLirKernelGenerator` | `codegen.lir` | Public entry point: `LIRGraph` + `ScratchLayout` + key → `KernelProgram.source("mojo", …)` |
| `MojoNativeLirSourceGenerator` | `codegen.lir` | Emits Mojo text from the LIR expression graph (parallel loop nest or serial fallback) |
| `MojoLirArtifactStore` | `codegen.lir` | Writes the generated `kernel.mojo` to the cache directory |
| `MojoDialect` | `codegen` | Enum `V0_26_1` / `V0_26_2` / `V1`: the Mojo syntax to generate (GPU module `std.gpu` or `max.gpu`, `fn` or `def`, wrapper `main` and `compile_function` form), parsed from `mojo --version` |
| `MojoKernelSourceCli` | `codegen` | Standalone `main()`: template generator for f32 rank-1/rank-2 elementwise kernels (`add/sub/mul/div/relu`), with `--mojo-version` |
| `MojoRuntime` | `bridge` | JNI bindings to `libjota_mojo` (`initWithBackend`, `shutdown`, `lastError`). In main code only `isAvailable()` is used, by the provider probe |

---

## 1. Position in the jota stack

```mermaid
flowchart TB
    User["Tensor API (jota-tensor)<br/>Environment / Tensor.materialize()"]
    subgraph core["jota-tensor (SPI + IR)"]
        SPI["DeviceRuntimeProvider / DeviceRuntime<br/>ComputeEngine / MemoryDomain"]
        IR["TIRGraph → TIRToLIRLowerer → LIRGraph<br/>LIRStandardPipeline · ScratchAnalysisPass"]
        KS["KernelService · KernelBackend<br/>KernelProgram · KernelCacheKey"]
        Clike["clike.LIRKernelArgsBuilder<br/>CLikeExprSupport"]
    end
    subgraph mojo["jota-backend-mojo"]
        Prov["MojoRuntimeProvider"]
        DR["MojoDeviceRuntime"]
        CE["MojoComputeEngine"]
        Gen["codegen.lir:<br/>MojoLirKernelGenerator"]
        KB["MojoKernelBackend"]
        Mem["MojoMemoryDomain / Allocator / Memory"]
        Br["bridge.MojoRuntime (JNI)"]
    end
    subgraph hip["jota-backend-hip"]
        HDR["HipDeviceRuntime"]
        HKB["HIP KernelBackend<br/>(hipModuleLoad / launch)"]
        HMD["HIP MemoryDomain<br/>(HipDevicePtr)"]
        HR["HipRuntime (arch name)"]
    end
    Tool["mojo CLI<br/>(mojo build --emit asm)"]
    Nm["llvm-nm"]
    GPU["AMD GPU (ROCm)"]
    Lib["libjota_mojo.so"]

    User --> SPI
    Prov -.->|extends| SPI
    Prov --> DR --> CE
    CE --> IR
    CE --> Gen
    CE --> Clike
    CE --> KS
    KS --> KB
    KB -->|delegates| HKB
    KB --> Tool
    KB --> Nm
    KB --> HR
    Mem -->|delegates| HMD
    DR --> HDR
    HDR --> HKB
    HDR --> HMD
    HKB --> GPU
    HMD --> GPU
    Prov --> Br --> Lib
```

The module has **no GPU driver code of its own**.
Device work goes through HIP, and kernel builds go through an external process (`mojo`).
The JNI library `libjota_mojo` is checked for availability only.
Its context API is exposed in Java but nothing in the main code calls it.

---

## 2. Class diagrams

### 2a. Runtime and SPI wiring

```mermaid
classDiagram
    direction LR
    class DeviceRuntimeProvider { <<abstract, external>> }
    class DeviceRuntime { <<interface, external>> }
    class ComputeEngine { <<interface, external>> }
    class KernelService { <<external>> }
    class HipDeviceRuntime { <<external>> }
    class HipRuntimeProvider { <<external>> }
    class FileKernelProgramStore { <<external>> }

    class MojoRuntimeProvider {
        <<final>>
        +deviceType() DeviceType = MOJO
        +priority() int = 50
        +probe() RuntimeProbe
        #createForDevice(Device) DeviceRuntime
        +properties(int) Map
        +capabilities(int) Set = gpu, fp32, kernel.compilation
        -isCommandAvailable(List~String~)$ boolean
    }
    class MojoDeviceRuntime {
        <<final>>
        -executionEngine : MojoExecutionEngine~?~
        -computeEngine : ComputeEngine
        +device() Device
        +memoryDomain() MemoryDomain
        +computeEngine() ComputeEngine
        +kernelService() Optional~KernelService~
    }
    class MojoExecutionEngine~T~ {
        <<interface>>
        +memoryDomain() MojoMemoryDomain~T~
        +addressOf(T) long
        +kernelService() KernelService
        +getOrCompile(KernelProgram, KernelCacheKey) KernelExecutable
    }
    class HipMojoExecutionEngine {
        <<final>>
        -memoryDomain : MojoMemoryDomain~HipDevicePtr~
        -kernelService : KernelService
        +addressOf(HipDevicePtr) long
    }
    class MojoComputeEngine~T~ {
        <<final>>
        +device() Device
        +execute(TIRGraph, List~Tensor~) MemoryView
        +precompile(TIRGraph)
        +supportsParallelPrecompile() true
    }

    MojoRuntimeProvider --|> DeviceRuntimeProvider
    MojoRuntimeProvider ..> MojoDeviceRuntime : creates
    MojoRuntimeProvider ..> HipRuntimeProvider : probe()
    MojoRuntimeProvider ..> MojoKernelBackend : resolveCompilerExecutable()
    MojoRuntimeProvider ..> MojoRuntime : isAvailable()
    MojoDeviceRuntime ..|> DeviceRuntime
    MojoDeviceRuntime --> MojoExecutionEngine
    MojoDeviceRuntime --> MojoComputeEngine
    HipMojoExecutionEngine ..|> MojoExecutionEngine
    HipMojoExecutionEngine ..> HipDeviceRuntime : creates
    HipMojoExecutionEngine --> MojoMemoryDomain
    HipMojoExecutionEngine --> KernelService
    KernelService --> MojoKernelBackend : backend
    KernelService --> FileKernelProgramStore : source + binary stores
    MojoComputeEngine ..|> ComputeEngine
    MojoComputeEngine --> MojoExecutionEngine
```

`MojoDeviceRuntime`'s constructor always builds a `HipMojoExecutionEngine`.
`MojoExecutionEngine<T>` is the seam where a non-HIP engine (CUDA, for example) could be added later, but HIP is the only implementation today.

`HipMojoExecutionEngine` builds a `KernelService` with these parts:

- `backend`: `new MojoKernelBackend(hipRuntime.kernelService().backend())`
- `sourceStore`: `FileKernelProgramStore(KernelCachePaths.programRoot(MOJO)/source)`
- `binaryStore`: `FileKernelProgramStore(KernelCachePaths.programRoot(MOJO)/binary)`

### 2b. Memory decorators (device identity)

```mermaid
classDiagram
    direction TB
    class MemoryDomain~T~ { <<interface, external>> }
    class MemoryAllocator~T~ { <<interface, external>> }
    class Memory~T~ { <<interface, external>> }
    class MemoryView~T~ { <<external>> }

    class MojoMemoryDomain~T~ {
        <<final>>
        -device : Device (MOJO:n)
        -delegate : MemoryDomain~T~
        -allocator : MojoMemoryAllocator~T~
        +memoryAllocator() MemoryAllocator~T~
        +directAccess() delegate
        +memoryOperations() delegate
        +copy(MemoryView, MemoryView)
        -unwrap(MemoryView)$ MemoryView
        +close()
    }
    class MojoMemoryAllocator~T~ {
        <<final>>
        -device : Device
        -delegate : MemoryAllocator~T~
        +allocateMemory(byteSize, align) Memory~T~
        +defaultByteAlignment() delegate
        +memoryGranularity() delegate
    }
    class MojoMemory~T~ {
        <<final>>
        -device : Device
        -delegate : Memory~T~
        ~delegate() Memory~T~
        +base() T
        +byteSize() long
        +device() Device (MOJO)
    }

    MojoMemoryDomain ..|> MemoryDomain
    MojoMemoryAllocator ..|> MemoryAllocator
    MojoMemory ..|> Memory
    MojoMemoryDomain o-- MemoryDomain : delegate (HIP)
    MojoMemoryAllocator o-- MemoryAllocator : delegate (HIP)
    MojoMemory o-- Memory : delegate (HIP)
    MojoMemoryDomain --> MojoMemoryAllocator : creates
    MojoMemoryAllocator ..> MojoMemory : wraps each allocation
    MojoMemoryDomain ..> MemoryView : unwrap() before delegate.copy
```

All three classes follow the same pattern.
Each is a **thin decorator** that only changes `device()` to the `MOJO` device of the same index, so tensors report `mojo:N` instead of `hip:N`.
Each constructor rejects a delegate that is already a Mojo wrapper, so wrappers never nest.
`copy` unwraps `MojoMemory` views back to the raw HIP memory, because the HIP domain cannot copy a foreign `Memory` type.

### 2c. Kernel compilation (`MojoKernelBackend`)

```mermaid
classDiagram
    direction LR
    class KernelBackend { <<interface, external>> }
    class KernelExecutableCache { <<external>> }
    class KernelProgram { <<external>> }
    class HipRuntime { <<external>> }

    class MojoKernelBackend {
        <<final>>
        ~COMPILER_PROPERTY = "jota.mojo.compiler"$
        ~COMPILER_ENV = "JOTA_MOJO_COMPILER"$
        -CACHE_HITS, CACHE_MISSES, NATIVE_COMPILES, BINARY_LOADS$
        -COMPILE_NANOS_TOTAL, COMPILE_NANOS_MAX, SEEN_KEYS$
        -delegate : KernelBackend
        +compile(KernelProgram, KernelCacheKey) KernelExecutable
        +load(KernelProgram, KernelCacheKey) KernelExecutable
        +getOrCompile(KernelProgram, KernelCacheKey) KernelExecutable
        +cache() KernelExecutableCache
        -adaptProgram(KernelProgram, key)$ KernelProgram
        -compileMojoToGpuBinary(KernelProgram, key)$ KernelProgram
        -loadCachedBinaryProgram(...)$ KernelProgram
        -persistBinary / clearPersistedBinary$
        -resolveTarget()$ String
        ~resolveCompilerExecutable()$ String
        ~dialect()$ MojoDialect
        ~wrapper(source, fnName, MojoDialect)$ String
        -extractElfFromAsm(Path)$ byte[]
        -decodeCString(String, int)$ byte[]
        -readElfSymbol(byte[], String)$ String
        -execCapture(List, timeout)$ String
    }
    class MojoCachePaths {
        <<final, utility>>
        +cacheRoot()$ Path
        +sourceRoot()$ Path
        +lirSourceRoot()$ Path
        +lirKernelDir(key)$ Path
        +lirSourcePath(key)$ Path
        +lirWrapperPath(key)$ Path
        +lirAsmPath(key)$ Path
        +lirBinaryPath(key)$ Path
        +lirEntryPath(key)$ Path
        +lirBridgeSourcePath(key)$ Path
    }

    MojoKernelBackend ..|> KernelBackend
    MojoKernelBackend o-- KernelBackend : delegate (HIP)
    MojoKernelBackend --> KernelExecutableCache : delegate.cache()
    MojoKernelBackend ..> MojoCachePaths
    MojoKernelBackend ..> HipRuntime : deviceArchName()
    MojoKernelBackend ..> KernelProgram : SOURCE(mojo) → BINARY(hip)
```

`MojoKernelBackend` is a **program adapter**.
Any `KernelProgram` with `kind=SOURCE` and `language="mojo"` is turned into `KernelProgram.binary("hip", elf, entrySymbol, options)` before it reaches the HIP backend.
Other programs, such as HIP C++ source or a precompiled binary, pass through unchanged.
If an executable is already in the in-memory cache, the backend returns it before adapting anything.
If a cached on-disk ELF fails to load, the backend deletes it, rebuilds it once, and tries again.

### 2d. Code generation (`codegen.lir`)

```mermaid
classDiagram
    direction LR
    class LIRGraph { <<external>> }
    class LIRExprGraph { <<external>> }
    class ScratchLayout { <<external>> }
    class CLikeExprSupport { <<external>> }

    class MojoLirKernelGenerator {
        <<final, public>>
        -dialect : MojoDialect
        +MojoLirKernelGenerator()
        +MojoLirKernelGenerator(MojoDialect)
        +generate(LIRGraph, ScratchLayout, KernelCacheKey) KernelProgram
    }
    class MojoDialect {
        <<enumeration>>
        V0_26_1
        V0_26_2
        V1
        +LATEST$ = V1
        +gpuModule() String
        +kernelKeyword() String
        +mainSignature() String
        +compileFunctionCall(ctx, fn) String
        +fromVersionText(String)$ Optional~MojoDialect~
    }
    class MojoNativeLirSourceGenerator {
        <<final>>
        -dialect : MojoDialect
        -graph : LIRGraph
        -exprGraph : LIRExprGraph
        -scratchLayout : ScratchLayout
        -kernelName : String
        -lines : List~String~
        -buffers : Map~BufferRef,String~
        -scalarInputs : Map~Integer,String~
        -cachedScalars : Map~LIRExprNode,String~
        -scopes : Deque~Scope~
        ~generate() String
        -enterBlock() / exitBlock()
        -declare(name, expr)
        -emitHeader() / emitProlog()
        -tryEmitParallelBody(LIRExprNode) boolean
        -emitNode / emitStructuredFor / emitStore
        -emitScalarExpr / emitScalarInline
        -emitUnary / emitBinary / emitTernary / emitCast
        -emitIndexExpr(LIRExprNode) String
        -scalarLiteral / castExpr / forceCastExpr
        -mojoType / mojoScalarType / mojoPointerType
    }
    class MojoLirArtifactStore {
        <<final, utility>>
        ~persist(KernelCacheKey, KernelProgram)$
    }

    MojoComputeEngine --> MojoLirKernelGenerator
    MojoLirKernelGenerator --> MojoDialect
    MojoNativeLirSourceGenerator --> MojoDialect
    MojoLirKernelGenerator ..> MojoNativeLirSourceGenerator : new per kernel
    MojoLirKernelGenerator ..> MojoLirArtifactStore : persist kernel.mojo
    MojoNativeLirSourceGenerator --> LIRGraph
    MojoNativeLirSourceGenerator --> LIRExprGraph : resolve(node)
    MojoNativeLirSourceGenerator --> ScratchLayout
    MojoNativeLirSourceGenerator ..> CLikeExprSupport : float literals
    MojoLirArtifactStore ..> MojoCachePaths
```

The entry point name is `hip_lir_<cacheKey with '-'→'_'>`.
`MojoComputeEngine` builds its generator with `MojoKernelBackend.dialect()`, which is read once from `-Djota.mojo.version` or else from `mojo --version` (default `MojoDialect.LATEST`).
The program is `KernelProgram.source("mojo", src, fnName, {})`.

### 2e. Native bridge and stand-alone tool

```mermaid
classDiagram
    direction LR
    class NativeLibraryLoader { <<external>> }
    class MojoRuntime {
        <<final, bridge>>
        +ABI_VERSION = 1$
        +DEFAULT_BACKEND = "hip"$
        -LIB_NAME = "jota_mojo"$
        -AVAILABLE : AtomicReference~Boolean~$
        +isAvailable()$ boolean
        +requireAvailable()$
        +init(abi, target)$ long
        +initWithBackend(abi, target, backend)$ long «native»
        +shutdown(handle)$ «native»
        +lastError(handle)$ String «native»
    }
    class libjota_mojo {
        <<native C, src/main/native>>
        jota_mojo_context_init()
        jota_mojo_context_shutdown()
        jota_mojo_context_last_error()
    }
    class MojoKernelSourceCli {
        <<final, main>>
        +main(String[])$
        -renderRank1(op, name)$ String
        -renderRank2(op, name, lhsB0, lhsB1, rhsB0, rhsB1)$ String
        -expression(op, lhsIdx, rhsIdx)$ String
        -kernelNameFor(op, rank, cacheKey)$ String
    }
    MojoRuntime ..> NativeLibraryLoader : load("jota_mojo")
    MojoRuntime ..> libjota_mojo : JNI
    MojoRuntimeProvider ..> MojoRuntime : probe
```

`MojoKernelSourceCli` is not part of the runtime path.
It is a separate generator for hand-written-style kernels (`--operation add|sub|mul|div|relu --rank 1|2 --dtype f32 --output <file>` and optional `--{lhs,rhs}-broadcast{0,1}`).
Kernels from this tool use `global_idx.x` and explicit strides.

---

## 3. Execution sequence (`MojoComputeEngine.execute`)

```mermaid
sequenceDiagram
    autonumber
    participant T as Tensor / Environment
    participant CE as MojoComputeEngine
    participant IR as TIRToLIRLowerer + LIRStandardPipeline
    participant SC as ScratchAnalysisPass
    participant MD as MojoMemoryDomain
    participant G as MojoLirKernelGenerator
    participant KS as KernelService
    participant KB as MojoKernelBackend
    participant H as HIP KernelBackend

    T->>CE: execute(TIRGraph, inputs)
    CE->>T: inputs[i].materialize() (skip ScalarInput)
    CE->>IR: lower(graph, runtime layouts) then pipeline.run
    IR-->>CE: LIRGraph
    CE->>SC: analyze(lir) → ScratchLayout (verify if -Djota.verifyScratch)
    CE->>MD: allocate outputs (+ scratch, 64-byte aligned)
    CE->>CE: cacheKey = SHA-256(LIR text + scratch offsets) + "-mojo-lir-v13"
    CE->>G: generate(lir, scratch, key)
    G-->>CE: KernelProgram SOURCE "mojo" (kernel.mojo persisted)
    CE->>KS: bindKernelName(key) and register(program, key)
    CE->>CE: resolveInputs: copy non-MOJO views to device (BufferSpec)
    CE->>CE: LIRKernelArgsBuilder.buildGroupedWithWorkspaceScalar(..., scratchPtr I64)
    CE->>CE: LaunchConfig: 256 threads, ceil(outSize/256) blocks
    CE->>KS: loadRegisteredExecutable(key)
    alt not registered as executable
        CE->>KB: getOrCompile(program, key)
        KB->>H: getOrCompile(BINARY hip ELF, key)
        H-->>CE: KernelExecutable
    end
    CE->>H: executable.launch(config, args, stream)
    alt default runtime aliases native
        CE-->>T: copy output[0] to host MemorySegment (row-major)
    else
        CE-->>T: output[0] (device MemoryView)
    end
```

`precompile(TIRGraph)` takes the same path up to `getOrCompile`, using the static layouts from `TensorInput.layout()`.
It swallows any `RuntimeException`, because some graphs need layouts that are only known at run time.

---

## 4. Mojo → GPU binary compilation

```mermaid
flowchart TD
    A["getOrCompile(program, key)"] --> B{"delegate.cache().get(key)?"}
    B -- hit --> Z["return cached KernelExecutable<br/>CACHE_HITS++"]
    B -- miss --> C{"SOURCE &amp; language == mojo?"}
    C -- no --> P["pass program through unchanged"]
    C -- yes --> D{"kernel.elf + kernel.entry<br/>on disk?"}
    D -- yes --> E["KernelProgram.binary(hip, elf, entry)<br/>BINARY_LOADS++"]
    D -- no --> F["write kernel.mojo + kernel.wrapper.mojo<br/>(wrapper adds main() with<br/>DeviceContext(api=hip).compile_function[...]()<br/>in the detected MojoDialect)"]
    F --> G["mojo build wrapper --emit asm<br/>--target-accelerator &lt;gfxXXXX&gt; -o kernel.mojo.s<br/>(+ jota.mojo.compile.flags, timeout 30 s)"]
    G --> H["extractElfFromAsm: find \\177ELF string<br/>and decode the C-string escapes"]
    H --> I["readElfSymbol: llvm-nm, first T/t symbol<br/>(fallback: fnName)"]
    I --> J["persist kernel.elf + kernel.entry<br/>NATIVE_COMPILES++"]
    J --> E
    E --> K["HIP delegate.getOrCompile(BINARY)"]
    P --> K
    K -- ok --> Y["KernelExecutable"]
    K -- RuntimeException on cached binary --> R["clearPersistedBinary, then<br/>compileMojoToGpuBinary again<br/>and delegate.getOrCompile"]
    R --> Y
```

The GPU target is resolved in this order: `-Djota.mojo.target`, then `JOTA_MOJO_TARGET`, then `HipRuntime.deviceArchName(currentDevice)`, then the hard-coded default `gfx1103`.
With `-Djota.mojo.kernel.summary=true`, a shutdown hook prints the cache and compile counters.

---

## 5. Code generation: LIR → Mojo source

```mermaid
flowchart TD
    S["generate()"] --> I["imports: max.gpu or std.gpu (block_idx, block_dim, thread_idx), std.math"]
    I --> H["emitHeader: def or fn name(inputN: UnsafePointer[T], scalarN: T,<br/>outputN: UnsafePointer[T], scratch_ptr: UInt64)"]
    H --> P["emitProlog: var gid = block_idx.x * block_dim.x + thread_idx.x<br/>var scratchK = UnsafePointer[T](unsafe_from_address=Int(scratch_ptr + offset))"]
    P --> Q{"tryEmitParallelBody:<br/>body is a perfect StructuredFor nest,<br/>lb = 0, step = 1, no iterArgs?"}
    Q -- yes --> PAR["total = product of upper bounds<br/>if gid &lt; total:<br/>  de-linearize gid into loop indices (% and //)<br/>  emit innermost statements (skip Yield)"]
    Q -- no --> SER["if block_idx.x == 0 and thread_idx.x == 0:<br/>  emitNode(body) serially<br/>(for loops, iterArgs as reductions)"]
```

| LIR node | Mojo emission |
|---|---|
| `Block` / `StructuredFor` / `Store` / `Yield` | statements / `for i in range(lb, ub, step):` / `buf[Int(off // size)] = v` / iterArg update |
| `SConst` | typed literal (`Float32(…)`, `Int64(…)`, `True`). NaN becomes `0.0` and ±Inf becomes ±max finite |
| `SInput` / `SRef` / `SLoad` / `SFromIndex` | `scalarN` / name / `buf[idx]` / forced cast of the index expression |
| `SUnary` | `-x`, abs via ternary, `exp` as `e ** x`, `log` as `log2(x)·ln2`, `sqrt`, trig (FP64 computed in Float32), `1/x`, `not`, `~` |
| `SBinary` | arithmetic, min/max via ternary, `**`, logical ops via truthiness, bitwise ops, masked shifts, `>>>` via `UInt32`/`UInt64` |
| `STernary` / `SCast` | `(t if c else f)` / `TypeName(x)` |
| `IConst` / `IVar` / `IBinary` / `IFromScalar` | integer index arithmetic (`//`, `%`, `<<`, …) / `Int(scalar)` |

A scalar subexpression used more than once (`useCount > 1`, unary/binary/ternary/cast) is hoisted into a temporary `tN` and cached by node identity.
Every local is declared with `var`, which Mojo 1.x requires.
A `var` lives only in its indented block, so the generator tracks declared names per block and drops cached temporaries when their block ends; a value needed again after a loop is recomputed instead of read from a temporary declared inside the loop.
Literals and shift amounts carry the operand's type (`Float32(1.0) / x`, `x << Int32(...)`), and a Bool becomes a float through `Int` (`Float32(Int(b))`), because Mojo 1.x no longer mixes these implicitly.

---

## 6. On-disk cache layout

```mermaid
flowchart LR
    R["KernelCachePaths.versionRoot()"] --> M["mojo/"]
    M --> S["sources/"] --> L["lir/"]
    L --> K["&lt;cacheKey&gt;/"]
    K --> k1["kernel.mojo (generated source)"]
    K --> k2["kernel.wrapper.mojo (source + main())"]
    K --> k3["kernel.mojo.s (mojo --emit asm)"]
    K --> k4["kernel.elf (AMDGPU code object)"]
    K --> k5["kernel.entry (symbol for hipModuleGetFunction)"]
    L --> hb["&lt;cacheKey&gt;.hip (lirBridgeSourcePath: defined, unused)"]
    PR["KernelCachePaths.programRoot(MOJO)"] --> ps["source/ (KernelService sourceStore)"]
    PR --> pb["binary/ (KernelService binaryStore)"]
```

---

## 7. Configuration knobs

| Key | Kind | Default | Used by |
|---|---|---|---|
| `jota.mojo.compiler` / `JOTA_MOJO_COMPILER` | sysprop / env | `mojo` | `MojoKernelBackend`, provider probe |
| `jota.mojo.target` / `JOTA_MOJO_TARGET` | sysprop / env | HIP device arch, else `gfx1103` | `MojoKernelBackend.resolveTarget` |
| `jota.mojo.compile.flags` | sysprop | empty (quote-aware tokenized) | extra `mojo build` args |
| `jota.mojo.compile.timeout.seconds` | sysprop | `30` | `mojo build` timeout |
| `jota.mojo.kernel.summary` | sysprop | `false` | shutdown-hook stats line |
| `jota.verifyScratch` | sysprop | `false` | `MojoComputeEngine` scratch verification |
| `jota.mojo.version` | sysprop | output of `mojo --version` | `MojoKernelBackend.dialect()`: which `MojoDialect` to generate |

Provider-reported properties: `device.name=Mojo (via HIP)`, `device.vendor=Modular`, `device.kind=gpu`, `mojo.execution.backend=hip`.
