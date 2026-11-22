# jota-tensor — Architecture

`jota-tensor` is the backend-independent core of Jota: the lazy `Tensor` API, the two intermediate representations (TIR, LIR), the scheduling of a traced graph into kernels, and the runtime SPI that the device backends (`jota-backend-panama`, `-c`, `-cuda`, `-hip`, `-metal`, `-opencl`, `-mojo`) implement.

Value types such as `DataType`, `Shape`, `Stride`, `Layout`, `Device` and the `memory` package (`MemoryView`, `MemoryDomain`, `Memory`, …) come from the core module and appear here only as references.

| Package | Classes | Role |
|---|---|---|
| `com.qxotic.jota` | `Environment`, `EnvironmentImpl`, `TypeRules` | Global/scoped configuration: default device, default float type, runtime registry; dtype promotion rules |
| `com.qxotic.jota.tensor` | 22 | Public `Tensor` API, its lazy implementations, tracing to TIR, scheduled execution |
| `com.qxotic.jota.ir` | `IRNode`, `TIRToLIRLowerer`, `TextRenderUtils` | Common IR root and the TIR → LIR lowering |
| `com.qxotic.jota.ir.tir` | 39 | **Tensor IR**: immutable, shape-typed operation DAG, its passes, scheduler and reference interpreter |
| `com.qxotic.jota.ir.lir` | 35 | **Loop IR**: explicit loops, index math, loads/stores over buffers; hash-consed expression graph |
| `com.qxotic.jota.ir.lir.scratch` | 4 | Liveness analysis and scratch-memory layout for intermediate buffers |
| `com.qxotic.jota.runtime` | 27 | Device runtimes, compute engines, kernel programs, compilation, caches, launch arguments |
| `com.qxotic.jota.runtime.spi` | 3 | `DeviceRuntimeProvider` service-loader SPI and probing |
| `com.qxotic.jota.runtime.clike` | 11 | Shared LIR → C-like source generator (C, CUDA, HIP, OpenCL, Metal dialects) |
| `com.qxotic.jota.random` | 4 | Counter-based random number keys and algorithms |

---

## 1. Overview: from a `Tensor` call to a kernel launch

```mermaid
flowchart LR
    subgraph API["tensor package"]
        T["Tensor (API)"] --> TS["TensorSupport / ConstantFolder"]
        TS --> TR["Tracer + IRTensorOps"]
        TR --> IC["IRComputation (LazyComputation)"]
    end
    subgraph TIR["ir.tir"]
        G["TIRGraph"] --> P["TIRCSEPass, TIRConstantFoldingPass"]
        P --> S["TIRSchedulePass"]
        S --> SP["ScheduledProgram = KernelStep*"]
    end
    subgraph EXE["tensor + runtime"]
        SE["ScheduledExecutor"] --> CE["ComputeEngine.execute(stepGraph)"]
    end
    subgraph LIR["ir / ir.lir (in backend engines)"]
        L["TIRToLIRLowerer"] --> LG["LIRGraph"]
        LG --> LP["LIRStandardPipeline"]
        LP --> SA["ScratchAnalysisPass"]
    end
    subgraph GEN["runtime.clike + runtime"]
        KG["CLikeKernelGenerator + dialect"] --> KP["KernelProgram"]
        KP --> KB["KernelBackend.getOrCompile"]
        KB --> KX["KernelExecutable.launch(KernelArgs)"]
    end
    IC -- "materialize()" --> G
    SP --> SE
    CE --> L
    SA --> KG
```

Key ideas:

* **Tensors are lazy.** Every op returns a new `Tensor`; nothing runs until `materialize()`.
* **Tracing** turns a chain of eager-looking calls into an immutable `TIRGraph` (a DAG of sealed `TIRNode` records).
* **Scheduling** splits the TIR graph into `KernelStep`s, each with one materialized output.
* Each step is handed to the device's `ComputeEngine`.
  GPU/C engines lower it to **LIR**, optimize it, generate source code through the C-like generator, compile it (with caching), and launch it.

---

## 2. Environment and runtime discovery (SPI)

`Environment` is the entry point for configuration.
It is resolved through a scoped value (`Environment.with(env, action)`) or the global default (`Environment.global()`), and gives access to the `RuntimeRegistry`.
`DefaultRuntimeRegistry` is filled by `EnvironmentImpl` from every `DeviceRuntimeProvider` found by `ServiceLoader`, filtered by include/exclude backend selectors; each provider reports a `RuntimeProbe` so missing drivers produce a `RuntimeDiagnostic` instead of an error.

```mermaid
classDiagram
    direction LR
    class Environment {
        <<interface>>
        +nativeDevice() Device
        +defaultDevice() Device
        +defaultFloat() DataType
        +runtimes() RuntimeRegistry
        +current()$ Environment
        +global()$ Environment
        +with(Environment, Supplier)$ T
        +runtimeFor(Device)$ DeviceRuntime
        +computeEngineFor(Device)$ ComputeEngine
        +memoryDomainFor(Device)$ MemoryDomain
        +runtimeDiagnostics()$ List~RuntimeDiagnostic~
    }
    class EnvironmentImpl {
        <<record>>
        selectNativeBackend(DefaultRuntimeRegistry)$ Device
        registerProvider(registry, provider, included, excluded)$
    }
    class RuntimeRegistry {
        <<interface>>
        +runtimeFor(Device) DeviceRuntime
        +devices() Set~Device~
        +diagnostics() List~RuntimeDiagnostic~
    }
    class DefaultRuntimeRegistry {
        -Map~Device,DeviceRuntime~ runtimes
        -List~RuntimeDiagnostic~ diagnostics
    }
    class DeviceRuntimeProvider {
        <<abstract>>
        +availableProviders()$ List~DeviceRuntimeProvider~
        +deviceType()* DeviceType
        +priority() int
        +probe()* RuntimeProbe
        +availableDevices() List~AvailableDevice~
        +create(Device) DeviceRuntime
        #createForDevice(Device)* DeviceRuntime
    }
    class RuntimeProbe {
        <<record>>
        status: RuntimeProbeStatus
        message: String
        hint: String
        cause: Throwable
        +isAvailable() boolean
    }
    class RuntimeProbeStatus {
        <<enumeration>>
    }
    class RuntimeDiagnostic {
        <<record>>
        providerId: String
        deviceType: DeviceType
        probe: RuntimeProbe
    }
    class AvailableDevice {
        <<record>>
        device: Device
        properties: Map
        capabilities: Set
    }
    class DeviceRuntime {
        <<interface>>
        +device() Device
        +memoryDomain() MemoryDomain
        +computeEngine() ComputeEngine
        +kernelService() Optional~KernelService~
        +registerKernel(name, KernelProgram) KernelExecutable
        +launchKernel(name, LaunchConfig, args)
    }
    class ComputeEngine {
        <<interface>>
        +device() Device
        +execute(TIRGraph, List~Tensor~) MemoryView
        +precompile(TIRGraph)
        +supportsParallelPrecompile() boolean
    }
    class NativeLibraryLoader {
        loads backend JNI/FFM libs per os-arch
    }

    Environment <|.. EnvironmentImpl
    Environment --> RuntimeRegistry
    RuntimeRegistry <|.. DefaultRuntimeRegistry
    EnvironmentImpl ..> DeviceRuntimeProvider : ServiceLoader
    DefaultRuntimeRegistry o-- DeviceRuntime : lazily created
    DefaultRuntimeRegistry o-- RuntimeDiagnostic
    DeviceRuntimeProvider ..> DeviceRuntime : creates
    DeviceRuntimeProvider ..> RuntimeProbe
    DeviceRuntimeProvider ..> AvailableDevice
    RuntimeProbe --> RuntimeProbeStatus
    RuntimeDiagnostic --> RuntimeProbe
    DeviceRuntime --> ComputeEngine
    DeviceRuntime --> KernelService : optional
```

Backends supply the concrete classes, e.g. `PanamaRuntimeProvider` / `PanamaDeviceRuntime` / `PanamaLirComputeEngine`, `HipRuntimeProvider` / `HipDeviceRuntime` / `HipComputeEngine`, and likewise for CUDA, OpenCL, Metal, Mojo and C.

---

## 3. Tensor API and its implementations

`Tensor` is the only public type of the package (≈160 methods: element-wise, comparison, bitwise/logical, reductions, matmul, gather/embedding, view ops, factories such as `zeros`, `iota`, `rand`, `concat`…).
All implementations share `AbstractTensorImpl`, which dispatches each op through `TensorSupport`: constant folding first, then the IR ops of the active trace, or a fresh one-op trace.

```mermaid
classDiagram
    direction TB
    class Tensor {
        <<interface>>
        +dataType() DataType
        +layout() Layout
        +shape() Shape
        +device() Device
        +materialize() MemoryView
        +add(Tensor) Tensor
        +sum(...) Tensor
        +matmul(Tensor) Tensor
        +reshape(Shape) Tensor
        +to(Device) Tensor
        +zeros(Shape)$ Tensor
        +iota(long)$ Tensor
        +rand(RandomKey, Shape, DataType)$ Tensor
        +of(MemoryView)$ Tensor
    }
    class AbstractTensorImpl {
        <<abstract>>
        ~isMaterializedInternal()* boolean
        ~isLazyInternal()* boolean
        ~tryGetMaterializedInternal()* Optional~MemoryView~
        ~computationInternal()* Optional~LazyComputation~
        ~scalarConstantBitsInternal() OptionalLong
    }
    class MaterializedTensorImpl {
        -MemoryView view
    }
    class LazyTensorImpl {
        -LazyComputation computation
        -DataType dtype
        -Layout layout
        -Device device
        +materialize() MemoryView  (cached)
    }
    class IRTensorImpl {
        -TIRNode node
        -Device device
        ~node() TIRNode
    }
    class LazyComputation {
        <<sealed interface>>
        +inputs() List~Tensor~
        +attributes() Map
        +execute() MemoryView
    }
    class IRComputation {
        <<record>>
        graph: TIRGraph
        inputTensors: List~Tensor~
        ~optimizeGraph(TIRGraph) TIRGraph
    }
    class ConstantComputation {
        <<record>>
        rawBits: long
        dataType: DataType
        shape: Shape
        device: Device
    }
    class RangeComputation {
        <<record>>
        count: long
        device: Device
    }
    class RandomComputation {
        -Shape shape
        -DataType dtype
        -RandomKey key
    }

    Tensor <|.. AbstractTensorImpl
    AbstractTensorImpl <|-- MaterializedTensorImpl
    AbstractTensorImpl <|-- LazyTensorImpl
    AbstractTensorImpl <|-- IRTensorImpl
    LazyTensorImpl --> LazyComputation
    LazyComputation <|.. IRComputation
    LazyComputation <|.. ConstantComputation
    LazyComputation <|.. RangeComputation
    LazyComputation <|.. RandomComputation
    IRComputation --> TIRGraph
    IRTensorImpl --> TIRNode
```

* `MaterializedTensorImpl` wraps an existing `MemoryView` (already computed data).
* `LazyTensorImpl` holds a `LazyComputation` and materializes it once on demand.
* `IRTensorImpl` exists **only during tracing**: it is a symbolic tensor pointing at a `TIRNode`; calling `materialize()` on it is an error.

### Package-private helpers

```mermaid
classDiagram
    direction LR
    class TensorFactory {
        <<utility>>
        of(MemoryView) Tensor
        lazy(LazyComputation, dtype, layout, device) Tensor
        zeros / ones / full / iota
        rand / randn / uniform / normal / randInt
        concat / stack / split
    }
    class TensorSupport {
        <<utility>>
        irOps() TensorOps
        dispatchFoldedBinaryOp(...)
        dispatchScalarBinaryOp(...)
    }
    class ConstantFolder {
        <<utility>>
        folds ops on scalar-broadcast constants
    }
    class TensorStaticOps {
        <<utility>>
    }
    class TensorSemantics {
        <<utility>>
        shape / axis checks
    }
    class TensorTypeSemantics {
        <<utility>>
        result dtype rules
    }
    class TypeRules {
        <<utility>>
        dtype promotion
    }
    class InternalTensorAccess {
        <<utility>>
        isMaterialized(Tensor)
        computation(Tensor)
        irNode(Tensor) TIRNode
    }
    class TensorOps {
        <<interface>>
        add / multiply / exp / ...
        sum / product / max / min
        gather / dot / matmul
        viewTransform / reshape
        cast / to / contiguous
    }
    class IRTensorOps {
        builds TIRNodes, returns IRTensorImpl
    }
    class IRConstantFolder {
        <<utility>>
        foldUnary / foldBinary / foldCast
    }

    TensorOps <|.. IRTensorOps
    TensorSupport ..> ConstantFolder
    TensorSupport ..> Tracer
    ConstantFolder ..> IRConstantFolder
    TensorTypeSemantics ..> TypeRules
    IRTensorOps ..> TensorSemantics
    IRTensorOps ..> TensorTypeSemantics
    TensorFactory ..> LazyTensorImpl
    TensorFactory ..> MaterializedTensorImpl
```

---

## 4. Tracing, scheduling and lazy execution

`Tracer` installs an `IRTensorOps` in a `ScopedValue`.
Inside the scope every op builds TIR nodes; at the end the output node becomes a `TIRGraph` wrapped in an `IRComputation`, returned as a `LazyTensorImpl`.
`Tracer.traceFunction(...)` returns a reusable function built on the same mechanism.

```mermaid
classDiagram
    direction LR
    class Tracer {
        -ScopedValue~TensorOps~ IR_CONTEXT$
        +isTracing()$ boolean
        +trace(List~Tensor~, Function)$ Tensor
        +traceFunction(Function)$ Function
        ~withIR(Supplier)$ T
        ~requireIROps()$ TensorOps
    }
    class IRComputation {
        <<record>>
        +execute() MemoryView
    }
    class TIRSchedulePass {
        -LoweringSupportChecker checker
        +run(TIRGraph) ScheduledProgram
    }
    class TIRStepGraphBuilder {
        cuts a sub-graph at step boundaries
    }
    class LoweringSupportChecker {
        tries TIRToLIRLowerer on a node
    }
    class ScheduledProgram {
        <<record>>
        steps: List~KernelStep~
        output: ScheduledOutputRef
    }
    class KernelStep {
        <<record>>
        graph: TIRGraph
        inputs: List~ScheduleInputRef~
        output: ValueId
    }
    class ValueId {
        <<record>>
        id: int
    }
    class ScheduleInputRef {
        <<sealed interface>>
    }
    class TensorInputRef {
        <<record>> inputId
    }
    class ScalarInputRef {
        <<record>> inputId
    }
    class ProducedValueRef {
        <<record>> valueId
    }
    class ScheduledOutputRef {
        <<sealed interface>>
    }
    class ValueOutput {
        <<record>> valueId
    }
    class TensorInputOutput {
        <<record>> inputId
    }
    class ScalarInputOutput {
        <<record>> inputId
    }
    class ScheduledExecutor {
        ~execute(ScheduledProgram, ComputeEngine, List~Tensor~) MemoryView
    }

    Tracer ..> IRTensorOps : scoped
    Tracer ..> IRComputation : creates
    IRComputation ..> TIRCSEPass
    IRComputation ..> TIRConstantFoldingPass
    IRComputation ..> TIRSchedulePass
    IRComputation ..> ScheduledExecutor
    TIRSchedulePass --> LoweringSupportChecker
    TIRSchedulePass ..> TIRStepGraphBuilder
    TIRSchedulePass ..> ScheduledProgram : produces
    ScheduledProgram *-- KernelStep
    ScheduledProgram --> ScheduledOutputRef
    KernelStep --> ValueId
    KernelStep *-- ScheduleInputRef
    ScheduleInputRef <|.. TensorInputRef
    ScheduleInputRef <|.. ScalarInputRef
    ScheduleInputRef <|.. ProducedValueRef
    ScheduledOutputRef <|.. ValueOutput
    ScheduledOutputRef <|.. TensorInputOutput
    ScheduledOutputRef <|.. ScalarInputOutput
    ScheduledExecutor ..> ComputeEngine : execute each step
```

```mermaid
sequenceDiagram
    participant U as user code
    participant LT as LazyTensorImpl
    participant IC as IRComputation
    participant SP as TIRSchedulePass
    participant SE as ScheduledExecutor
    participant CE as ComputeEngine (backend)
    U->>LT: materialize()
    LT->>IC: execute()
    IC->>IC: TIRCSEPass, TIRConstantFoldingPass
    IC->>SP: run(graph)
    SP-->>IC: ScheduledProgram(steps, output)
    IC->>SE: execute(program, engine, inputs)
    opt engine.supportsParallelPrecompile()
        SE->>CE: precompile(step.graph) in parallel
    end
    loop each KernelStep
        SE->>SE: resolve inputs (graph inputs / produced values)
        SE->>CE: execute(step.graph, stepInputs)
        CE-->>SE: MemoryView (copied to device if intermediate)
    end
    SE-->>LT: output MemoryView (cached)
```

---

## 5. TIR — Tensor IR nodes

TIR nodes are immutable records forming a DAG.
Every node has a `DataType` and a `Shape`; only `TensorInput` and `ViewTransform` carry a layout.
`TIRNode` is sealed, so visitors and `switch` patterns are exhaustive.

```mermaid
classDiagram
    direction TB
    class IRNode {
        <<interface>>
        +dataType() DataType
    }
    class TIRNode {
        <<sealed interface>>
        +shape() Shape
        +accept(TIRVisitor~T~) T
    }
    class TIRGraph {
        <<record>>
        inputs: List~TIRNode~
        outputs: List~TIRNode~
    }
    class TensorInput {
        <<record>>
        id: int
        dataType: DataType
        layout: Layout
    }
    class ScalarInput {
        <<record>>
        id: int
        dataType: DataType
    }
    class ScalarConstant {
        <<record>>
        rawBits: long
        dataType: DataType
    }
    class IotaConstant {
        <<record>>
        count: long
        dataType: DataType
    }
    class RandomUniformOp {
        <<record>>
        dataType: DataType
        key0: long
        key1: long
    }
    class UnaryOp {
        <<record>>
        op: UnaryOperator
        input: TIRNode
    }
    class BinaryOp {
        <<record>>
        op: BinaryOperator
        left: TIRNode
        right: TIRNode
    }
    class TernaryOp {
        <<record>>
        op: TernaryOperator
        cond, trueValue, falseValue
    }
    class CastOp {
        <<record>>
        input: TIRNode
        targetDataType: DataType
    }
    class ReductionOp {
        <<record>>
        op: ReductionOperator
        input: TIRNode
        axes: int[]
        keepDims: boolean
        accumulatorType: DataType
    }
    class GatherOp {
        <<record>>
        input: TIRNode
        indices: TIRNode
        axis: int
    }
    class ViewTransform {
        <<record>>
        input: TIRNode
        operation: ViewOperation
    }
    class Contiguous {
        <<record>>
        input: TIRNode
    }

    IRNode <|-- TIRNode
    TIRGraph o-- TIRNode
    TIRNode <|.. TensorInput
    TIRNode <|.. ScalarInput
    TIRNode <|.. ScalarConstant
    TIRNode <|.. IotaConstant
    TIRNode <|.. RandomUniformOp
    TIRNode <|.. UnaryOp
    TIRNode <|.. BinaryOp
    TIRNode <|.. TernaryOp
    TIRNode <|.. CastOp
    TIRNode <|.. ReductionOp
    TIRNode <|.. GatherOp
    TIRNode <|.. ViewTransform
    TIRNode <|.. Contiguous
```

Operators and view operations:

```mermaid
classDiagram
    direction LR
    class UnaryOperator {
        <<enumeration>>
        NEGATE, ABS, EXP, LOG, SQRT, SIN, COS, TANH, ...
    }
    class BinaryOperator {
        <<enumeration>>
        ADD, SUBTRACT, MULTIPLY, DIVIDE, MIN, MAX, bitwise, logical, compare
    }
    class TernaryOperator {
        <<enumeration>>
        WHERE
    }
    class ReductionOperator {
        <<enumeration>>
        SUM, PROD, MIN, MAX
    }
    class runtime_BinaryOp {
        <<enumeration>>
        runtime.BinaryOp
    }
    class runtime_UnaryOp {
        <<enumeration>>
        runtime.UnaryOp
    }
    class ViewOperation {
        <<sealed interface>>
    }
    class Transpose {
        <<record>>
        permutation: int[]
    }
    class Reshape {
        <<record>>
        shape: Shape
    }
    class Unsqueeze {
        <<record>>
        axis: int
    }
    class Broadcast {
        <<record>>
        shape: Shape
    }
    class Expand {
        <<record>>
        shape: Shape
    }
    class Slice {
        <<record>>
        axis, start, end, step
    }
    ViewOperation <|.. Transpose
    ViewOperation <|.. Reshape
    ViewOperation <|.. Unsqueeze
    ViewOperation <|.. Broadcast
    ViewOperation <|.. Expand
    ViewOperation <|.. Slice
    ViewTransform --> ViewOperation
    UnaryOp --> UnaryOperator
    BinaryOp --> BinaryOperator
    TernaryOp --> TernaryOperator
    ReductionOp --> ReductionOperator
```

`runtime.BinaryOp` / `runtime.UnaryOp` are the tensor-level op identifiers used by `AbstractTensorImpl`/`ConstantFolder` dispatch; they are mapped onto the TIR operator enums.

---

## 6. TIR passes, visitors and the reference interpreter

```mermaid
classDiagram
    direction LR
    class TIRVisitor~T~ {
        <<interface>>
        visitTensorInput(TensorInput) T
        visitBinaryOp(BinaryOp) T
        visitReductionOp(ReductionOp) T
        visitViewTransform(ViewTransform) T
        ...one per TIRNode()
    }
    class TIRRewriter {
        memoized identity rewrite
    }
    class TIRPass {
        <<interface>>
        +run(TIRGraph) TIRGraph
        +name() String
    }
    class TIRCSEPass {
        structural hashing of nodes
    }
    class TIRConstantFoldingPass {
        folds constants and identities
    }
    class TIRTextRenderer {
        TIRGraph to text
    }
    class TIRToLIRLowerer {
        +lower(TIRGraph) LIRGraph
        +lower(TIRGraph, List~Layout~) LIRGraph
    }
    class TIRInterpreter {
        +execute(TIRGraph, inputs)$ List~MemoryView~
    }
    class TIREvalContext {
        AutoCloseable arena + domain
    }
    class TIREvalVisitor {
        evaluates nodes in Java on MemorySegment
    }
    class IRConstantFolder {
        <<utility>>
    }
    class TextRenderUtils {
        <<utility>>
    }

    TIRVisitor <|.. TIRRewriter
    TIRVisitor <|.. TIRTextRenderer
    TIRVisitor <|.. TIREvalVisitor
    TIRVisitor <|.. TIRToLIRLowerer
    TIRPass <|.. TIRCSEPass
    TIRPass <|.. TIRConstantFoldingPass
    TIRCSEPass ..> TIRRewriter : CSERewriter
    TIRConstantFoldingPass ..> TIRRewriter : ConstantFoldingRewriter
    TIRConstantFoldingPass ..> IRConstantFolder
    TIRInterpreter ..> TIREvalVisitor
    TIREvalVisitor --> TIREvalContext
    TIRTextRenderer ..> TextRenderUtils
```

`TIRInterpreter` is a slow but complete Java evaluator, used as an oracle by tests and by backends that fall back to it.
`TIRToLIRLowerer` is the bridge to the next level.

---

## 7. LIR — Loop IR

LIR makes iteration and memory explicit.
A `LIRGraph` has buffer/scalar inputs, output buffers and a body expression.
Nodes live in an `LIRExprGraph`, which hash-conses them (one node per structure), keeps use lists, and runs a canonicalization worklist.
Nodes are split into **scalar** (`S*`) values, **index** (`I*`) arithmetic, and **statements** (`Block`, `Store`, `Yield`, `StructuredFor`); `LIRExprKind` names each kind.

```mermaid
classDiagram
    direction TB
    class LIRGraph {
        <<record>>
        exprGraph: LIRExprGraph
        inputs: List~LIRInput~
        outputs: List~BufferRef~
        body: LIRExprNode
    }
    class LIRGraphBuilder {
        addInput(DataType, Layout) BufferRef
        addScalarInput(DataType) ScalarInput
        addOutput(...) BufferRef
        build(body) LIRGraph
    }
    class LIRExprGraph {
        +resolve(LIRExprNode) LIRExprNode
        +processWorklist()
        +scalarConst / scalarBinary / scalarLoad ...
        +indexConst / indexVar / indexBinary ...
        +block / store / yield / structuredFor
    }
    class LIRInput {
        <<sealed interface>>
        +dataType() DataType
    }
    class BufferRef {
        <<record>>
        id: int
        dataType: DataType
        layout: Layout
    }
    class LIR_ScalarInput {
        <<record>>
        ir.lir.ScalarInput
        id: int
        dataType: DataType
    }
    class LIRExprNode {
        <<abstract>>
        +id() int
        +kind() LIRExprKind
        +dataType() DataType
        +inputs() LIRExprNode[]
        +isPure() boolean
        +uses() Use
        +replaceInput(int, LIRExprNode)
        +canonicalize(LIRExprGraph)* LIRExprNode
    }
    class ScalarNode {
        <<abstract>>
    }
    class IndexNode {
        <<abstract>>
    }
    class SConst
    class SInput
    class SUnary
    class SBinary
    class STernary
    class SCast
    class SLoad {
        buffer: BufferRef
        offset: index
    }
    class SFromIndex
    class SRef
    class IConst
    class IVar
    class IBinary {
        op: IndexBinaryOp
    }
    class IFromScalar
    class Block {
        statements
    }
    class Store {
        buffer: BufferRef
        offset, value
    }
    class Yield {
        values
    }
    class StructuredFor {
        indexName: String
        iterArgs: List~LoopIterArg~
        body: Block
    }
    class LoopIterArg {
        <<record>>
        name: String
        dataType: DataType
        init: LIRExprNode
    }
    class Use {
        user + inputIndex linked list
    }

    LIRGraph --> LIRExprGraph
    LIRGraph o-- LIRInput
    LIRGraph --> LIRExprNode : body
    LIRGraphBuilder ..> LIRGraph : builds
    LIRInput <|.. BufferRef
    LIRInput <|.. LIR_ScalarInput
    LIRExprGraph o-- LIRExprNode : hash-consed
    LIRExprNode --> Use
    LIRExprNode <|-- ScalarNode
    LIRExprNode <|-- IndexNode
    LIRExprNode <|-- Block
    LIRExprNode <|-- Store
    LIRExprNode <|-- Yield
    LIRExprNode <|-- StructuredFor
    ScalarNode <|-- SConst
    ScalarNode <|-- SInput
    ScalarNode <|-- SUnary
    ScalarNode <|-- SBinary
    ScalarNode <|-- STernary
    ScalarNode <|-- SCast
    ScalarNode <|-- SLoad
    ScalarNode <|-- SFromIndex
    ScalarNode <|-- SRef
    IndexNode <|-- IConst
    IndexNode <|-- IVar
    IndexNode <|-- IBinary
    IndexNode <|-- IFromScalar
    StructuredFor *-- LoopIterArg
    StructuredFor *-- Block
    SLoad --> BufferRef
    Store --> BufferRef
```

(`LIRGraphBuilder` is `LIRGraph.Builder`; `LIR_ScalarInput` is `ir.lir.ScalarInput`, not to be confused with the TIR record of the same name.)

Reductions lower to a `StructuredFor` whose `LoopIterArg`s carry the accumulators and whose body ends with a `Yield`; `SRef` reads a loop-carried value by name.

---

## 8. LIR passes, scratch memory and interpreter

```mermaid
classDiagram
    direction LR
    class LIRPass {
        <<interface>>
        +run(LIRGraph) LIRGraph
        +name() String
    }
    class LIRCanonicalizerPass {
        canonicalize + fold via worklist
    }
    class LIRCSEPass {
        dedupe structurally equal nodes
    }
    class LIRStandardPipeline {
        -List~LIRPass~ passes
        -int iterations
        +withIterations(int)
        +withPass(LIRPass)
        +run(LIRGraph) LIRGraph
        +optimize(LIRGraph)$ LIRGraph
    }
    class ScratchAnalysisPass {
        +analyze(LIRGraph) ScratchLayout
    }
    class LivenessInterval {
        <<record>>
        buffer: BufferRef
        firstUse: int
        lastUse: int
    }
    class ScratchLayout {
        <<record>>
        offsets: Map~BufferRef,Long~
        totalByteSize: long
    }
    class ScratchVerificationPass {
        verify(LIRGraph, ScratchLayout) VerificationResult
    }
    class VerificationResult {
        <<record>>
        valid: boolean
        errors: List~String~
        warnings: List~String~
    }
    class LIRInterpreter {
        executes LIRGraph in Java
    }
    class LIRTextRenderer {
        LIRGraph to text
    }

    LIRPass <|.. LIRCanonicalizerPass
    LIRPass <|.. LIRCSEPass
    LIRStandardPipeline o-- LIRPass
    LIRStandardPipeline ..> LIRTextRenderer : verbose dump
    ScratchAnalysisPass ..> LivenessInterval
    ScratchAnalysisPass ..> ScratchLayout : produces
    ScratchVerificationPass ..> ScratchLayout
    ScratchVerificationPass ..> VerificationResult
    LIRInterpreter ..> ScalarArg
```

`ScratchAnalysisPass` computes liveness intervals of intermediate buffers and packs them into a single scratch allocation (`ScratchLayout.offsets` + `totalByteSize`), which the generated kernel receives as one extra buffer.

---

## 9. Kernel programs, compilation and caching

### Kernel programs and execution

```mermaid
classDiagram
    direction LR
    class KernelService {
        <<record>>
        backend: KernelBackend
        sourceStore: KernelProgramStore
        binaryStore: KernelProgramStore
        +register(name, KernelProgram, KernelCacheKey) KernelExecutable
        +loadRegisteredKernel(KernelCacheKey) Optional~KernelProgram~
        +loadRegisteredExecutable(name) Optional~KernelExecutable~
        +bindKernelName(name, KernelCacheKey)
    }
    class KernelBackend {
        <<interface>>
        +compile(KernelProgram, KernelCacheKey) KernelExecutable
        +load(KernelProgram, KernelCacheKey) KernelExecutable
        +getOrCompile(KernelProgram, KernelCacheKey) KernelExecutable
        +cache() KernelExecutableCache
    }
    class KernelProgramStore {
        <<interface>>
        see Kernel caching
    }
    class KernelExecutableCache {
        <<interface>>
        see Kernel caching
    }
    class KernelProgram {
        <<record>>
        kind: Kind
        language: String
        payload: Object
        entryPoint: String
        options: Map
        +source(lang, src, entry)$
        +binary(lang, bytes, entry)$
    }
    class KernelProgramKind {
        <<enumeration>>
        SOURCE
        BINARY
    }
    class KernelExecutable {
        <<interface>>
        +launch(LaunchConfig, KernelArgs, ExecutionStream)
        +close()
    }
    class JavaKernel {
        <<interface>>
        +execute(MemoryDomain, KernelArgs, Memory scratch)
        +scratchByteSize() long
    }
    class LaunchConfig {
        <<record>>
        gridDim x,y,z
        blockDim x,y,z
        sharedMemBytes
        autoLaunch
        +auto()$ / grid(..)$ / block(..)
    }
    class ExecutionStream {
        <<record>>
        device: Device
        handle: Object
        isDefault: boolean
    }
    class KernelArgs {
        +entries() List~Entry~
        addBuffer / addScalar / addMetadata
    }
    class KernelArgsEntry {
        <<record>>
        kind: BUFFER|SCALAR|METADATA
        value: Object
        dataType: DataType
    }
    class ScalarArg {
        <<record>>
        rawBits: long
        dataType: DataType
    }

    KernelService --> KernelBackend
    KernelService --> KernelProgramStore : source + binary
    KernelBackend --> KernelExecutableCache
    KernelBackend ..> KernelExecutable : produces
    KernelBackend ..> KernelProgram
    KernelProgram --> KernelProgramKind
    KernelExecutable ..> LaunchConfig
    KernelExecutable ..> KernelArgs
    KernelExecutable ..> ExecutionStream
    JavaKernel ..> KernelArgs
    KernelArgs *-- KernelArgsEntry
    KernelArgs ..> ScalarArg
    DeviceRuntime --> KernelService
```

### Kernel caching

```mermaid
classDiagram
    direction LR
    class KernelService {
        <<record>>
        sourceStore: KernelProgramStore
        binaryStore: KernelProgramStore
    }
    class KernelBackend {
        <<interface>>
        +cache() KernelExecutableCache
    }
    class KernelExecutableCache {
        <<interface>>
    }
    class KernelProgram {
        <<record>>
    }
    class KernelCacheKey {
        -String value
        +of(String)$ KernelCacheKey
    }
    class KernelProgramHasher {
        hashes program into key
    }
    class KernelProgramStore {
        <<interface>>
        +root() Path
        +load(KernelCacheKey) Optional~KernelProgram~
    }
    class FileKernelProgramStore {
        <<record>>
        root: Path
    }
    class KernelCache {
        <<interface>>
        +root() Path
        +entryFor(KernelCacheKey) KernelCacheEntry
    }
    class DiskKernelCache {
        <<record>>
        root: Path
        packageName: String
    }
    class KernelCacheEntry {
        <<record>>
        key, packageName, className
        directory, sourcePath, classFilePath
    }
    class KernelCachePaths {
        <<utility>>
        cache directory resolution
    }

    KernelService --> KernelProgramStore : source + binary
    KernelBackend --> KernelExecutableCache
    KernelProgramStore <|.. FileKernelProgramStore
    KernelProgramStore ..> KernelProgram : loads
    KernelProgramStore ..> KernelCacheKey
    KernelProgramHasher ..> KernelProgram
    KernelProgramHasher ..> KernelCacheKey
    KernelCache <|.. DiskKernelCache
    DiskKernelCache ..> KernelCachePaths
    DiskKernelCache ..> KernelCacheEntry
    KernelCacheEntry --> KernelCacheKey
```

### Per-backend executable caches

`KernelExecutableCache` (nested in `KernelBackend`) is the in-memory map from a `KernelCacheKey` to a loaded `KernelExecutable`.
Each backend module holds its own private `InMemoryKernelCache` implementation, one per `KernelBackend` instance.
The Mojo backend has none: it wraps another backend and returns that backend's cache.

```mermaid
classDiagram
    direction TB
    namespace jota-tensor {
        class KernelBackend {
            <<interface>>
            +cache() KernelExecutableCache
        }
        class KernelExecutableCache {
            <<interface>>
            +get(KernelCacheKey) KernelExecutable
            +put(KernelCacheKey, KernelExecutable)
        }
    }
    namespace jota-backend-panama {
        class JavaKernelBackend
        class PanamaCache["InMemoryKernelCache"] {
            map: ConcurrentHashMap
        }
    }
    namespace jota-backend-c {
        class CKernelBackend
        class CCache["InMemoryKernelCache"] {
            map: ConcurrentHashMap
        }
    }
    namespace jota-backend-cuda {
        class CudaKernelBackend
        class CudaCache["InMemoryKernelCache"] {
            map: ConcurrentHashMap
        }
    }
    namespace jota-backend-hip {
        class HipKernelBackend
        class HipCache["InMemoryKernelCache"] {
            map: ConcurrentHashMap
        }
    }
    namespace jota-backend-metal {
        class MetalKernelBackend
        class MetalCache["InMemoryKernelCache"] {
            map: ConcurrentHashMap
        }
    }
    namespace jota-backend-opencl {
        class OpenClKernelBackend
        class OpenClCache["InMemoryKernelCache"] {
            map: ConcurrentHashMap
        }
    }
    namespace jota-backend-mojo {
        class MojoKernelBackend {
            delegate: KernelBackend
            +cache() delegate.cache()
        }
    }

    KernelBackend --> KernelExecutableCache : cache()
    KernelExecutableCache <|.. PanamaCache
    KernelExecutableCache <|.. CCache
    KernelExecutableCache <|.. CudaCache
    KernelExecutableCache <|.. HipCache
    KernelExecutableCache <|.. MetalCache
    KernelExecutableCache <|.. OpenClCache
    JavaKernelBackend *-- PanamaCache
    CKernelBackend *-- CCache
    CudaKernelBackend *-- CudaCache
    HipKernelBackend *-- HipCache
    MetalKernelBackend *-- MetalCache
    OpenClKernelBackend *-- OpenClCache
    MojoKernelBackend --> KernelBackend : delegate
```

* `KernelService` is a device runtime's registry of named kernels; it is also how user code registers hand-written kernels (`DeviceRuntime.registerKernel/launchKernel`).
* `DiskKernelCache` + `JavaKernel` serve the Panama backend, which generates and compiles Java classes; GPU/C backends cache compiled binaries through `KernelBackend.cache()` and `FileKernelProgramStore`.

---

## 10. C-like code generation (`runtime.clike`)

One generator serves every C-family backend.
A backend provides a `CLikeDialect` (e.g. `HipDialect`, `CudaDialect`, `OpenClDialect`, `MetalDialect`, C) that subclasses `CLikeLirSourceGenerator` and fills in the dialect-specific pieces: kernel signature, thread-id expression, type names, literals, casts and math function names.

```mermaid
classDiagram
    direction LR
    class CLikeKernelGenerator {
        -CLikeDialect dialect
        +generate(LIRGraph, ScratchLayout, KernelCacheKey) KernelProgram
    }
    class CLikeDialect {
        <<interface>>
        creates the source generator for a graph
    }
    class CLikeLirSourceGenerator {
        <<abstract>>
        -LIRGraph graph
        -ScratchLayout scratchLayout
        -String kernelName
        +generate() String
        #appendPreamble(StringBuilder)*
        #kernelDeclaration()* String
        #emitProlog()*
        #linearIdSpec()* LinearIdSpec
        #renderDataTypeToken(DataType)* String
        #renderCastExpression(...)* String
        #renderBufferReadExpression(...)* String
        #renderBufferWriteStatement(...)* String
        #float32MathFunctionName(String)* String
    }
    class BufferBinding {
        name: String
        readOnly: boolean
    }
    class CLikeParallelLoopSupport {
        <<utility>>
        maps outer loops to thread ids
    }
    class LinearIdSpec {
        <<record>>
        indexType, linearExpr, oneLiteral
    }
    class CLikeKernelSignatureSupport {
        <<utility>>
        input / output argument renderers
    }
    class CLikeExprSupport {
        <<utility>>
    }
    class CLikeScalarSupport {
        <<utility>>
        CastFn, ToFloatFn
    }
    class CLikeLogicalSupport {
        <<utility>>
    }
    class CLikeDataTypeModel {
        <<utility>>
        DataType to C types
    }
    class CLikeSourceSupport {
        <<utility>>
    }
    class LIRKernelArgsBuilder {
        +build(LIRGraph, inputs, outputs) KernelArgs
        +buildGrouped(...) KernelArgs
    }

    CLikeKernelGenerator --> CLikeDialect
    CLikeDialect ..> CLikeLirSourceGenerator : subclass per backend
    CLikeKernelGenerator ..> KernelProgram : source
    CLikeLirSourceGenerator *-- BufferBinding
    CLikeLirSourceGenerator ..> CLikeParallelLoopSupport
    CLikeParallelLoopSupport ..> LinearIdSpec
    CLikeLirSourceGenerator ..> CLikeKernelSignatureSupport
    CLikeLirSourceGenerator ..> CLikeExprSupport
    CLikeLirSourceGenerator ..> CLikeScalarSupport
    CLikeLirSourceGenerator ..> CLikeLogicalSupport
    CLikeLirSourceGenerator ..> CLikeDataTypeModel
    CLikeLirSourceGenerator ..> CLikeSourceSupport
    LIRKernelArgsBuilder ..> KernelArgs
```

Typical backend `ComputeEngine.execute(TIRGraph, inputs)` (e.g. `HipComputeEngine`):

```mermaid
sequenceDiagram
    participant CE as HipComputeEngine
    participant LW as TIRToLIRLowerer
    participant PL as LIRStandardPipeline
    participant SA as ScratchAnalysisPass
    participant GEN as CLikeKernelGenerator(HipDialect)
    participant KB as KernelBackend
    participant AB as LIRKernelArgsBuilder
    participant KX as KernelExecutable
    CE->>LW: lower(tirGraph, inputLayouts)
    LW-->>CE: LIRGraph
    CE->>PL: run(lirGraph)
    PL-->>CE: optimized LIRGraph
    CE->>SA: analyze(lirGraph)
    SA-->>CE: ScratchLayout
    CE->>CE: cacheKey(lirGraph, scratchLayout)
    CE->>GEN: generate(lirGraph, scratchLayout, key)
    GEN-->>CE: KernelProgram (source)
    CE->>KB: getOrCompile(program, key)
    KB-->>CE: KernelExecutable
    CE->>AB: build(lirGraph, inputs, outputs + scratch)
    AB-->>CE: KernelArgs
    CE->>KX: launch(LaunchConfig, args, stream)
```

---

## 11. Random numbers

Random tensors are counter-based: a `RandomKey` (two 64-bit words) plus the element index fully determine each value, so the same result is produced by `RandomComputation` (eager), by the TIR interpreter, and by generated kernels (`RandomUniformOp` carries `key0`/`key1`).

```mermaid
classDiagram
    direction LR
    class RandomKey {
        <<interface>>
        split / fold
    }
    class LcgRandomKey {
        two 64-bit words
    }
    class RandomKeys {
        <<utility>>
        +key(long seed)$ RandomKey
    }
    class RandomAlgorithms {
        <<utility>>
        +uniformFp32(index, key)$ float
        +uniformFp64(index, key)$ double
        +key0(RandomKey)$ long
        +key1(RandomKey)$ long
    }
    RandomKey <|.. LcgRandomKey
    RandomKeys ..> LcgRandomKey : creates
    RandomAlgorithms ..> RandomKey
    RandomComputation ..> RandomAlgorithms
    TIREvalVisitor ..> RandomAlgorithms
    RandomUniformOp ..> RandomAlgorithms : same algorithm in kernels
```
