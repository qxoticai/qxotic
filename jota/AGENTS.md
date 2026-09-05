# Jota Agent Guide

Guidance for coding agents working in `jota/`, the tensor engine of the qxotic monorepo.
Read the module README for the user-facing story; this file is about how the code is laid out and how to prove a change.

## What jota is

Jota (JVM Open Tensor Algebra) is a tensor algebra library with lazy evaluation, kernel compilation and multi-device support.
It is one subtree of the qxotic multi-module Maven build; the root `Makefile` and root `pom.xml` own the shared configuration.
Only `jota-core` and `jota-memory` are published to Maven Central at 0.2.0; `jota-tensor` and the backends build from source.

## Modules

| module | role |
|---|---|
| `jota-core` | data types, devices, shapes, strides, layouts, views; a real JPMS module `com.qxotic.jota` that exports only `com.qxotic.jota` |
| `jota-memory` | the memory abstraction: domains, memories, views, access, bulk operations; backends on heap arrays, ByteBuffer, Unsafe and MemorySegment |
| `jota-tensor` | tensors, tracing, the TIR and LIR intermediate representations, scheduling, the native library loader |
| `jota-backend-panama` | the always-on Java backend; the one the default test run exercises |
| `jota-backend-c`, `-cuda`, `-hip`, `-metal`, `-opencl`, `-mojo` | native and GPU backends, each behind a profile of the same name (`-Pc`, `-Pcuda`, `-Phip`, `-Pmetal`, `-Popencl`, `-Pmojo`, or `-Pall`) |

`com.qxotic.jota.impl` is internal: it is not exported by the module descriptor and is not documented.
Code that needs it from the module path must ask for an export rather than reach in.

## Build and test

Work from the repository root with `make`, or from `jota/` with Maven.
`mvnd` keeps a warm daemon and makes the edit, build, test loop faster; the commands below accept either.

```bash
make jota-test                       # from the root: core, memory, and the tensor suite on the Java backend
cd jota && mvnd test                 # the same, from the subtree
mvnd -Pc test                        # add the C backend (needs cmake and a C compiler)
mvnd test -Dtest=ShapeTest           # one class
mvnd test -Dtest=ShapeTest#nested    # one method
mvnd spotless:apply                  # Google Java Format, AOSP style; run before finishing
```

The tensor suite lives in `jota-tensor/src/testkit/java`, not under `src/test`.
It needs a runtime provider, so every backend compiles it as its own tests through `build-helper`, and `jota-tensor` alone runs nothing.
`jota-memory/src/testkit/java` is shared the same way; there are no test jars anywhere in the reactor.
A backend's tests run only when its profile is active; the panama backend is always on.
Tests run with `--enable-native-access=ALL-UNNAMED`, set once in `jota/pom.xml`, because jota-memory and the native backends call restricted methods.
Java 25 is required; the root enforcer says so.

Native libraries are packaged under `META-INF/native/<os>/<arch>/`.
The `os.name` and `os.arch` properties in `jota/pom.xml` are normalised names the loader looks up, set by the platform profiles from the building JVM; `native-aarch64` is what makes Apple Silicon and linux-aarch64 land in the right directory.

## Architecture

### Tensor evaluation

Lazy tensors: `Tracer.trace(input, fn)` records operations into a `TIRGraph` (`ir/tir/`: `TensorInput`, `ScalarInput`, `BinaryOp`, `ReductionOp` and friends).
`tensor.materialize()` hands the graph to `ScheduledExecutor`, which lowers and runs it on the active backend.

Eager tensors: a materialized tensor (`MaterializedTensorImpl`) executes operations directly on a `MemoryView`.
This is the default outside a traced context.

### Memory layers

1. `MemoryDomain<B>`: the device context, with allocator, access and operations.
2. `Memory<B>`: the backing buffer.
3. `MemoryView<B>`: a view with a `Layout` (shape plus stride) over a memory.
4. `MemoryAccess<B>`: element reads and writes.
5. `MemoryOperations<B>`: bulk copy, broadcast, reshape.

### Shapes

Shapes are immutable and nest, CuTe style: `Shape.of(2, Shape.of(3, 4), 5)`.
Wrap-around indexing convention: a `_prefix` wraps with respect to the input, a `suffix_` with respect to the output.
A `Layout` is a shape plus a stride, with contiguity checks.

### Compilation pipeline

`Tracer` records into a `TIRGraph`; `ir/TIRToLIRLowerer` lowers TIR to LIR; `ir/lir/LIRStandardPipeline` runs the optimisation passes.
The active backend's compute engine compiles or executes the LIR: `LIRKernelCompiler` on panama, `CKernelCompiler` on the C backend, source generation on the GPU backends.
`TIRInterpreter` and `LIRInterpreter` are the interpreted fallbacks.
Compiled kernels are cached under the path `KernelCachePaths` resolves.

## Key files

- `jota-tensor/.../tensor/Tensor.java`, `TensorOps.java`: the tensor interface and its operations.
- `jota-tensor/.../tensor/Tracer.java`, `ScheduledExecutor.java`: tracing and execution.
- `jota-tensor/.../ir/TIRToLIRLowerer.java`, `ir/lir/LIRStandardPipeline.java`: lowering and passes.
- `jota-tensor/.../runtime/NativeLibraryLoader.java`: how a backend's native library is found and loaded.
- `jota-memory/.../memory/MemoryView.java`: the memory abstraction.
- `jota-core/.../impl/ShapeImpl.java`: the shape implementation, internal.

## Conventions

- Formatting is decided by Spotless; the build fails on unformatted code.
- Every backend must handle strided operands on-device; bailing out to the CPU breaks the layout-contract tests.
- A change to a backend runs the default suite and that backend's profile suite.
- Long Markdown files keep one sentence per line.
