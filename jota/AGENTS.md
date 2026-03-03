# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Jota (JVM Open Tensor Algebra) is a tensor algebra library with lazy evaluation, kernel compilation, and multi-device support. Part of the llm4j multi-module Maven project.

## Build Commands

```bash
# Always use mvnd (never mvn) to keep a warmed daemon and faster edit/build/test loops.
# From jota/ directory

# Build
mvnd clean compile

# Test suites
mvnd test                 # Core suite (Panama/default runtime)
mvnd -Pc test             # C backend suite (runs regular tests against C runtime + C-specific tests)
mvnd -Phip test           # HIP backend suite (requires HIP/ROCm toolchain/runtime)
mvnd -Pmetal test         # Metal backend suite (runs on macOS; non-mac skips native/test steps)
mvnd -Phip,backend-matrix test   # Optional backend matrix mode for HIP profile

# Targeted tests
mvnd test -Dtest=TestClass
mvnd test -Dtest=TestClass#testMethod

# Formatting (always run before finishing changes)
mvnd spotless:check
mvnd spotless:apply
```

## Development Workflow (Quick Reference)

- Work from `jota/` root and always use `mvnd`.
- Fast compile loop: `mvnd clean compile` (first run) then `mvnd compile`.
- Run core tests during normal iteration: `mvnd test`.
- Validate C backend behavior with full C profile: `mvnd -Pc test`.
- Validate HIP backend behavior: `mvnd -Phip test` (with HIP toolchain/runtime available).
- Validate Metal backend behavior: `mvnd -Pmetal test` (native/tests execute on macOS; non-mac skips).
- Use `backend-matrix` only when explicitly needed; default profiles run deterministic single-backend tests.
- Run a focused test while iterating: `mvnd test -Dtest=ClassName` or `mvnd test -Dtest=ClassName#methodName`.
- Before finishing any change, run formatting: `mvnd spotless:apply` (then optionally `mvnd spotless:check`).
- If touching backend-specific code, run both core (`mvnd test`) and the corresponding backend profile suite.
- Keep Panama as the default/core runtime path; treat C/HIP/Metal as profile-scoped validation paths.

**Java Version:** 25 (source/target)

## Architecture

### Tensor Evaluation Model

**Lazy Tensors** - Expression tracing for kernel compilation:
- `Tracer.trace(input, fn)` records operations into an `ExpressionGraph`
- Graph nodes: `InputNode`, `UnaryOpNode`, `BinaryOpNode`, etc.
- Materialization triggers compilation/execution via `tensor.materialize()`

**Eager Tensors** - Immediate evaluation:
- `EagerTensorOps` executes operations directly on `MemoryView`
- Default when not in traced context

**Optimizing Call Sites** - Kernel specialization:
```java
OptimizingCallSite site = Jota.createOptimizingCallSite(t -> t.relu());
Tensor result = site.apply(inputTensor); // Compiles on first call
```

### Memory Abstraction Layers

1. `MemoryDomain<B>` - Device context (allocator, access, operations)
2. `Memory<B>` - Backing buffer (ByteBuffer, MemorySegment, arrays, etc.)
3. `MemoryView<B>` - View with Layout (shape + stride) over Memory
4. `MemoryAccess<B>` - Element read/write
5. `MemoryOperations<B>` - Bulk copy/broadcast/reshape

**Memory backends:** Java heap arrays, ByteBuffer, sun.misc.Unsafe (off-heap), Panama Foreign Memory API (MemorySegment)

### Shape System (CuTe-inspired)

- Immutable shapes with nested structure: `Shape.of(2, Shape.of(3, 4), 5)`
- **Wrap-around indexing convention:** `_prefix` = wrap wrt input, `suffix_` = wrap wrt output
- Layout = Shape + Stride with contiguity checks

### Kernel Compilation Pipeline

`JavaKernelCompiler`: ExpressionGraph → Java source → compile via javax.tools → load → cache

Fallback: `KernelInterpreter` for interpreted execution

## Key Source Files

- `tensor/Tensor.java` - Core tensor interface
- `tensor/Tracer.java` - Expression tracing
- `tensor/JavaKernelCompiler.java` - JIT kernel generation
- `memory/MemoryView.java` - Memory abstraction
- `impl/ShapeImpl.java` - Shape implementation
- `tensor/TensorOps.java` - Operations interface

## Code Style

- Google Java Format (AOSP style) via Spotless
- UNIX line endings, trailing whitespace trimmed
