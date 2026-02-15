# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Jota (JVM Open Tensor Algebra) is a tensor algebra library with lazy evaluation, kernel compilation, and multi-device support. Part of the llm4j multi-module Maven project.

## Build Commands

```bash
# Always use mvnd (instead of mvn) that keeps a warmed-up daemon to speedup the compile/test/run loop
# From jota directory
mvnd clean compile        # Build
mvnd test                 # Run all tests
mvnd test -Dtest=TestClass           # Run specific test class
mvnd test -Dtest=TestClass#testMethod  # Run specific test method
mvnd spotless:check       # Check formatting
mvnd spotless:apply       # Apply formatting (Google Java Format, AOSP style)
```

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
