---
sidebar_position: 2
---

# jota

**One tensor API, every backend.** A tensor library with pluggable CPU/GPU backends and first-class GraalVM Native Image support. Write tensor code once, run it on Panama, C, CUDA, HIP, Metal, OpenCL or Mojo by adding a jar.

jota's IRs are simple and MLIR-like; no data-dependent control flow:

- **TIR** (Tensor IR): high-level tensor operations, no control flow
- **LIR** (Loop IR): explicit loops for optimization, scheduling, and kernel generation

## Backends

| Backend | Artifact | Notes |
|---------|----------|-------|
| Java | `jota-backend-panama` | default on JVM; not Native Image compatible |
| C | `jota-backend-c` | default on Native Image |
| OpenCL | `jota-backend-opencl` | cross-platform GPU |
| HIP | `jota-backend-hip` | AMD only |
| CUDA | `jota-backend-cuda` | NVIDIA, Linux/Windows |
| Metal | `jota-backend-metal` | Apple |
| Mojo | `jota-backend-mojo` | experimental |

The low-level API (memory, allocators, nested layouts) stays within reach when you need it. Tensors are lazy and opaque; jota traces computation graphs with dynamic inputs. Operations are strict: no implicit auto-broadcasting, no lossy type promotion.

## Usage

Pick the smallest API you need. Each layer includes the preceding layers transitively:

- `jota-core`: data types, devices, shapes, strides, layouts
- `jota-memory`: `Memory`, `MemoryView`, allocators, access, transfers
- `jota-tensor`: tensors, environments, runtimes, IR, kernel compilation

Memory-only applications need one dependency:

```xml
<dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>jota-memory</artifactId>
    <version>0.2.0</version>
</dependency>
```

Tensor applications use `jota-tensor`:

```xml
<dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>jota-tensor</artifactId>
    <version>0.2.0</version>
</dependency>
```

## Backend dependencies

Put the backend JAR on the classpath; it becomes available on supported platforms. **No `-Djava.library.path`**: native libraries are bundled and auto-extracted. Use `-Djava.library.path` only for custom overrides.

```xml
<!-- Java backend (not Native Image compatible) -->
<dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>jota-backend-panama</artifactId>
    <version>0.2.0</version>
</dependency>

<!-- C backend (CPU via Panama) -->
<dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>jota-backend-c</artifactId>
    <version>0.2.0</version>
</dependency>

<!-- GraalVM Native Image convenience -->
<dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>jota-graal</artifactId>
    <version>0.2.0</version>
</dependency>

<!-- AMD GPU -->
<dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>jota-backend-hip</artifactId>
    <version>0.2.0</version>
</dependency>

<!-- NVIDIA GPU (Linux/Windows) -->
<dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>jota-backend-cuda</artifactId>
    <version>0.2.0</version>
</dependency>

<!-- Apple GPU (macOS) -->
<dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>jota-backend-metal</artifactId>
    <version>0.2.0</version>
</dependency>

<!-- Cross-platform GPU -->
<dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>jota-backend-opencl</artifactId>
    <version>0.2.0</version>
</dependency>
```

## GraalVM Native Image

`jota-graal` gives out-of-the-box Native Image support across C/HIP/CUDA/Metal/OpenCL/Mojo. Panama is excluded because it depends on runtime class loading and JIT.

In Native Image, `Device.NATIVE` resolves to an available `MemorySegment`-capable backend (typically `Device.C`).

Control backend registration when backend jars are present:

```bash
-Djota.backends.include=c,opencl
-Djota.backends.exclude=hip,opencl
```

Tokens are provider ids: `hip`, `cuda`, `opencl`, `c`, `metal`, `panama`, `mojo`. If both `include` and `exclude` name the same backend, `exclude` wins.

Requirements: GraalVM Native Image toolchain, and a C compiler on PATH for kernel compilation.

## Example

jota tracing produces a reusable computation graph; the C backend emits a compact kernel. See [`Mandelbrot.java`](https://github.com/qxoticai/qxotic/blob/main/examples/src/main/java/com/qxotic/jota/examples/demos/Mandelbrot.java).

## Performance

Work in progress. Hand-tuned Llama reaches ~90% of llama.cpp inference throughput (float32) and ~70% for prompt ingestion, using custom kernels and pre-allocated memory, without flash attention.

LLM inference rewards carefully fused kernels over generic compiler output. jota supports custom kernels for those cases and is adequate for the rest.
