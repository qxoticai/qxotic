# Jota

[![Java](https://img.shields.io/badge/Java-25+-blue)](https://openjdk.org/projects/jdk/25/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)

Jota is a tensor library for the JVM, heavily inspired by tinygrad and JAX.
Supports multiple backends with first-class GraalVM's Native Image support.

The Jota IRs are intentionally simple, MLIR-like, with no data-dependent control flow allowed.
- **TIR** Tensor IR: high-level tensor operations, no control flow
- **LIR** Loop IR: Low-level IR with explicit loops used for optimization, scheduling and kernel generation

## Backends

- [Java](https://github.com/qxoticai/qxotic/tree/main/jota/jota-backend-panama) (default on JVM)
- [C](https://github.com/qxoticai/qxotic/tree/main/jota/jota-backend-c) (default on Native Image)
- [OpenCL](https://github.com/qxoticai/qxotic/tree/main/jota/jota-backend-opencl)
- [HIP](https://github.com/qxoticai/qxotic/tree/main/jota/jota-backend-hip) (AMD only)
- [CUDA](https://github.com/qxoticai/qxotic/tree/main/jota/jota-backend-cuda) (NVIDIA, Linux/Windows)
- [Metal](https://github.com/qxoticai/qxotic/tree/main/jota/jota-backend-metal) (Apple)
- [Mojo](https://github.com/qxoticai/qxotic/tree/main/jota/jota-backend-mojo) (**experimental**)

## Overview

Underneath the Tensor API, the low-level API is also accessible e.g. memory, allocators and nested layouts (like CuTe).  
Tensors are lazy and opaque. Jota relies on tracing to capture re-usable computation graphs with dynamic inputs.  
Tensor operations are strict and explicit, not magic auto-broadcasting, or lossy type promotions.

The Java language is too rigid for implementing a usable DSL to generate kernels, but a nicer API could be implemented in Kotlin or Scala.

## Quick Example

[Mandelbrot.java](https://github.com/qxoticai/qxotic/blob/96abe2e3546ec133ffd2daa39a0303fbbe241912/examples/src/main/java/com/qxotic/jota/examples/demos/Mandelbrot.java#L83-L106)
produces the following C kernel https://gist.github.com/mukel/beb94917ae62dd0791afc84abe6829e2

<p align="center">
    <img width="400" height="300" alt="Image" src="https://github.com/user-attachments/assets/f27089ec-2d94-403e-ba35-6471d5ed7228" />
</p>

## Performance

This is a work in progress, hand-tuned Llama can reach ~90% inference throughput of llama.cpp (float32)
and around ~70% for prompt ingestion; by using custom kernels, pre-allocated memory, but nothing too specialized e.g. no flash attention.

LLM inference has been optimized A LOT, to the point that traditional compiler magic cannot compete with the carefully crafted, fused kernels.  
Jota supports custom kernels for these cases and provides decent performance for the rest.

## Usage

```xml
<dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>jota-core</artifactId>
    <version>0.1.0</version>
</dependency>
```

### Backend Dependencies

Just include the backend `.jar` in the classpath and it will be automatically available on platforms that support it.  
**No `-Djava.library.path` required.** Native libraries are bundled in the JAR and auto-extracted at runtime. Use `-Djava.library.path` only for custom native library overrides.


```xml
<!-- Java backend (not compatible with Native Image) -->
<dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>jota-backend-panama</artifactId>
    <version>0.1.0</version>
</dependency>

<!-- C backend (CPU via Panama) -->
<dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>jota-backend-c</artifactId>
    <version>0.1.0</version>
</dependency>

<!-- GraalVM Native Image convenience dependency -->
<dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>jota-graal</artifactId>
    <version>0.1.0</version>
</dependency>

<!-- AMD GPU -->
<dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>jota-backend-hip</artifactId>
    <version>0.1.0</version>
</dependency>

<!-- NVIDIA GPU (Linux/Windows) -->
<dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>jota-backend-cuda</artifactId>
    <version>0.1.0</version>
</dependency>

<!-- Apple GPU (macOS) -->
<dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>jota-backend-metal</artifactId>
    <version>0.1.0</version>
</dependency>

<!-- Cross-platform GPU -->
<dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>jota-backend-opencl</artifactId>
    <version>0.1.0</version>
</dependency>
```

## GraalVM Native Image

Use `jota-graal` for out-of-the-box Native Image support across C/HIP/CUDA/Metal/OpenCL/Mojo backends. Panama is intentionally excluded from Native Image support because it relies on runtime class loading/JIT behavior.

In Native Image runtime, `Device.NATIVE` resolves to an available backend that supports `MemorySegment` (typically `Device.C`).

You can control backend registration even when backend jars are present on classpath:

```bash
-Djota.backends.include=c,opencl
-Djota.backends.exclude=hip,opencl
```

Accepted tokens are provider id (`hip`, `cuda`, `opencl`, `c`, `metal`, `panama`, `mojo`).
If both include and exclude mention the same backend, exclude takes priority.

Requirements:

- GraalVM Native Image toolchain
- C compiler available on PATH (required for kernel compilation)

## Development

```bash
mvnd compile
mvnd spotless:apply

mvnd test

# Core (Panama)
mvnd test

# Backends
mvnd -Pc test        # C
mvnd -Phip test      # HIP (AMD only)
mvnd -Pcuda test     # CUDA (NVIDIA, Linux/Windows; ignored on unsupported platforms)
mvnd -Pmojo test     # Mojo (AMD only)
mvnd -Popencl test   # Cross-platform
mvnd -Pmetal test    # Apple (macOS only)

mvnd -Pall test      # Run unit tests on all available backends
```

## AI Use Disclaimer
This project was developed with the help of AI, Claude, Codex and Kimi.
The LLMs vastly automate the typing part and serve as an excellent coding buddy, but "make PyTorch in Java, no mistakes" is still very far.  

As a solo developer, I've had a great time working on this.

I have the cheap plans for Claude, Codex and Kimi ~20 USD each, sometimes I ran out of tokens and I go and edit the code myself.  
This is a great opportunity to massage the codebase, simplify and delete code as much as I can, the models can produce code at great speed, but are not as good keeping complexity at bay.
