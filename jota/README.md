# jota

[![Java](https://img.shields.io/badge/Java-25+-blue)](https://openjdk.org/projects/jdk/25/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)

**One tensor API, every backend.** A tensor engine for the JVM, inspired by tinygrad and JAX —
with first-class GraalVM Native Image support.

Write tensor code once. Run it on Panama, C, CUDA, HIP, Metal, OpenCL or Mojo by adding a jar —
backends self-register, no launch flags, no `-Djava.library.path`.

## Why jota

- **Lazy, traceable tensors.** Jota traces re-usable computation graphs with dynamic inputs and
  compiles them into kernels.
- **Strict by default.** No magic auto-broadcasting, no lossy type promotions. Operations say
  what they do.
- **Simple IRs.** MLIR-like, no data-dependent control flow: **TIR** (tensor IR) for high-level
  ops, **LIR** (loop IR) for optimization, scheduling and kernel generation.
- **Native-image first.** `jota-graal` makes every backend except Panama compile to native.
- **Escape hatches included.** The low-level API — memory, allocators, nested layouts (CuTe-style)
  — is right there when you need it, and custom kernels plug in for the hot paths.

## Backends

| Backend | Artifact | Notes |
|---------|----------|-------|
| Java (Panama) | `jota-backend-panama` | default on the JVM; not Native Image compatible |
| C | `jota-backend-c` | default on Native Image |
| CUDA | `jota-backend-cuda` | NVIDIA, Linux/Windows |
| HIP | `jota-backend-hip` | AMD |
| Metal | `jota-backend-metal` | Apple |
| OpenCL | `jota-backend-opencl` | cross-platform GPU |
| Mojo | `jota-backend-mojo` | experimental |

Put the backend jar on the classpath and it becomes available on supported platforms —
native libraries are bundled and auto-extracted. Use `-Djava.library.path` only for custom
overrides. For GraalVM Native Image, add `jota-graal`.

## Usage

Choose the smallest API you need — each layer includes the previous ones transitively:

| Artifact | What you get |
|----------|--------------|
| `jota-core` | data types, devices, shapes, strides, layouts |
| `jota-memory` | `Memory`, `MemoryView`, allocators, access, transfers |
| `jota-tensor` | tensors, environments, runtimes, IR, kernel compilation |

```xml
<dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>jota-tensor</artifactId>
    <version>0.2.0</version>
</dependency>
```

## Quick example

[Mandelbrot.java](https://github.com/qxoticai/qxotic/blob/96abe2e3546ec133ffd2daa39a0303fbbe241912/examples/src/main/java/com/qxotic/jota/examples/demos/Mandelbrot.java#L83-L106)
traces a computation graph once and emits
[this C kernel](https://gist.github.com/mukel/beb94917ae62dd0791afc84abe6829e2):

<p align="center">
    <img width="400" height="300" alt="Mandelbrot rendered by a jota-generated kernel" src="https://github.com/user-attachments/assets/f27089ec-2d94-403e-ba35-6471d5ed7228" />
</p>

## Performance

Work in progress. A hand-tuned Llama port reaches ~90% of llama.cpp's inference throughput
(float32) and ~70% for prompt ingestion — custom kernels and pre-allocated memory, nothing exotic
like flash attention.

LLM inference has been optimized to the point where generic compiler output can't compete with
carefully fused kernels. Jota supports custom kernels for exactly those cases, and delivers decent
performance for everything else.

## GraalVM Native Image

`jota-graal` gives out-of-the-box Native Image support across the C/HIP/CUDA/Metal/OpenCL/Mojo
backends. Panama is intentionally excluded — it relies on runtime class loading and JIT.

At native runtime, `Device.NATIVE` resolves to an available `MemorySegment`-capable backend
(typically `Device.C`). Control registration when several backend jars are present:

```bash
-Djota.backends.include=c,opencl
-Djota.backends.exclude=hip,opencl
```

Tokens are provider ids; `exclude` wins over `include`. Requirements: the GraalVM toolchain and a
C compiler on `PATH` for kernel compilation.

## Development

```bash
mvn test               # core (Panama backend)
mvn -Pc test           # C
mvn -Pcuda test        # CUDA (NVIDIA)
mvn -Phip test         # HIP (AMD)
mvn -Pmetal test       # Metal (macOS)
mvn -Pall test         # every backend available on this machine
```

`mvnd` works everywhere `mvn` does.

## A note on how jota was built

This project was developed with AI pair programmers (Claude, Codex, Kimi). They type fast and make
excellent coding buddies — but "make PyTorch in Java, no mistakes" is still very far away. When
the tokens run out, the maintainer edits by hand: massaging the codebase, simplifying, and
deleting. Models produce code at great speed; keeping complexity at bay is still a human sport.

## License

Apache 2.0
