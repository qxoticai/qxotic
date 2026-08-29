# jota

**One tensor API, every backend.** A tensor engine for the JVM, inspired by tinygrad and JAX, with
first-class GraalVM Native Image support.

[![Java](https://img.shields.io/badge/Java-25+-blue)](https://openjdk.org/projects/jdk/25/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)

Write tensor code once. Run it on Panama, C, CUDA, HIP, Metal, OpenCL or Mojo by adding a jar.
Backends self-register, with no launch flags and no `-Djava.library.path`.

## One graph, one generated kernel

[Mandelbrot.java](https://github.com/qxoticai/qxotic/blob/96abe2e3546ec133ffd2daa39a0303fbbe241912/examples/src/main/java/com/qxotic/jota/examples/demos/Mandelbrot.java#L83-L106)
traces a computation graph once and emits
[this C kernel](https://gist.github.com/mukel/beb94917ae62dd0791afc84abe6829e2):

<p align="center">
    <img width="400" height="300" alt="Mandelbrot rendered by a jota-generated kernel" src="https://github.com/user-attachments/assets/f27089ec-2d94-403e-ba35-6471d5ed7228" />
</p>

## Why jota

- **Lazy, traceable tensors.** jota traces reusable computation graphs with dynamic inputs and
  compiles them into kernels.
- **Strict by default.** No magic auto-broadcasting, no lossy type promotions. Operations say what
  they do.
- **Simple IRs.** MLIR-like, with no data-dependent control flow. **TIR** (tensor IR) for
  high-level ops, **LIR** (loop IR) for optimization, scheduling and kernel generation.
- **Native-image first.** `jota-graal` makes every backend except Panama compile to native.
- **Escape hatches included.** The low-level API for memory, allocators and nested layouts
  (CuTe-style) is available when needed, and custom kernels plug in for the hot paths.

## Install

Pick the smallest API that fits. Each layer includes the previous ones transitively:

| Artifact | Contents |
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

Put the backend jar on the classpath and it becomes available on supported platforms. Native
libraries are bundled and auto-extracted. Use `-Djava.library.path` only for custom overrides. For
GraalVM Native Image, add `jota-graal`.

## Performance

Work in progress. A hand-tuned Llama port reaches about 90% of llama.cpp's inference throughput
(float32) and about 70% for prompt ingestion, using custom kernels and pre-allocated memory,
nothing exotic like flash attention.

LLM inference has been optimized to the point where generic compiler output cannot compete with
carefully fused kernels. jota supports custom kernels for exactly those cases, and delivers decent
performance for everything else.

## GraalVM Native Image

`jota-graal` gives out-of-the-box Native Image support across the C, HIP, CUDA, Metal, OpenCL and
Mojo backends. Panama is intentionally excluded, since it relies on runtime class loading and JIT.

At native runtime, `Device.NATIVE` resolves to an available `MemorySegment`-capable backend,
typically `Device.C`. Control registration when several backend jars are present:

```bash
-Djota.backends.include=c,opencl
-Djota.backends.exclude=hip,opencl
```

Tokens are provider ids, and `exclude` wins over `include`. Requires the GraalVM toolchain and a C
compiler on `PATH` for kernel compilation.

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

Part of [Quixotic](../README.md), an open stack for local AI on the JVM.

## A note on how jota was built

This project was developed with AI pair programmers (Claude, Codex, Kimi). They type fast and make
excellent coding buddies, but "make PyTorch in Java, no mistakes" is still very far away. When the
tokens run out, the maintainer edits by hand: massaging the codebase, simplifying, and deleting.
Models produce code at great speed. Keeping complexity at bay is still a human sport.

## License

Apache 2.0
