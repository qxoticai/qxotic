# [Quixotic AI](https://qxotic.ai)

[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)

**AI sovereignty for the JVM.**

The JVM powers global finance, big data, and mission-critical infrastructure. Quixotic provides core building blocks for running LLM inference natively on the JVM, model loading, tokenization, and tensor operations, with native-performance CPU/GPU backends where needed. No external services, no Python interop, no ONNX bridges.

---

## Capabilities

- **Write Once, Accelerate Everywhere** - A single Tensor API across Panama, C, CUDA, HIP, Metal, OpenCL, and Mojo. Switch backends with one line.
- **GraalVM Native Image** - First-class support for small footprint and fast startup.
- **JVM-Native Architecture** - Built from first principles for the JVM. No Python dependencies, no external runtimes.
- **On-Device LLM Inference** - Run large language models locally with quantization and efficient memory management.
- **Vector Embeddings** - Fast vector operations for RAG pipelines and semantic search.

---

## Modules

| Module | Description                                               |
|--------|-----------------------------------------------------------|
| [`jota`](./jota) | Tensor engine with CPU/GPU backends                       |
| [`tokenizers`](./tokenizers) | TikToken-compatible, BPE and common LLM tokenizers        |
| [`gguf`](./gguf) | Pure Java read/write for llama.cpp's GGUF model format    |
| [`safetensors`](./safetensors) | Pure Java read/write for HuggingFace's Safetensors format |

---

## Jota Backends

The tensor engine supports multiple backends, packaged as separate artifacts:

| Backend | Artifact | Runtime Dependencies                  |
|---------|----------|---------------------------------------|
| Java (Panama) | [`jota-backend-panama`](./jota/jota-backend-panama) | Any JVM (not Native Image compatible) |
| C | [`jota-backend-c`](./jota/jota-backend-c) | `gcc` or `clang`                      |
| CUDA | [`jota-backend-cuda`](./jota/jota-backend-cuda) | NVIDIA driver + `nvcc`                |
| HIP | [`jota-backend-hip`](./jota/jota-backend-hip) | ROCm + `hipcc`                        |
| Metal | [`jota-backend-metal`](./jota/jota-backend-metal) | Xcode CLI tools (`xcrun`)             |
| OpenCL | [`jota-backend-opencl`](./jota/jota-backend-opencl) | OpenCL ICD runtime                    |
| Mojo | [`jota-backend-mojo`](./jota/jota-backend-mojo) | `mojo` CLI + ROCm runtime (experimental) |

Just include the backend JAR on the classpath, it becomes available automatically. No `-Djava.library.path` required.

For GraalVM Native Image, add `jota-graal`.