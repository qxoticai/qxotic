<h1 align="center"><a href="https://qxotic.ai">Quixotic AI</a></h1>

<p align="center"><strong>AI sovereignty for the JVM.</strong></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
  <a href="https://www.graalvm.org/latest/reference-manual/native-image/"><img src="https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F" alt="GraalVM Native Image"></a>
  <a href="https://x.com/qxoticai"><img src="https://img.shields.io/badge/-grey?logo=X" alt="Qxotic AI on X"></a>
  <a href="https://bsky.app/profile/qxotic.ai"><img src="https://img.shields.io/badge/-grey?logo=bluesky&logoColor=f5f5f5" alt="Qxotic AI on Bluesky"></a>
</p>

The JVM powers global finance, big data, and mission-critical infrastructure. Quixotic provides core building blocks for running LLM inference natively on the JVM, including model loading, tokenization, tensor operations, a native quantized matmul engine, and an LLM inference server with native-performance CPU/GPU backends where needed. No external services, no Python interop, no ONNX bridges.

---

## Capabilities

- **Write Once, Accelerate Everywhere** - A single Tensor API across Panama, C, CUDA, HIP, Metal, OpenCL, and Mojo. Switch backends with one line.
- **LLM Inference Engine** - Run 9+ model families with Vector API-accelerated kernels and an OpenAI-compatible server.
- **Native Quantized Matmul** - Hand-tuned SIMD kernels (x86 SSE3 through AVX-512, ARM NEON through i8mm, Apple Metal) in a single fat JAR.
- **GraalVM Native Image** - First-class support for small footprint and fast startup.
- **JVM-Native Architecture** - Built from first principles for the JVM. No Python dependencies, no external runtimes.
- **On-Device LLM Inference** - Run large language models locally with quantization and efficient memory management.
- **Vector Embeddings** - Fast vector operations for RAG pipelines and semantic search.

---

## Modules

| Module | Description                                               |
|--------|-----------------------------------------------------------|
| [`jota`](./jota) | Tensor engine with CPU/GPU backends                       |
| [`toknroll`](./toknroll) | Tiktoken-compatible, BPE and common LLM tokenizers        |
| [`jinfer`](./jinfer) | JVM (LLM) Inference Engine    |
| [`jam`](./jam) | **FAST** native quantized matrix multiplication for CPUs    |
| [`gguf`](./gguf) | Pure Java read/write for llama.cpp's GGUF model format    |
| [`safetensors`](./safetensors) | Pure Java read/write for HuggingFace's Safetensors format |

---

## What runs where

jinfer dispatches by GGUF architecture; add the matching artifact (or `jinfer-models-all`) and the
provider is discovered automatically.

| Family | Capability | Artifact |
|--------|-----------|----------|
| Gemma 4 | chat, vision, audio, MTP | `jinfer-gemma4` |
| Qwen 3 | chat, embeddings, reranking | `jinfer-qwen3` |
| Qwen 3.5 / 3.8 | chat, vision, MTP | `jinfer-qwen35` |
| LFM 2.5 | chat, embeddings, ColBERT reranking, vision | `jinfer-lfm2` |
| Llama family | chat (Llama, Ministral, MiniCPM, SmolLM, Granite) | `jinfer-llama` |
| gpt-oss / Nemotron-H / Maple | chat | `jinfer-gptoss`, `jinfer-nemotronh`, `jinfer-maple` |
| Inflect | speech synthesis | `jinfer-inflect2` |

Embeddings and reranking are exposed by `jinfer-lfm2` and `jinfer-qwen3`; multimodal models attach
their projector with `--mmproj <clip.gguf>`. The OpenAI-compatible server lives in `jinfer-server`;
`jinfer-chat`, `jinfer-cache` and `jinfer-hub` provide the conversation, prompt-cache and model-download
layers. For a single version for the whole release, import `jinfer-bom`.

---

## Building

`make help` lists the build's entry points (`test`, `jinfer-test`, `jota-test`, `jar`,
`native`, `format`, `clean`, ...); `make -C jinfer help` and `make -C jota help` list the
subtree-local ones. The Makefiles wrap the Maven reactor below - use it directly when you need
finer selection.

This is a single Maven reactor rooted at the repository root.
No subtree is dependency-closed - `jinfer` alone pulls in `gguf`, `json`, `toknroll` and four `jam` artifacts - so always build from the root and select what you want with `-pl … -am`:

```bash
mvnd -pl jinfer/jinfer-cli,jinfer/jinfer-bench -am package   # the jinfer CLI + benchmark
mvnd -pl jam/jam-vector -am verify                              # jam and its backend parity tests
mvnd package                                                    # everything
```

Add `-Pnative` to produce GraalVM native images (`jinfer`, `jinfer-bench`), and `-Pformat` to apply Spotless.

Running Maven inside a subdirectory (`mvn -f jinfer`) only works once the outer modules are already installed, because that reactor cannot see them.
`mvn install` from the root once, or just use `-pl … -am`.

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
