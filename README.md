<h1 align="center"><a href="https://qxotic.ai">Quixotic AI</a></h1>

<p align="center"><strong>AI sovereignty for the JVM.</strong></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
  <a href="https://www.graalvm.org/latest/reference-manual/native-image/"><img src="https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F" alt="GraalVM Native Image"></a>
  <a href="https://x.com/qxoticai"><img src="https://img.shields.io/badge/-grey?logo=X" alt="Qxotic AI on X"></a>
  <a href="https://bsky.app/profile/qxotic.ai"><img src="https://img.shields.io/badge/-grey?logo=bluesky&logoColor=f5f5f5" alt="Qxotic AI on Bluesky"></a>
</p>

Run LLMs where your data already lives: inside the JVM. Quixotic is a complete, open stack for
local inference in Java — tokenization, model formats, tensor math, native quantized matmul, and a
full inference engine with chat, vision, embeddings, reranking and speech.

**No Python. No ONNX. No external services. Just jars.**

---

## The stack

| Module | What it is | One-liner |
|--------|-----------|-----------|
| [`jinfer`](./jinfer) | LLM inference engine | **Local LLM inference for the JVM** — chat, vision, embeddings, speech |
| [`jam`](./jam) | Native quantized matmul | **Just a matmul** — the fastest one on the JVM |
| [`jota`](./jota) | Tensor engine | **One tensor API, every backend** — Panama, C, CUDA, HIP, Metal, OpenCL, Mojo |
| [`toknroll`](./toknroll) | LLM tokenization | **Token-perfect.** Byte-exact parity with the reference tokenizers |
| [`gguf`](./gguf) | GGUF reader/writer | llama.cpp's model format, pure Java, zero dependencies |
| [`safetensors`](./safetensors) | Safetensors reader/writer | HuggingFace's model format, pure Java, zero dependencies |
| [`json`](./json) | JSON parser/printer | **~10 KB, zero dependencies, reflection-free** — Jackson not required |

---

## Why Quixotic

- **In-process, by design.** Models load, tokenize and generate inside your JVM. No sidecar
  servers, no IPC, no Python runtime to babysit.
- **Write once, accelerate everywhere.** A single tensor API across CPU and GPU backends. Switch
  backends with one line — or none: discovery is automatic.
- **Fast where it counts.** Hand-tuned SIMD kernels (SSE3 → AVX-512-VNNI, NEON → I8MM, Apple
  Metal) that match — and often beat — llama.cpp at equal ISA.
- **Native-image first.** Every module is GraalVM-ready out of the box: small footprint,
  millisecond startup, single self-contained binary.
- **OpenAI-compatible.** Point your existing tools at the bundled server; speak LangChain4j or
  Spring AI from application code.

---

## Quick start

Talk to a model in one file, no build required — [JBang](https://www.jbang.dev/) resolves
everything:

```bash
cd jinfer/examples/scripts
jbang Chat.java "Invent a tiny language for talking to houseplants."   # streaming chat
jbang Json.java "Ada Lovelace, born 1815 in London."                   # schema-perfect JSON
jbang Narrate.java photo.jpg                                          # vision + speech
```

Models download once from Hugging Face and stay in the local cache. Or drop
[`jinfer`](./jinfer) into your application with two dependencies and a builder — see the
[jinfer README](./jinfer/README.md).

---

## What runs where

jinfer dispatches by model architecture; add the matching artifact (or `jinfer-models-all`) and
the provider is discovered automatically.

| Family | Capability | Artifact |
|--------|-----------|----------|
| Gemma 4 | chat, vision, audio, MTP | `jinfer-gemma4` |
| Qwen 3 | chat, embeddings, reranking | `jinfer-qwen3` |
| Qwen 3.5 / 3.8 | chat, vision, MTP | `jinfer-qwen35` |
| LFM 2.5 | chat, embeddings, ColBERT reranking, vision | `jinfer-lfm2` |
| Llama family | chat (Llama, Ministral, MiniCPM, SmolLM, Granite) | `jinfer-llama` |
| gpt-oss / Nemotron-H / Maple | chat | `jinfer-gptoss`, `jinfer-nemotronh`, `jinfer-maple` |
| Inflect | speech synthesis | `jinfer-inflect2` |

---

## Building

Java 25, one Maven reactor, one entry point:

```bash
make help        # every build entry point
make jar         # the jinfer CLI jar
make native      # a GraalVM native binary (jinfer/jinfer)
make test        # the full suite
```

The Makefiles wrap the Maven reactor — use it directly for finer selection:

```bash
mvn -pl jinfer/jinfer-cli -am package -DskipTests   # the CLI and everything it needs
mvn package                                          # everything
```

`jota` is dependency-closed (`mvn -f jota/pom.xml` works standalone); the other subtrees build
from the root. Details live in each module's README.

---

## License

Apache 2.0
