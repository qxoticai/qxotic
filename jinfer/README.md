# jinfer

[![Java 21+](https://img.shields.io/badge/Java-21%2B-007396?logo=java&logoColor=white)](https://openjdk.org/projects/jdk/21/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache)](LICENSE)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)

**LLM inference engine for the JVM.** Model-agnostic API. Vector-accelerated kernels.
[JAM](../jam)-backed prefill. OpenAI-compatible server. Pure Java, zero Python.

**9 model families · Vector API (AVX2/AVX-512/NEON) · JAM GEMM fallback · GGUF-native · GraalVM Native Image**

---

## Why jinfer?

- **Model-agnostic.** A single `Model.forward()/generate()` interface. Each architecture is one file.
  Adding a new model means implementing one class. Sampler, tokenizer, chat templates, and server
  are shared across all models.
- **Vector API kernels.** F16 decode, Q8_0 GEMM, flash attention, and RoPE use the Java Vector API
  for portable SIMD. Runs fast on x86 and ARM.
- **JAM prefill backend.** Drop `jam.jar` on the classpath and Q8_0 GEMM delegates to hand-tuned
  native assembly (SSE3 through AVX-512 VNNI, NEON through i8mm, Apple Metal). No code changes.
- **OpenAI-compatible server.** `/v1/chat/completions`, `/v1/completions`, `/v1/models`,
  `/v1/responses` with streaming, function calling (tools), and structured output.
- **GraalVM Native Image.** `make native` produces a self-contained binary. AOT-preload a GGUF
  into the image for instant time-to-first-token.
- **Zero Python.** Pure Java + optional native (JAM). No ONNX, no transformers, no llama.cpp.

---

## Quick start

```java
// High-level API
var engine = Engine.loadGGUF("model.gguf");
engine.chat(List.of(Engine.message("user", "Tell me a joke")),
            LLMOptions.builder().maxTokens(256).build(),
            token -> System.out.print(token));

// Low-level API
var model = Model.loadGGUF("model.gguf");         // auto-detects architecture
var state = model.createInferenceState(model.tokenize("Hello!"));
while (model.sample(state) != model.eosTokenId())
    model.forward(state, nextToken);
```

### CLI

```bash
mvn package
java --enable-preview --add-modules jdk.incubator.vector \
  -jar target/jinfer.jar --model ./model.gguf --chat
java -jar target/jinfer.jar --model ./model.gguf --server --port 17325
```

Server endpoints:

| `GET /v1/models` | `POST /v1/chat/completions` | `POST /v1/completions` |
| `GET /health` | `POST /v1/responses` | `GET /metrics` |

Streaming, `temperature`, `top_p`, `seed`, `max_tokens`, and `stop` supported.
Function calling with `tools`/`tool_choice` (auto, none, required, named).

---

## Models

Auto-detected from GGUF metadata. Each architecture is a single-file `Model` implementation.

| Model | Architecture | Variants |
|---|---|---|
| **Gemma 4** | Google Gemma 4 | E2B, E4B, A4B (MoE) |
| **Qwen 3.5** | Qwen 3.5 | Dense, MoE |
| **Nemotron 3** | NVIDIA Nemotron | Hybrid Mamba2 + Attention + MoE |
| **Llama 3** | Meta Llama 3.x | Dense |
| **Ministral 3** | Mistral Ministral | Dense (YaRN RoPE) |
| **gpt-oss** | OpenAI gpt-oss | MXFP4 MoE |
| **LFM 2.5** | Liquid AI LFM 2.5 | Dense, MoE (short-convolution) |
| **MiniCPM** | MiniCPM | Dense (Llama + 3 scalars) |
| **IBM Granite 4.1** | Granite | Dense (Llama + 4 scalars) |

Supported GGUF dtypes: `F16`, `BF16`, `F32`, `Q4_0`, `Q4_1`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`.

---

## Performance

**Q8_0 GEMM tile** (`-Djinfer.Q8_0GemmTile`): `auto` picks `4x4` on AVX-512 with capable compilers,
`avx256` on AVX2, `neon` on ARM. Override for your environment.

**JAM backend:** drop `jam.jar` on the classpath to route Q8_0 GEMM through native assembly.
No config, no API changes.

**Flash attention:** on by default (`-Djinfer.flashAttention=false` to disable). Requires inline
hints in `$JAVA_FLAGS` (the Makefile and native image handle these).

**GraalVM 25+** recommended for best JIT performance and Native Image Vector API support.

---

## GraalVM Native Image

```bash
make native                              # self-contained binary
PRELOAD_GGUF=model.gguf make native      # embed model, instant TTFT
./jinfer --model ./model.gguf --chat
```

---

## Build

Java 25 required. Uses `--enable-preview` for `MemorySegment` mmap.

```bash
mvn package                    # -> jinfer-server/target/jinfer.jar
make jar                       # alternative via Makefile
```

---

## What jinfer does NOT do

- **No training / fine-tuning.** Inference only.
- **No quantization.** Reads quantized GGUF files; doesn't create them.
- **No GPU scheduling.** JAM Metal backend handles Apple GPU matmul. No CUDA/Metal graph engine.

---

## License

Apache 2.0
