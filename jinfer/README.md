# jinfer

[![Java 21+](https://img.shields.io/badge/Java-21%2B-007396?logo=java&logoColor=white)](https://openjdk.org/projects/jdk/21/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache)](LICENSE)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)

**JVM (LLM) Inference Engine.** Model-agnostic, vector-accelerated, [JAM](../jam)-backed prefill,
with an OpenAI-compatible server. Pure Java, no Python, no llama.cpp.

**9 model families · Vector API (AVX2 / AVX-512 / NEON) · JAM GEMM · GGUF-native · GraalVM Native Image**

---

## Why jinfer?

- **Model-agnostic.** One `Model.forward()/generate()` interface, and each architecture is a single file.
  Adding a model is writing one class — the sampler, tokenizer, chat templates, and server come for free.
- **Vector API kernels.** F16 decode, Q8_0 GEMM, flash attention, and RoPE run on the Java Vector API —
  portable SIMD that's fast on both x86 and ARM.
- **JAM prefill.** Drop `jam.jar` on the classpath and Q8_0 GEMM quietly routes through hand-tuned native
  kernels (SSE3 → AVX-512 VNNI, NEON → i8mm, Apple Metal). No code changes, no config.
- **OpenAI-compatible server.** `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models` —
  with streaming, tool calls, and structured output.
- **GraalVM Native Image.** `make native` gives you one self-contained binary; preload a GGUF into the
  image for an instant first token.
- **Zero Python.** Pure Java plus an optional native lib. No ONNX, no transformers, no llama.cpp.

---

## Quick start

```java
// High-level
var engine = Engine.loadGGUF("model.gguf");
engine.chat(List.of(Engine.message("user", "Tell me a joke")),
            LLMOptions.builder().maxTokens(256).build(),
            token -> System.out.print(token));

// Low-level
var model = Model.loadGGUF("model.gguf");        // architecture auto-detected
var state = model.createInferenceState(model.tokenize("Hello!"));
int token;
while ((token = model.sample(state)) != model.eosTokenId()) {
    System.out.print(model.detokenize(token));
    model.forward(state, token);
}
```

### CLI

```bash
mvn package
java --enable-preview --add-modules jdk.incubator.vector \
  -jar target/jinfer.jar --model ./model.gguf --chat
java -jar target/jinfer.jar --model ./model.gguf --server --port 17325
```

For full speed add `--enable-native-access=ALL-UNNAMED` plus the inline hints from
[Performance](#performance) — the Makefile and the native image set these for you.

Server: streaming, `temperature`, `top_p`, `seed`, `max_tokens`, `stop`, and function calling
(`tools` / `tool_choice`: auto, none, required, named). Endpoints: `/v1/models`, `/v1/chat/completions`,
`/v1/completions`, `/v1/responses`, `/health`, `/metrics`.

---

## Models

Auto-detected from GGUF metadata; each architecture is a single-file `Model`.

| Model | Architecture | Variants | Key features |
|---|---|---|---|
| **Gemma 4** | Google Gemma 4 | E2B, E4B, A4B (MoE) | Per-layer embeddings, sliding-window attention, logit soft-capping |
| **Qwen 3.5** | Qwen 3.5 | Dense, MoE | GQA with QK normalization |
| **Nemotron 3** | NVIDIA Nemotron | Hybrid Mamba2 + Attention + MoE | Hybrid SSM-transformer |
| **Llama 3** | Meta Llama 3.x | Dense | Standard Llama transformer, llama3 RoPE scaling |
| **Ministral 3** | Mistral Ministral | Dense | YaRN RoPE, attention-temperature scaling, sliding window |
| **gpt-oss** | OpenAI gpt-oss | MXFP4 MoE | MXFP4-quantized expert weights |
| **LFM 2.5** | Liquid AI LFM 2.5 | Dense, MoE | Short-convolution layers |
| **MiniCPM** | MiniCPM | Dense | Llama architecture + 3 extra scalars |
| **IBM Granite 4.1** | Granite | Dense | Llama architecture + custom QK attention scale |

Supported GGUF dtypes: `F16` `BF16` `F32` `Q4_0` `Q4_1` `Q4_K` `Q5_K` `Q6_K` `Q8_0`.

---

## Performance

- **Q8_0 GEMM tile** (`-Djinfer.Q8_0GemmTile`): `auto` picks `4x4` on AVX-512 (with a capable compiler),
  `avx256` on AVX2, `neon` on ARM. Override if you know better.
- **JAM backend:** `jam.jar` on the classpath routes Q8_0 GEMM through native assembly — no config, no
  API change.
- **Flash attention:** on by default (`-Djinfer.flashAttention=false` to turn it off); wants the inline
  hints in `$JAVA_FLAGS`.
- **GraalVM 25+** recommended — best JIT and Native Image Vector API support.

---

## GraalVM Native Image

```bash
make native                            # self-contained binary
PRELOAD_GGUF=model.gguf make native    # embed the model, instant TTFT
./jinfer --model ./model.gguf --chat
```

---

## Build

Java 25 (`--enable-preview` for `MemorySegment` mmap).

```bash
mvn package      # -> jinfer-server/target/jinfer.jar
make jar         # same thing, via the Makefile
```

---

## What jinfer doesn't do

- **No training or fine-tuning** — inference only.
- **No quantization** — it reads quantized GGUF, doesn't create it.
- **No GPU scheduling** — Apple GPU matmul goes through JAM's Metal backend; there's no CUDA/Metal graph engine.

---

## License

Apache 2.0
