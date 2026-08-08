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
java -jar target/jinfer.jar --model ./model.gguf --server --port 17341
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
| **Qwen 3.5** | Qwen 3.5 | Dense, MoE | Hybrid gated-delta-net + periodic full attention |
| **Nemotron 3** | NVIDIA Nemotron | Hybrid Mamba2 + Attention + MoE | Hybrid SSM-transformer |
| **Llama 3** | Meta Llama 3.x | Dense | Standard Llama transformer, llama3 RoPE scaling |
| **Ministral 3** | Mistral Ministral | Dense | YaRN RoPE, attention-temperature scaling, sliding window |
| **gpt-oss** | OpenAI gpt-oss | MXFP4 MoE | MXFP4-quantized expert weights |
| **LFM 2.5** | Liquid AI LFM 2.5 | Dense, MoE | Short-convolution layers |
| **MiniCPM** | MiniCPM | Dense | Llama architecture + 3 extra scalars |
| **IBM Granite 4.1** | Granite | Dense | Llama architecture + custom QK attention scale |

Supported GGUF dtypes: `F16` `BF16` `F32` `Q4_0` `Q4_1` `Q4_K` `Q5_K` `Q6_K` `Q8_0`.

---

## Models from the hub

Anywhere jinfer takes a model file, it also takes a **model ref** - or a pasted browser URL, which normalizes to the same ref.
Downloads are parallel, resumable, and sha256-verified; a warm cache costs zero network requests.

```
[https://] host / owner/repo [@revision] [/path] [:quant]

hf.co/unsloth/gemma-4-E2B-it-GGUF              Q4_K_M at the repository root
hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0         that quant
hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf   that exact file
hf.co/ggml-org/models@a1b2c3d/bert-bge-small   at a pinned revision
modelscope.cn/Qwen/Qwen3-0.6B-GGUF:Q8_0        the other host
```

A ref names its host, so remote and local are told apart by a closed table, never by probing the filesystem - the same string means the same bytes on every machine.
The cache is the ref (`<cache>/hf.co/unsloth/...`), shared both ways with the HuggingFace hub cache: files fetched by `hf download` or `llama-server -hf` are found, not re-downloaded, and jinfer's own `hf.co` downloads land there for other tools to find.
`JINFER_MODELS` moves the cache (a bigger disk); `HF_TOKEN` (or a prior `hf auth login`) unlocks gated repos.

```bash
java -jar jinfer.jar --model hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0 --server
java -jar jinfer.jar pull hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0    # download, print the path
java -jar jinfer.jar list                                           # what is cached, as refs
```

### Dockerfiles and CI

`pull` exists so the download can be a **cached image layer**: bake the model once, rebuild the app freely.

```dockerfile
RUN java -jar jinfer.jar pull hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0
COPY app.jar /app/
```

For integration tests, the blessed smoke-test model is tiny and downloads in seconds:

```bash
java -jar jinfer.jar pull hf.co/ggml-org/stories15M_MOE:Q8_0
```

### Offline and air-gapped

`JINFER_OFFLINE=1` (or `-Djinfer.offline`) forbids the network entirely: a cached ref resolves without a single request, an uncached one fails fast and names what is missing.
No surprise egress, ever - resolution happens before a model loads, and nothing on an inference path touches the network at all.

The air-gap workflow is the cache being plain files.
An explicit `JINFER_MODELS` means "my cache lives here, all of it" - it opts out of the shared HuggingFace-cache layout, so the one directory is self-contained and shippable:

```bash
# on a connected machine
JINFER_MODELS=/models/jinfer java -jar jinfer.jar pull hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0
# ship the directory (rsync, USB, artifact store)
rsync -a /models/jinfer/ airgapped:/models/jinfer/
# on the air-gapped machine: same ref, zero network
JINFER_MODELS=/models/jinfer JINFER_OFFLINE=1 \
  java -jar jinfer.jar --model hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0 --server
```

Pin refs to a `@revision` (a commit is immutable) and the same ref is byte-reproducible everywhere, forever.
Behind a mirror or a blocked route, `HF_ENDPOINT` redirects the fetch while the ref and the cache entry stay canonical.

---

## Performance

- **Q8_0 GEMM tile** (`-Djinfer.Q8_0GemmTile`): `auto` picks `4x4` on AVX-512 (with a capable compiler),
  `avx256` on AVX2, `neon` on ARM. Override if you know better.
- **JAM backend:** `jam.jar` on the classpath routes Q8_0 GEMM through native assembly — no config, no
  API change.
- **Flash attention:** always on; wants the inline hints in `$JAVA_FLAGS`.
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
mvn package      # -> jinfer-cli/target/jinfer.jar
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
