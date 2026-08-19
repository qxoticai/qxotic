# jinfer

[![Java 25](https://img.shields.io/badge/Java-25-007396?logo=java&logoColor=white)](https://openjdk.org/projects/jdk/25/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache)](LICENSE)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)

Pure-Java, in-process GGUF inference built around `MemoryView`: chat, embeddings, reranking,
speech, multimodal projection, prompt caching and an OpenAI-compatible server. Vector API kernels
provide the portable floor; [JAM](../jam) accelerates matrix multiplication when available.

## Quick start

Build the CLI:

```bash
mvn -pl jinfer-cli -am package -DskipTests

java --enable-preview \
  --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -jar jinfer-cli/target/jinfer.jar \
  --model hf.co/LiquidAI/LFM2.5-350M-GGUF:LFM2.5-350M-Q8_0.gguf \
  --chat
```

One-shot generation replaces `--chat` with `--prompt "..."`. `--context-capacity 0` uses the
model's declared maximum; negative values and capacities above the model maximum are rejected.

The smallest application-facing API is usually one of the framework adapters:

```java
try (var model = JinferChatModel.builder()
        .model("hf.co/LiquidAI/LFM2.5-350M-GGUF:LFM2.5-350M-Q8_0.gguf")
        .build()) {
    System.out.println(model.chat("Tell me a short joke."));
}
```

Use artifact `com.qxotic:jinfer-langchain4j:0.1.0`; Spring AI users use
`com.qxotic:jinfer-spring-ai:0.1.0`. See the [LangChain4j guide](jinfer-langchain4j/README.md),
[Spring AI guide](jinfer-spring-ai/README.md), and [runnable jbang examples](examples/scripts/README.md).

## OpenAI-compatible server

```bash
java --enable-preview \
  --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -jar jinfer-cli/target/jinfer.jar \
  --model hf.co/LiquidAI/LFM2.5-350M-GGUF:LFM2.5-350M-Q8_0.gguf \
  --server --port 17341
```

The server implements `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`,
`/v1/tokenize`, `/v1/detokenize`, `/health`, and Prometheus `/metrics`. Chat supports streaming,
tools/tool choice, structured output, reasoning, stop strings, deterministic seeds and
multimodal content where the loaded model has a projector.

Loopback is the safe default. Binding to a non-loopback host requires `--api-key`; queue, body,
thread, generation and stalled-write limits have explicit CLI flags. Run `--help` for the complete
contract.

## Models and capabilities

Architecture dispatch comes from the model providers on the classpath. The aggregate distribution
currently carries:

- Gemma 4, including E2B/E4B vision, E2B conformer audio and MTP sidecars;
- Qwen 3 and Qwen 3.5 dense/MoE, including embedding, reranking and Qwen 3.5 MTP;
- LFM 2.5 chat, embedding, ColBERT reranking and LFM 2.5 VL projection;
- Llama-family models, including Llama, Ministral, MiniCPM, SmolLM and Granite variants;
- gpt-oss, Nemotron-H, Maple, and Inflect speech synthesis.

The exact set is extensible through `ModelProvider`. GGUF support includes dense F32/F16/BF16 and
the quantized formats used by those ports: Q4_0, Q4_1, Q5_1, Q4_K, Q5_K, Q6_K, Q8_0, MXFP4,
NVFP4, Q1_0, TQ1_0 and TQ2_0.

### Capability → artifact map

Depend on one architecture module, the `jinfer-models-all` aggregate, or the `jinfer-bom` catalog:

| Capability | Artifact | Notes |
|------------|----------|-------|
| chat / instruct | `jinfer-<model>` | per-architecture port |
| embeddings | `jinfer-lfm2`, `jinfer-qwen3` | `EmbeddingModel` |
| reranking | `jinfer-lfm2` (ColBERT), `jinfer-qwen3` | `Reranker` |
| vision | `jinfer-gemma4`, `jinfer-lfm2`, `jinfer-qwen35` | `--mmproj <clip.gguf>` |
| audio input | `jinfer-gemma4` | E2B conformer |
| speech synthesis | `jinfer-inflect2` | Inflect TTS |
| MTP speculation | `jinfer-gemma4`, `jinfer-qwen35` | embedded MTP head |
| OpenAI server | `jinfer-server` | `/v1/chat/completions`, `/v1/completions`, `/v1/responses` |
| prompt cache | `jinfer-cache` | sessions + checkpoint tree + JKVF |
| hub + downloads | `jinfer-hub` | `hf.co/...`, `modelscope.cn/...` |

Add `jinfer-models-all` to get every architecture provider, or add individual `jinfer-<model>`
artifacts to keep the footprint small. `Models.load` discovers the providers present and names the
missing artifact when an architecture is unsupported.

### Version catalog (BOM)

```xml
<dependencyManagement>
  <dependencies>
    <dependency>
      <groupId>com.qxotic</groupId>
      <artifactId>jinfer-bom</artifactId>
      <version>0.1.0</version>
      <type>pom</type>
      <scope>import</scope>
    </dependency>
  </dependencies>
</dependencyManagement>
```

After the import, declare jinfer artifacts without versions; the BOM also pins the substrate
(`jota-memory`, `gguf`, `toknroll`, `jam`) to the release they were built against.

Multimodal models attach auxiliary files by capability:

```bash
jinfer \
  --model hf.co/unsloth/gemma-4-E2B-it-GGUF:Q4_K_M \
  --with media=hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf \
  --chat
```

`--mmproj` is shorthand for `--with media=...`. Other companion roles, such as
`speculation=<mtp.gguf>`, are declared by the architecture port and validated at load time.

## Models from a hub

Anywhere a tool or framework builder accepts a model string, it accepts a local GGUF path, a
supported model reference, or a pasted browser URL:

```text
hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0
hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf
hf.co/ggml-org/models@a1b2c3d/bert-bge-small
modelscope.cn/Qwen/Qwen3-0.6B-GGUF:Q8_0
```

Downloads are resumable and checksum-verified. Warm resolution makes no request. `JINFER_MODELS`
moves the cache, `HF_TOKEN` unlocks gated Hugging Face repositories, and `JINFER_OFFLINE=1` (or
`-Djinfer.offline`) forbids network access.

```bash
java -jar jinfer-cli/target/jinfer.jar pull hf.co/ggml-org/stories15M_MOE:Q8_0
java -jar jinfer-cli/target/jinfer.jar list
```

Model resolution happens before loading; inference paths never fetch content. Media codecs likewise
decode only caller-provided local files or bytes.

## Caching

`PromptCache` is the single core entry point. It combines:

- a bounded number of retained conversation states for append-only continuation;
- a content-addressed checkpoint tree under a byte budget;
- optionally one persisted catalog, mounted read-only or allowed to grow.

Every restore stops one position short and re-ingests the last token so logits are fresh. Cache
misses recompute; incompatible model identities cannot match. Framework users normally need only
`retainSessions(n)`, `promptCache(path)`, and `withCachedPrompt(...)`; the framework guides show
their exact semantics.

The CLI exposes a growing catalog as `--cache file.jkv` and a read-only one as `--cache-ro
file.jkv` in instruct and server modes.

## Performance and observability

- Vector API kernels run on x86 and ARM; JAM is selected automatically when a compatible backend
  is on the classpath.
- Prompt ingestion is batched; generation uses one serial state per pipeline. Create or fork
  another pipeline for actual parallel inference.
- JFR events cover model load, queue/prefill/decode/TTFT, cache state, media projection and MTP.
  The packaged `jinfer.jfc` enables the useful aggregate events while leaving per-token decode off.
- The server exports request outcomes, phase timings, token counters and cache/media gauges through
  `/metrics`.

For JVM runs, the repository's `hotspot_compiler` file contains the current inlining hints for hot
Vector API helpers.

## Native image

With GraalVM Native Image 25.0.3 or newer:

```bash
make native
./jinfer --model ./model.gguf --chat
```

`PRELOAD_GGUF=model.gguf make native` embeds load metadata/tokenizer data for faster startup. Media
decoding uses ffmpeg in the native image so `java.desktop` does not have to be pulled into the
binary.

## Build and test

Java 25 is required.

```bash
mvn test
mvn -pl jinfer-cli -am package -DskipTests
make jar       # copies jinfer-cli/target/jinfer.jar to ./jinfer.jar
```

Model-backed integration tests are opt-in and use the repository's `TestModels` cache lookup; unit
tests and weights-free contract tests run without downloading models.

## Scope

jinfer performs inference. It does not train, fine-tune or quantize models. GPU matmul can be
provided by a JAM backend, but jinfer is not a CUDA/Metal graph runtime.

## License

Apache 2.0
