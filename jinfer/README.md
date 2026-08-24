# Jinfer

[![Java 25+](https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white)](https://openjdk.org/projects/jdk/25/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache)](LICENSE)
[![GraalVM Native Image](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)

**Local LLM inference inside your JVM. No server. No Python. No external processes.**

Jinfer runs chat, streaming, tool calling, constrained output, multimodal input, embeddings,
reranking and speech synthesis from Java. Vector API kernels provide the portable backend.
[JAM](../jam) accelerates matrix multiplication when available.

## Choose an API

| Use case | Start here |
|----------|------------|
| LangChain4j applications | [`jinfer-langchain4j`](jinfer-langchain4j/README.md) |
| Spring AI and Spring Boot applications | [`jinfer-spring-ai`](jinfer-spring-ai/README.md) |
| Terminal and HTTP access | [CLI and OpenAI-compatible server](#cli) |
| Small executable examples | [JBang demos](examples/scripts/README.md) |

## Try the demos

Install [JBang](https://www.jbang.dev/), then run these commands from a repository checkout:

```bash
cd jinfer/examples/scripts

jbang Chat.java "Invent a tiny language for talking to houseplants."  # streaming text
jbang Json.java "Ada Lovelace, born 1815 in London."                  # constrained JSON
jbang Narrate.java photo.jpg                                         # vision and speech
jbang Detect.java street.jpg "person, bicycle, traffic light"         # annotated PNG
```

The scripts request Java 25 and resolve their dependencies through JBang. Models download on first
use and remain in the Jinfer cache. `Detect.java` uses a 12B vision model and requests a 16 GB heap;
start with `Chat.java` or `Json.java` for a smaller download.

## Add Jinfer to an application

### Maven

Import the BOM once:

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

Add one API and one or more model providers:

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jinfer-langchain4j</artifactId>
</dependency>
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jinfer-lfm2</artifactId>
</dependency>
```

Add `jinfer-models-all` instead of `jinfer-lfm2` to include every model provider.

For faster matrix multiplication, add either JAM backend or both. When both are present, Jinfer
prefers the native backend and uses the Java Vector backend as a fallback.

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jam-native</artifactId>
  <scope>runtime</scope>
</dependency>
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jam-vector</artifactId>
  <scope>runtime</scope>
</dependency>
```

### JBang

```java
//JAVA 25
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//DEPS com.qxotic:jinfer-bom:0.1.0@pom
//DEPS com.qxotic:jinfer-langchain4j
//DEPS com.qxotic:jinfer-lfm2
//DEPS com.qxotic:jam-native com.qxotic:jam-vector
```

Use `jinfer-gemma4` for vision/audio, `jinfer-qwen3` for embeddings/reranking and
`jinfer-inflect2` for speech.

## Generate text from Java

```java
try (var model = JinferChatModel.builder()
        .model("hf.co/LiquidAI/LFM2.5-350M-GGUF:Q8_0")
        .build()) {
    System.out.println(model.chat("Explain virtual threads in one sentence."));
}
```

Spring Boot applications use `jinfer-spring-ai-spring-boot-starter` instead. See the
[LangChain4j guide](jinfer-langchain4j/README.md) and
[Spring AI guide](jinfer-spring-ai/README.md).

Run Java applications with:

```text
--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
```

## CLI

Build the CLI from source:

```bash
mvn -pl jinfer/jinfer-cli -am package -DskipTests

java --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -jar jinfer/jinfer-cli/target/jinfer.jar \
  --model hf.co/LiquidAI/LFM2.5-350M-GGUF:Q8_0 \
  --chat
```

Replace `--chat` with `--prompt "..."` for one-shot generation.

## Capabilities

- Chat, streaming, reasoning and tools
- JSON Schema and GBNF constrained generation
- Images, audio and video
- Embeddings and reranking
- Speech synthesis
- Prompt and session caching
- OpenAI-compatible server
- GraalVM native image

See the [JBang demo gallery](examples/scripts/README.md) for speech, object detection, semantic
search, reranking and prompt-cache accounting. The repository also includes complete
[local RAG](jinfer-example-local-rag/README.md) and
[judge advisor](jinfer-example-judge-advisor/README.md) applications.

## OpenAI-compatible server

```bash
java --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -jar jinfer/jinfer-cli/target/jinfer.jar \
  --model hf.co/LiquidAI/LFM2.5-350M-GGUF:Q8_0 \
  --server --port 54154
```

The server implements `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`,
`/v1/tokenize`, `/v1/detokenize`, `/health` and Prometheus `/metrics`. Chat supports streaming,
tools and tool choice, structured output, reasoning, stop strings, deterministic seeds and
multimodal content when the loaded model has a projector.

Loopback is the safe default. Binding to a non-loopback host requires `--api-key`; queue, body,
thread, generation and stalled-write limits have explicit CLI flags. Run `--help` for the complete
contract.

## Models and capabilities

Architecture dispatch comes from the model providers on the classpath. The included providers
support:

- Gemma 4, including vision, audio and MTP sidecars
- Qwen 3 and Qwen 3.5 dense and MoE models, including embeddings, reranking and MTP
- LFM 2.5 chat, embeddings, ColBERT reranking and vision
- Llama-family models, including Llama, Ministral, MiniCPM, SmolLM and Granite variants
- gpt-oss, Nemotron-H, Maple and Inflect speech synthesis

Applications can add architectures through `ModelProvider`. Supported tensor formats include dense
F32, F16 and BF16 plus Q4_0, Q4_1, Q5_1, Q4_K, Q5_K, Q6_K, Q8_0, MXFP4, NVFP4, Q1_0, TQ1_0 and
TQ2_0.

### Capability and artifact map

The BOM manages every artifact below. Choose `jinfer-models-all` or only the providers you use:

| Capability | Artifact | Notes |
|------------|----------|-------|
| chat and instruct | `jinfer-<model>` | per-architecture provider |
| embeddings | `jinfer-lfm2`, `jinfer-qwen3` | `EmbeddingModel` |
| reranking | `jinfer-lfm2` (ColBERT), `jinfer-qwen3` | `ScoringModel` (`JinferScoringModel`) |
| vision | `jinfer-gemma4`, `jinfer-lfm2`, `jinfer-qwen35` | `--mmproj <clip.gguf>` |
| audio input | `jinfer-gemma4` | Gemma 4 audio projectors |
| speech synthesis | `jinfer-inflect2` | Inflect TTS |
| MTP speculation | `jinfer-gemma4`, `jinfer-qwen35` | embedded MTP head |
| OpenAI server | `jinfer-server` | `/v1/chat/completions`, `/v1/completions`, `/v1/responses` |
| prompt caching | `jinfer-cache` | sessions, checkpoints and JKVF persistence |
| model resolution | [`jinfer-hub`](jinfer-hub/README.md) | Hugging Face and ModelScope references |

Add `jinfer-models-all` to get every architecture provider, or add individual `jinfer-<model>`
artifacts to keep the footprint small. `Models.load` discovers the providers present and names the
missing artifact when an architecture is unsupported.

Multimodal models attach auxiliary files by capability:

```bash
java --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -jar jinfer/jinfer-cli/target/jinfer.jar \
  --model hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0 \
  --with media=hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf \
  --chat
```

`--mmproj` is shorthand for `--with media=...`. Other companion roles, such as
`speculation=<mtp.gguf>`, are declared by the architecture port and validated at load time.

## Models from a hub

Java framework builders keep remote and local sources explicit. `model(String)` accepts a
supported model reference. `modelPath(Path)` accepts a local model file and does not access the
network. A plain URL is not a model reference. Download it first, then pass its path.

Companions follow the same rule: `companion(capability, String)` accepts a model reference and
`companionPath(capability, Path)` accepts a local file.

```text
hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0
hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf
hf.co/ggml-org/models@a1b2c3d/bert-bge-small
modelscope.cn/Qwen/Qwen3-0.6B-GGUF:Q8_0
```

A bare repository reference uses llama.cpp's `Q4_K_M` default so the same shorthand selects the
same file in both tools. Jinfer's best-supported quant is `Q8_0`, so the examples pin it explicitly.

Downloads are resumable and checksum-verified. Resolving a cached reference makes no network
request. A disk-space check accounts for completed chunks before a transfer starts or resumes. The
hub reads each override from the system property first, then the environment variable:

| Setting | Default | Effect |
|---------|---------|--------|
| `-Djinfer.models` / `JINFER_MODELS` | `~/.cache/jinfer` (macOS: `~/Library/Caches/jinfer`, Windows: `%LOCALAPPDATA%\jinfer`) | Move the model cache |
| `HF_TOKEN` | unset | Authenticate access to gated Hugging Face repositories |
| `-Djinfer.offline` / `JINFER_OFFLINE=1` | off | Forbid network access; resolve from the cache only |
| `-Djinfer.downloadThreads` / `JINFER_DOWNLOAD_THREADS` | 4 to 8, from the core count | Parallel chunk connections per download |
| `JINFER_SKIP_DISK_CHECK=1` | off | Skip the disk-space check, for network mounts that report zero free space |

```bash
java -jar jinfer/jinfer-cli/target/jinfer.jar pull hf.co/ggml-org/stories15M_MOE:Q8_0
java -jar jinfer/jinfer-cli/target/jinfer.jar list
```

Model resolution happens before loading; inference paths never fetch content. Media codecs likewise
decode only caller-provided local files or bytes.

## Caching

`PromptCache` combines:

- a bounded number of retained conversation states for append-only continuation
- a content-addressed checkpoint tree under a byte budget
- an optional persisted catalog, mounted read-only or allowed to grow

Cache misses use normal prompt ingestion, and entries from another model cannot match. Framework
users normally need only `retainSessions(n)`, `promptCache(path)` and `withCachedPrompt(...)`.

The CLI exposes a growing catalog as `--cache file.jkv` and a read-only one as `--cache-ro
file.jkv` in instruct and server modes.

## Performance and observability

Use [`jinfer-bench`](jinfer-bench/README.md) to measure prefill, decode, embedding throughput, TTFT,
prompt-cache hits, MTP and projected media. Its README defines the workloads and rerun rules.

- Vector API kernels run on x86 and ARM; JAM is selected automatically when a compatible backend
  is on the classpath.
- Prompt ingestion is batched; generation uses one serial state per pipeline. Create or fork
  another pipeline for parallel inference.
- Jinfer uses dedicated worker pools rather than the JVM common pool. Set
  `-Djinfer.computeThreads=N` for Java prefill work and `-Djinfer.decodeThreads=N` for generation.
  JAM providers also own their workers. Set `-Djam.threads=N` globally or
  `-Djam.<provider>.threads=N` for one provider. The [`jinfer-bench`](jinfer-bench/README.md) `-t
  N` option configures all of these pools together.
- JFR events cover model load, queue/prefill/decode/TTFT, cache state, media projection and MTP.
  The packaged `jinfer.jfc` enables the useful aggregate events while leaving per-token decode off.
- The server exports request outcomes, phase timings, token counters and cache/media gauges through
  `/metrics`.

For JVM runs, [`hotspot_compile_commands`](hotspot_compile_commands) contains the current inlining hints for hot
Vector API helpers.

## Native image

With GraalVM Native Image 25.0.3 or newer:

```bash
make -C jinfer native
./jinfer/jinfer --model ./model.gguf --chat
```

`PRELOAD_GGUF=model.gguf make -C jinfer native` embeds model metadata and tokenizer data for faster
startup. Media decoding uses FFmpeg in the native image, which avoids including `java.desktop` in
the binary.

## Build and test

Java 25 is required. Run these commands from the repository root:

```bash
mvn test                                            # all modules
mvn -pl jinfer/jinfer-cli -am package -DskipTests   # the CLI and everything it needs
make -C jinfer test                                 # only the jinfer subtree's tests
make -C jinfer jar                                  # the CLI jar, copied to jinfer/jinfer.jar
```

Model-backed integration tests are opt-in and use the repository's `TestModels` cache lookup; unit
tests and weights-free contract tests run without downloading models.

Running Maven from `jinfer/` requires the sibling projects to be installed first. Using the root
reactor with `-pl ... -am` avoids that prerequisite.

## Scope

Jinfer performs inference. It does not train, fine-tune or quantize models. A JAM backend may
provide GPU matrix multiplication, but Jinfer is not a CUDA or Metal graph runtime.

## License

Apache 2.0
