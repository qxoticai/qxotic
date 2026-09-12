---
sidebar_position: 1
---

# jinfer

**Local LLM inference for the JVM. No server. No Python. No ONNX. Just a jar.**

In-process GGUF inference: chat, streaming, tool calling, structured output, embeddings,
reranking, speech, multimodal projection, prompt caching, and an OpenAI-compatible server.
Models download from the hub on first use and stay cached.

Vector API kernels provide the portable floor. [jam](/jam) accelerates matmul when a compatible
backend is on the classpath.

## Quick start

Build the CLI:

```bash
mvn -pl jinfer/jinfer-cli -am package -DskipTests

java --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -jar jinfer/jinfer-cli/target/jinfer.jar \
  --model LiquidAI/LFM2.5-350M-GGUF:Q8_0 \
  --chat
```

Replace `--chat` with `--prompt "..."` for one-shot generation. `--context-capacity 0` uses the model's declared maximum. Negative values and capacities above the model maximum are rejected.

Use a framework adapter for application code:

- LangChain4j: `com.qxotic:jinfer-langchain4j`, [guide](/jinfer/langchain4j)
- Spring AI: `com.qxotic:jinfer-spring-ai`, [guide](/jinfer/spring-ai)

Single-file demos live in [`jinfer/examples/scripts`](https://github.com/qxoticai/qxotic/tree/main/jinfer/examples/scripts).

## OpenAI-compatible server

```bash
java --add-modules jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -jar jinfer/jinfer-cli/target/jinfer.jar \
  --model LiquidAI/LFM2.5-350M-GGUF:Q8_0 \
  --server --port 54154
```

Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`, `/v1/tokenize`, `/v1/detokenize`, `/health`, Prometheus `/metrics`. Chat supports streaming, tools, structured output, reasoning, stop strings, deterministic seeds, and multimodal content when the model has a projector.

Loopback is the default. Non-loopback binding requires `--api-key`. Queue, body, admission, generation, and stalled-write limits have explicit CLI flags. Run `--help` for the full contract.

## Models and capabilities

Architecture dispatch comes from providers on the classpath.

| Family | Capabilities |
|--------|--------------|
| Gemma 4 | chat, E2B/E4B vision, E2B conformer audio, MTP |
| Qwen 3 | embeddings, reranking |
| Qwen 3.5 | chat, vision, MTP |
| LFM 2.5 | chat, embeddings, ColBERT reranking, VL projection |
| Laguna XS 2.1 | chat |
| Mellum 2 | chat |
| Ling 3 | chat |
| Llama family | chat (Llama, Ministral, MiniCPM, SmolLM, Granite) |
| gpt-oss, Nemotron-H | chat |
| Inflect | speech synthesis |
| Kokoro 82M | speech synthesis |

GGUF support: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K, Q8_0, MXFP4, NVFP4, Q1_0, TQ1_0, TQ2_0.

### Capability → artifact

| Capability | Artifact | Notes |
|------------|----------|-------|
| chat / instruct | `jinfer-<model>` | per-architecture port |
| embeddings | `jinfer-lfm2`, `jinfer-qwen3` | `EmbeddingModel` |
| reranking | `jinfer-lfm2` (ColBERT), `jinfer-qwen3` | `Reranker` |
| vision | `jinfer-gemma4`, `jinfer-lfm2`, `jinfer-qwen35` | `--mmproj <clip.gguf>` |
| audio input | `jinfer-gemma4` | E2B conformer |
| speech synthesis | `jinfer-inflect2`, `jinfer-kokoro` | Kokoro needs a voice GGUF and eSpeak on `PATH` |
| MTP speculation | `jinfer-gemma4`, `jinfer-qwen35` | embedded MTP head |
| prompt cache | `jinfer-cache` | sessions + checkpoint tree + JKVF |
| hub + downloads | `jinfer-hub` | `owner/repo`, `modelscope.cn/...` |

Add `jinfer-models-all` for every provider, or individual `jinfer-<model>` artifacts. `Models.load` discovers present providers and names the missing artifact when an architecture is unsupported.

### Version catalog (BOM)

```xml
<dependencyManagement>
  <dependencies>
    <dependency>
      <groupId>com.qxotic</groupId>
      <artifactId>jinfer-bom</artifactId>
      <version>0.2.0</version>
      <type>pom</type>
      <scope>import</scope>
    </dependency>
  </dependencies>
</dependencyManagement>
```

After the import, declare jinfer artifacts without versions. The BOM also pins the substrate (`jota-memory`, `gguf`, `toknroll`, `jam`).

Attach multimodal projectors by capability:

```bash
jinfer \
  --model unsloth/gemma-4-E2B-it-GGUF:Q8_0 \
  --with media=unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf \
  --chat
```

`--mmproj` is shorthand for `--with media=...`. Other companion roles (e.g. `speculation=<mtp.gguf>`) are declared by the architecture port and validated at load.

## Models from a hub

Java framework builders keep remote and local sources explicit: `model(String)` accepts a model
reference, while `modelPath(Path)` accepts a local GGUF file and never touches the network. A
plain URL is not a model reference; download it first, then pass its path. Companions follow the
same rule: `companion(capability, String)` accepts a model reference and
`companionPath(capability, Path)` accepts a local file.

Model references use one of these forms:

```text
unsloth/gemma-4-E2B-it-GGUF:Q8_0
unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf
ggml-org/models@a1b2c3d/bert-bge-small
hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0          the host, written out
modelscope.cn/Qwen/Qwen3-0.6B-GGUF:Q8_0         another source
```

The host is optional and defaults to `hf.co`. Name one to reach another source. A host-less
reference never swallows a local path: an existing file of that name wins, anything spelled like a
path is a path, and `owner/model.gguf` is a path too, since no repository is named after a model
file. Pass local files to `modelPath(...)`.

A reference with no quant uses llama.cpp's `Q4_K_M` default so the same shorthand selects the same
file in both tools. Jinfer's best-supported quant is `Q8_0`, so the examples pin it explicitly.

Downloads are resumable and checksum-verified. Warm resolution makes no request. `JINFER_MODELS` moves the cache, `HF_TOKEN` unlocks gated repos, `JINFER_OFFLINE=1` (or `-Djinfer.offline`) forbids network access.

```bash
java -jar jinfer-cli/target/jinfer.jar pull ggml-org/stories15M_MOE:Q8_0
java -jar jinfer-cli/target/jinfer.jar list
```

Inference paths never fetch content. Media codecs decode only caller-provided local files or bytes.

## Memory safety: confined arenas are refused

`Arena.ofConfined()` is refused for jinfer weights and state: loading a model or creating a state with one throws `IllegalArgumentException`.

Jinfer is multi-threaded by design.
Even when configured to use a single worker, execution may run on a different thread from the arena's owner - for example, a worker in a custom pool or a native pthread.
A confined arena permits access only from its owning Java thread; reducing the worker count or serializing calls does not satisfy that requirement.
Raw-address and native kernels bypass the JDK's confinement checks, so a confined arena would corrupt memory without a `WrongThreadException`.

The check runs once per weight mapping or state construction, never per generated token or kernel call.
It allocates a zero-byte buffer on the calling thread, then checks whether that buffer is accessible from another thread; it does not test concurrent allocation.
The temporary, never-started probe thread is local to the check; no `Thread` is retained in a static field.
Custom allocators may treat zero-byte allocations differently from real buffers, so the probe certifies the arena, not every buffer a custom allocator hands out.

Use `Arenas.newCrossThread()` for the runtime default, or `Arena.ofShared()`, `Arena.ofAuto()` or `Arena.global()`.
Custom state allocators must return cross-thread-accessible memory for **every** allocation.
Keep caller-owned arenas alive until all models, states and operations borrowing them have finished; closing borrowed memory during inference is also unsafe.
A state still permits only one operation at a time.

## Prompt caching

`PromptCache` combines:

- bounded retained conversation states for append-only continuation
- a content-addressed checkpoint tree under a byte budget
- optionally one persisted catalog, read-only or growing

Every restore stops one position short and re-ingests the last token, so logits stay fresh. Misses recompute. Incompatible model identities cannot match.

Framework users need only `retainSessions(n)`, `promptCache(path)`, and `withCachedPrompt(...)`; see the [LangChain4j](/jinfer/langchain4j#cached-prompts) and [Spring AI](/jinfer/spring-ai#prompt-caching) guides. The CLI exposes `--cache file.jkv` (growing) and `--cache-ro file.jkv` (read-only).

## The same knob at each face

One engine, four faces: the Java API, the CLI, the OpenAI-compatible server and the two framework providers.
A knob a face inherits keeps that face's name - `max_tokens` is OpenAI's, `maxOutputTokens` LangChain4j's, `maxTokens` Spring AI's `ChatOptions` - and a knob jinfer adds is named the same on every face.
This table is the translation.

| concept | Java API | CLI | server | LangChain4j | Spring AI |
|---|---|---|---|---|---|
| context capacity: the usable context of this model or state, what you set | `contextCapacity` (`ChatEngine`, `PromptCache.Options`) | `--context-capacity`, `-c` | `n_ctx` in `/props` | `contextCapacity` | `contextCapacity`, `spring.ai.jinfer.chat.context-capacity` |
| max context length: what the checkpoint was trained for, the ceiling of the above | `ContextConfiguration.maxContextLength()` | the default ceiling | `n_ctx_train` in `/props` | `contextCapacity(0)` selects it | same |
| max output tokens | `Request.maxOutputTokens` | `--max-output-tokens` | `max_tokens`, `max_completion_tokens` | `maxOutputTokens` | `maxTokens` |
| temperature, top-p, top-k, min-p, seed | `Sampling` | `--temp`, `--top-p`, `--top-k`, `--min-p`, `--seed` | `temperature`, `top_p`, `top_k`, `min_p`, `seed` | builder, per request `JinferChatRequestParameters` | `JinferChatOptions` |
| thinking | `Request.thinking` | `--think` | `chat_template_kwargs.enable_thinking` | `thinking` | `thinking` |
| reasoning budget | `Request.maxReasoningTokens` | `--max-reasoning-tokens` | `max_reasoning_tokens` | `maxReasoningTokens` | `maxReasoningTokens` |
| grammar (raw GBNF) | `Request.grammar` | - | `grammar` | `JinferChatRequestParameters.grammar` | `JinferChatOptions.grammar` |
| speculation depth | `ChatEngine.speculationDepth` | `--speculation-depth` | `--speculation-depth` at start | `speculationDepth` | `speculationDepth` |
| prompt cache on disk | `PromptCache.Options.withCatalog` | `--cache`, `--cache-ro` | `--cache`, `--cache-ro` at start | `promptCache` | `promptCache` |
| structured output | `Request.grammar` from `Grammar.schemaGbnf` | - | `response_format` | `AiServices` return type, described to the model in one line (`describeSchema`) | `outputSchema`, described by Spring AI's own converters, never by the provider |

## Performance and observability

- Vector API kernels run on x86 and ARM. JAM is selected automatically when present.
- Prompt ingestion is batched. Generation uses one serial state per pipeline; fork another pipeline for parallel inference.
- Threads: `-Djinfer.threads` (default: physical cores) sizes the one pool every kernel and jam backend runs on; there is no other thread knob.
- JFR events cover load, queue/prefill/decode/TTFT, cache state, media projection, and MTP. The packaged `jinfer.jfc` enables aggregate events and leaves per-token decode off.
- The server exports request outcomes, phase timings, token counters, and cache/media gauges at `/metrics`.

For JVM runs, `hotspot_compile_commands` holds the current inlining hints for hot Vector API helpers.

## Native image

GraalVM Native Image 25.0.3 or newer:

```bash
make native
./bin/jinfer --model ./model.gguf --chat
```

`PRELOAD_GGUF=model.gguf make native` embeds load metadata/tokenizer data for faster startup. Media decoding uses ffmpeg in the native image, so `java.desktop` is not pulled in.

## Scope

jinfer performs inference. It does not train, fine-tune, or quantize. GPU matmul can come from a JAM backend, but jinfer is not a CUDA/Metal graph runtime.
