# jinfer-spring-ai

[Spring AI](https://docs.spring.io/spring-ai/reference/) `ChatModel` backed by [jinfer](../README.md): in-process CPU inference over local GGUF files.
No server, no HTTP - the model runs inside your JVM.
Prompts go through jinfer's hand-written, oracle-validated chat-template codecs (token-exact, injection-inert by construction); unported models fall back to a hardened render of their own embedded Jinja template.

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jinfer-spring-ai</artifactId>
  <version>0.1.0</version>
</dependency>
```

Requires Spring AI 2.0+ (Spring Boot 4 / Framework 7).
Run your JVM with jinfer's flags:

```
--enable-preview --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
```

## Chat

```java
ChatModel model = JinferChatModel.builder()
        .model("hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF:Q8_0")
        .build();

String answer = model.call("What is the capital of France?");
```

`.model(String)` takes whatever you have: a local GGUF path, a hub ref, or a pasted browser URL.
A remote ref downloads at `build()` into the shared model cache (resumable, sha256-verified; warm builds cost zero requests, `JINFER_OFFLINE=1` forbids the network - see [models from the hub](../README.md#models-from-the-hub)).
`.modelPath(Path)` stays the never-network form.

## Zero-config with the Boot starter

With `jinfer-spring-ai-spring-boot-starter` on the classpath, one property and zero Java wires the bean:

```yaml
spring:
  ai:
    jinfer:
      chat:
        model: hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF:Q8_0   # a local path works too
```

The property takes the same one string as `.model(String)` and resolves at context startup, so a typo, a wrong quant, or a gated repo fails the boot with the hub's own message - never the first request at 2 p.m. under load.
Embeddings (`spring.ai.jinfer.embedding.model` + `spring.ai.model.embedding: jinfer`), reranking (`spring.ai.jinfer.rerank.model`), and speech (`spring.ai.jinfer.speech.model`) wire the same way; chat companions ride `spring.ai.jinfer.chat.companions.media`.

## Parameters

Model-level defaults on the builder, per-request overrides as `JinferChatOptions` on the `Prompt`.

```java
ChatModel model = JinferChatModel.builder()
        .modelPath(Path.of("models/LFM2.5-8B-A1B-Q8_0.gguf"))
        .contextLength(8192)      // 0 = the model's full context
        .temperature(0.7)
        .topP(0.95)
        .maxTokens(1024)
        .thinking(false)          // reasoning scaffold off (models without one ignore it)
        .seed(42)                 // deterministic sampling
        .build();

ChatResponse response = model.call(new Prompt(
        new UserMessage("Explain BPE in two sentences."),
        JinferChatOptions.builder().maxTokens(128).build()));   // this request only

System.out.println(response.getResult().getOutput().getText());
System.out.println(response.getMetadata().getUsage());                 // real token counts
System.out.println(response.getResult().getMetadata().getFinishReason()); // stop | length | tool_calls
```

String stop sequences are supported.
Unsupported knobs (penalties, per-request `model` switching) throw `IllegalArgumentException` instead of being silently ignored.

## Streaming

`stream(Prompt)` emits one `ChatResponse` per delta - text only, no metadata - and a final chunk carrying the finish reason, complete tool calls and token usage.
Generation runs on a virtual thread; disposing the subscription aborts it.

```java
model.stream(new Prompt(new UserMessage("Tell me a haiku about rivers.")))
        .doOnNext(chunk -> System.out.print(chunk.getResult().getOutput().getText()))
        .blockLast();
```

## Tools

`JinferChatOptions` implements `ToolCallingChatOptions`, so `ChatClient`'s tool loop (`ToolCallingAdvisor`) works unchanged - the model proposes tool calls, the framework executes them and loops.

```java
ToolCallback weather = MethodToolCallback.builder()
        .toolObject(new Weather())
        .toolDefinition(ToolDefinition.from(new Weather().getClass().getMethod("weather", String.class)))
        .build();

ChatResponse r = model.call(new Prompt(
        new UserMessage("What's the weather in Paris?"),
        JinferChatOptions.builder().toolCallbacks(weather).build()));

r.getResult().getOutput().getToolCalls();   // [{id, type:"function", name:"weather", arguments:"{...}"}]
```

## Structured output

`entity()` works out of the box via Spring AI's default prompt injection.
Opt into native enforcement with `useProviderStructuredOutput()`: the schema leaves the prompt and is compiled to a GBNF grammar that masks logits at decode time - invalid JSON becomes unrepresentable, not just unlikely, and no schema boilerplate spends prompt tokens.

```java
record Capital(String city, String country) {}

Capital capital = ChatClient.create(model)
        .prompt("What is the capital of France?")
        .call()
        .entity(Capital.class, ChatClient.EntityParamSpec::useProviderStructuredOutput);
```

For reasoning models the grammar stays dormant during the think span and activates at `</think>`.
An output schema combined with tools throws `IllegalArgumentException` (grammar-constrained output cannot admit tool-call syntax).

## Images and audio (Gemma 4)

Multimodal models take their encoders from a sidecar GGUF (llama.cpp's `mmproj` convention).
Media is decoded locally (bytes or a local file - this library never fetches over the network) and enters the model as embeddings, never as text.

```java
ChatModel gemma = JinferChatModel.builder()
        .model("hf.co/unsloth/gemma-4-12b-it-qat-GGUF:Q4_K_XL")
        .companion("media", "hf.co/unsloth/gemma-4-12b-it-qat-GGUF/mmproj-F32.gguf")
        .build();

ChatResponse seen = gemma.call(new Prompt(UserMessage.builder()
        .text("What is in this picture?")
        .media(new Media(MimeTypeUtils.IMAGE_PNG, new FileSystemResource("sign.png")))
        .build()));
```

## Embeddings

`JinferEmbeddingModel` runs an embedding GGUF (Qwen3-Embedding, LFM2.5-Embedding) in-process, so the whole RAG stack - vectors, store, chat - stays in one JVM with zero egress.
Inputs are packed into context-sized ragged batches: one forward pass embeds many segments, so ingesting hundreds of chunks costs a handful of prefills, not hundreds.
Usage reports exact token counts; `EmbeddingOptions.getDimensions()` truncates vectors.

```java
JinferEmbeddingModel embeddings = JinferEmbeddingModel.builder()
        .modelPath(Path.of("models/Qwen3-Embedding-0.6B-Q8_0.gguf"))
        .contextLength(2048)          // packing window; <= 0 = the model's maximum
        .build();

EmbeddingResponse r = embeddings.call(
        new EmbeddingRequest(List.of("first chunk", "second chunk"), null));
```

With the starter, `spring.ai.jinfer.embedding.model` (+ `context-length`) wires the bean; `spring.ai.model.embedding` selects the provider.

## Cached prompts

A cached prompt is paid for once and cheap forever after: `withCachedPrompt` prefills the prefix
(system prompt, tools, few-shot, even images) into an in-memory KV block tree and returns a model
view whose requests restore it instead of recomputing it - users only pay for their own data.
Content-addressed (no names), memory-only by default, byte-identical output to the uncached path.

```java
JinferChatModel base = JinferChatModel.builder().modelPath(gguf).build();

// prefill once; every call on the view pays only the user's text
JinferChatModel support = base.withCachedPrompt(
        List.of(new SystemMessage(SUPPORT_INSTRUCTIONS)), supportTools);
support.call("How do I reset my password?");

// several prompts share one tree - common prefixes are stored once
JinferChatModel sales = base.withCachedPrompt(
        List.of(new SystemMessage(SALES_INSTRUCTIONS)), salesTools);

// optional persistence: freeze everything into one artifact...
base.saveCachedPrompts(Path.of("dist/personas.jkv"));

// ...and mount it in the next process: re-declaring a stored prompt costs zero prefill
JinferChatModel base2 = JinferChatModel.builder()
        .modelPath(gguf)
        .loadCachedPrompts(Path.of("dist/personas.jkv"))   // model-seed-checked
        .build();
JinferChatModel support2 = base2.withCachedPrompt(
        List.of(new SystemMessage(SUPPORT_INSTRUCTIONS)), supportTools);  // instant
```

Rules: the base model never touches the tree (fully stateless by default); views are immutable,
composable (`withCachedPrompt` on a view branches on its prefix), and reject per-request
`toolCallbacks` (tools are welded into the cached prefix - on `ChatClient`, that means no
`defaultToolCallbacks` on a view); an edited prompt matches to the divergence point and pays only
the tail; a wrong-model artifact fails at `build()`. Requires a model with a native template codec
(the Jinja fallback makes no prefix-stability promise).

## Notes

- One generation runs at a time per loaded model; concurrent `call`s queue fairly.
- Reasoning is exposed as `AssistantMessage` metadata under `"thinking"` and replayed into the next request's assistant turn (the Ollama/OpenAI convention); in streaming, reasoning deltas arrive as chunks flagged `isThought` (core's `MessageAggregator` accumulates them on the `thoughts` lane).
- Observability: `call`/`stream` are instrumented with Micrometer (`gen_ai.client.operation`) - pass an `ObservationRegistry` (and optionally a `ChatModelObservationConvention`) on the builder; the starter wires the context's beans automatically. Usage reports real token counts plus `cacheReadInputTokens` for cached-prompt requests (tokens restored from the block tree) and phase timings (`prompt-eval-duration`, `eval-duration`) as metadata key-values.
- `JinferChatModel` is `AutoCloseable`: closing frees the prompt tree's native arenas and fails later requests fast; views share the base's engine, so closing any of them closes all. Spring Boot calls `close()` on shutdown automatically.
- Shaded/fat-jar consumers need Maven Shade's `ServicesResourceTransformer` (the architecture ports register via `ServiceLoader`).
