---
sidebar_position: 3
---

# Spring AI integration

**Spring AI, fully local.** In-process [Spring AI](https://spring.io/projects/spring-ai) chat, embeddings and reranking over GGUF models, with a Spring Boot starter. No server, no HTTP hop.

## Installation

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jinfer-spring-ai</artifactId>
  <version>0.2.0</version>
</dependency>
```

Run with:

```text
--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
```

## Chat

```java
try (var model = JinferChatModel.builder()
        .model("LiquidAI/LFM2.5-8B-A1B-GGUF:Q8_0")
        .contextLength(8192) // 0 = the model maximum; negative values are rejected
        .options(JinferChatOptions.builder()
                .temperature(0.7)
                .maxTokens(512)
                .thinking(false)
                .build())
        .build()) {

    System.out.println(model.call("What is the capital of France?"));
}
```

`model(String)` accepts a model ref (`owner/repo:quant`, or `modelscope.cn/...` for another source). `modelPath(Path)` is the
local form and never touches the network. Companions use the same split:
`companion(capability, String)` accepts a model reference; `companionPath(capability, Path)`
accepts a local file. Spring Boot properties remain path-or-ref strings and the autoconfiguration
routes them to the matching method. A `JinferChatOptions` value on the prompt overrides per
request:

A bare repository reference follows llama.cpp and selects `Q4_K_M`. Jinfer's best-supported quant
is `Q8_0`, so the examples specify it explicitly.

```java
ChatResponse response = model.call(new Prompt(
        new UserMessage("Answer in one sentence."),
        JinferChatOptions.builder().maxTokens(64).seed(42L).build()));
```

`stream(Prompt)` emits text deltas and a final response with finish reason, tool calls, and usage. Disposing the subscription cancels generation.

Tools use `ToolCallingChatOptions`. Structured output uses `outputSchema` or `ChatClient.entity(..., EntityParamSpec::useProviderStructuredOutput)`. The schema is enforced by the decoder.

## Images, audio, and video

Attach the projector as the `media` companion and pass Spring `Media` values:

```java
ChatModel vision = JinferChatModel.builder()
        .model("LiquidAI/LFM2.5-VL-3B-GGUF:Q8_0")
        .companion(
                "media",
                "LiquidAI/LFM2.5-VL-3B-GGUF/mmproj-LFM2.5-VL-3B-Q8_0.gguf")
        .build();

ChatResponse response = vision.call(new Prompt(UserMessage.builder()
        .text("Read this image.")
        .media(new Media(MimeTypeUtils.IMAGE_PNG, new FileSystemResource("page.png")))
        .build()));
```

Media bytes are decoded locally, then projected into embeddings. `videoSampler` configures frame selection. jinfer never fetches media URLs on an inference path.

## Prompt caching

Three separate controls:

- `retainSessions(n)`: keep up to `n` finished states in RAM for append-only reuse
- `withCachedPrompt(messages, tools)`: define a reusable, content-addressed prefix
- `promptCache(path)` / `saveCachedPrompts(path)`: mount or export a persisted catalog

```java
JinferChatModel base = JinferChatModel.builder()
        .modelPath(gguf)
        .retainSessions(2)
        .build();

JinferChatModel support = base.withCachedPrompt(
        List.of(new SystemMessage(SUPPORT_INSTRUCTIONS)), supportTools);
support.call("How do I reset my password?");

base.saveCachedPrompts(Path.of("personas.jkv"));

JinferChatModel restored = JinferChatModel.builder()
        .modelPath(gguf)
        .promptCache(Path.of("personas.jkv"))
        .build();
JinferChatModel restoredSupport = restored.withCachedPrompt(
        List.of(new SystemMessage(SUPPORT_INSTRUCTIONS)), supportTools);
```

The catalog is model-identity checked. An edited prompt reuses only its matching prefix; a miss recomputes and never changes the answer. Cached media projection is shared by cached-prompt views.

## Embeddings and reranking

```java
EmbeddingModel embeddings = JinferEmbeddingModel.builder()
        .model("Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0")
        .contextLength(2048)
        .build();

float[] vector = embeddings.embed("search query");

DocumentPostProcessor reranker = JinferDocumentPostProcessor.builder()
        .modelPath(Path.of("reranker.gguf"))
        .contextLength(2048)
        .topK(5)
        .build();
```

Embedding requests are packed into ragged batches. `EmbeddingOptions.dimensions` requests a supported Matryoshka width; fixed-width models accept only their native width. Typed query/document entry points apply retrieval framing; `call(EmbeddingRequest)` stays raw.

## Spring Boot

Use `jinfer-spring-ai-spring-boot-starter`:

```yaml
spring:
  ai:
    jinfer:
      chat:
        model: LiquidAI/LFM2.5-8B-A1B-GGUF:Q8_0
        context-length: 8192
        max-tokens: 512
```

Also exposed: `spring.ai.jinfer.embedding`, `spring.ai.jinfer.rerank`, `spring.ai.jinfer.speech`. Model resolution happens at startup; invalid configuration fails the boot.

## Lifetime and concurrency

One adapter is one serial pipeline. `fork()` creates another state over shared weights when the caller supplied the loaded model and owns its arena. Close adapters before closing that arena.

`JinferChatModel`, `JinferEmbeddingModel`, and `JinferDocumentPostProcessor` are `AutoCloseable`; Spring closes managed beans automatically.

Shaded jars must merge `META-INF/services` entries (e.g. Maven Shade's `ServicesResourceTransformer`) so architecture providers stay discoverable.
