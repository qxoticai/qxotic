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
  <version>0.3.1</version>
</dependency>
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jinfer-lfm2</artifactId>
  <version>0.3.1</version>
</dependency>
```

The provider loads nothing by itself: add one model family (`jinfer-lfm2` here), or `jinfer-models-all` for every family.
Spring Boot applications use `jinfer-spring-ai-spring-boot-starter` instead of `jinfer-spring-ai`, see the [module README](https://github.com/qxoticai/qxotic/blob/main/jinfer/jinfer-spring-ai/README.md).

Run with:

```text
--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
```

## Chat

```java
try (var model = JinferChatModel.builder()
        .model("LiquidAI/LFM2.5-8B-A1B-GGUF:Q8_0")
        .contextCapacity(8192) // 0 = the default, -1 = the model's maximum, other negatives rejected
        .options(JinferChatOptions.builder()
                .temperature(0.7)
                .maxTokens(512)
                .maxReasoningTokens(256) // this model always reasons: cap the span
                .build())
        .build()) {

    System.out.println(model.call("What is the capital of France?"));
}
```

`model(String)` accepts a model ref (`owner/repo:quant`, or `modelscope.cn/...` for another source). `modelPath(Path)` is the
local form and never touches the network. Companions use the same split:
`companion(capability, String)` accepts a model reference; `companionPath(capability, Path)`
accepts a local file. Spring Boot properties remain path-or-ref strings and the autoconfiguration
routes them to the matching method.

A bare repository reference follows llama.cpp and selects `Q4_K_M`. Jinfer's best-supported quant
is `Q8_0`, so the examples specify it explicitly.

A `JinferChatOptions` value on the prompt overrides per request:

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
        .contextCapacity(8192)
        .build();

float[] vector = embeddings.embed("search query");

DocumentPostProcessor reranker = JinferDocumentPostProcessor.builder()
        .modelPath(Path.of("reranker.gguf"))
        .contextCapacity(2048)
        .topK(5)
        .build();
```

The reranker needs the Spring AI RAG API, which `jinfer-spring-ai` declares as optional.
The starter brings it in; without the starter, add it at the same Spring AI version, for example by importing `spring-ai-bom`:

```xml
<dependency>
  <groupId>org.springframework.ai</groupId>
  <artifactId>spring-ai-rag</artifactId>
</dependency>
```

Embedding requests are packed into ragged batches.
`EmbeddingOptions.dimensions` requests a supported Matryoshka width (Qwen3-Embedding: 32 up to its own, EmbeddingGemma 2: 128 to 768); fixed-width models accept only their native width.
Typed query/document entry points apply retrieval framing; `call(EmbeddingRequest)` stays raw.

Vector stores call `add` and `similaritySearch` without options, and some size their table from `dimensions()`.
To store shorter vectors, set the width on the model: `dimensions(256)` on the builder, or `spring.ai.jinfer.embedding.dimensions=256` in the starter.
Every call that names no width then embeds at 256, `dimensions()` reports it, and observations record it.
`EmbeddingOptions.dimensions` on a call still wins; `0`, the default, is the model's own width, and a width the model cannot produce is refused at startup.

EmbeddingGemma 2 also embeds media documents (image, audio or video) once its projector is attached, with `companion("media", ...)` on the builder or `spring.ai.jinfer.embedding.companions.media` in the starter:

```java
float[] photo = embeddings.embed(Document.builder()
        .media(new Media(MimeTypeUtils.IMAGE_PNG, Path.of("photo.png").toUri()))
        .build());
```

The vector lands in the space text queries search.
As the model card says, media embeds without a prompt; text documents keep theirs.
Spring AI's vector stores still refuse non-text documents at `add`, so media vectors come from the `EmbeddingModel` directly.

Queries and documents cover search, the one task Spring AI has a word for.
EmbeddingGemma 2's other tasks (question answering, fact checking, code retrieval, classification, clustering, sentence similarity) are jinfer calls on the model's own module: see [jinfer-gemma-embedding2](../../jinfer/jinfer-gemma-embedding2/README.md).

## Spring Boot

Use `jinfer-spring-ai-spring-boot-starter`:

```yaml
spring:
  ai:
    jinfer:
      chat:
        model: LiquidAI/LFM2.5-8B-A1B-GGUF:Q8_0
        context-capacity: 8192
        max-tokens: 512
```

Also exposed: `spring.ai.jinfer.embedding`, `spring.ai.jinfer.rerank`, `spring.ai.jinfer.speech`, `spring.ai.jinfer.transcription`.
Rerank, speech and transcription activate when their `model` is set.
Embedding also needs explicit selection with `spring.ai.model.embedding=jinfer`, so a chat-only app is never asked for an embedding GGUF.
Model resolution happens at startup; invalid configuration fails the boot.

## Lifetime and concurrency

One adapter is one serial pipeline.
For another, load the weights once into an arena you own (`Models.load`, `Models.loadEmbedder`, `Models.loadReranker`) and build each adapter with `model(loaded)`; adapters share the weights and nothing else.
Close adapters before closing that arena.

`JinferChatModel`, `JinferEmbeddingModel`, `JinferDocumentPostProcessor`, `JinferSpeechModel`, and `JinferTranscriptionModel` are `AutoCloseable`; Spring closes managed beans automatically.

Shaded jars must merge `META-INF/services` entries (e.g. Maven Shade's `ServicesResourceTransformer`) so architecture providers stay discoverable.
