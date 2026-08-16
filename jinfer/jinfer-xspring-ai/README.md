# jinfer Spring AI integration (x)

In-process Spring AI chat, embeddings and reranking over GGUF models. There is no server or HTTP
hop: jinfer runs the model inside the application JVM.

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jinfer-xspring-ai</artifactId>
  <version>0.1.0</version>
</dependency>
```

Run with:

```text
--enable-preview --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
```

## Chat

```java
ChatModel model = JinferChatModel.builder()
        .model("hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF:Q8_0")
        .contextLength(8192) // 0 = the model maximum; negative values are rejected
        .options(JinferChatOptions.builder()
                .temperature(0.7)
                .maxTokens(512)
                .thinking(false)
                .build())
        .build();

String answer = model.call("What is the capital of France?");
```

`model(String)` accepts a local path, hub reference or URL. `modelPath(Path)` is the explicitly
offline form. Generation defaults are one `JinferChatOptions` value; a prompt can supply another
value for request-local overrides:

```java
ChatResponse response = model.call(new Prompt(
        new UserMessage("Answer in one sentence."),
        JinferChatOptions.builder().maxTokens(64).seed(42L).build()));
```

`stream(Prompt)` emits text deltas and a final response carrying finish reason, tool calls and
usage. Disposing the subscription cancels generation.

Tools use Spring AI's `ToolCallingChatOptions`; structured output uses `outputSchema` or
`ChatClient.entity(..., EntityParamSpec::useProviderStructuredOutput)`. The schema is enforced by
the decoder rather than merely injected into the prompt.

## Images, audio and video

Attach the model's projector as the `media` companion and pass ordinary Spring `Media` values:

```java
ChatModel vision = JinferChatModel.builder()
        .model("hf.co/LiquidAI/LFM2.5-VL-3B-GGUF:Q4_K_M")
        .companion("media", "hf.co/LiquidAI/LFM2.5-VL-3B-GGUF/mmproj-F16.gguf")
        .build();

ChatResponse response = vision.call(new Prompt(UserMessage.builder()
        .text("Read this image.")
        .media(new Media(MimeTypeUtils.IMAGE_PNG, new FileSystemResource("page.png")))
        .build()));
```

Media bytes are decoded locally, then projected into model embeddings. Video frame selection is
configurable with `videoSampler`; jinfer never fetches media URLs on an inference path.

## Prompt caching

There are three deliberately separate controls:

- `retainSessions(n)` keeps up to `n` finished conversation states in RAM for append-only reuse;
- `withCachedPrompt(messages, tools)` defines a reusable, content-addressed prefix;
- `promptCache(path)` mounts one persisted prompt catalog, while `saveCachedPrompts(path)` exports
  the prefixes currently known by the model.

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

The catalog is model-identity checked. An edited prompt reuses only its matching prefix; a miss
recomputes, never changes the answer. Cached media projection is shared by cached-prompt views.

## Embeddings and reranking

```java
EmbeddingModel embeddings = JinferEmbeddingModel.builder()
        .model("hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0")
        .contextLength(2048)
        .build();

float[] vector = embeddings.embed("search query");

DocumentPostProcessor reranker = JinferDocumentPostProcessor.builder()
        .modelPath(Path.of("reranker.gguf"))
        .contextLength(2048)
        .topK(5)
        .build();
```

Embedding requests are packed into ragged batches. `EmbeddingOptions.dimensions` requests a
supported Matryoshka width; fixed-width models accept only their native width. Typed query/document
entry points apply the model's retrieval framing, while `call(EmbeddingRequest)` remains raw.

## Spring Boot

Use `jinfer-xspring-ai-spring-boot-starter` and configure the same model string:

```yaml
spring:
  ai:
    jinfer:
      chat:
        model: hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF:Q8_0
        context-length: 8192
        max-tokens: 512
```

The starter also exposes `spring.ai.jinfer.embedding`, `spring.ai.jinfer.rerank` and
`spring.ai.jinfer.speech`. Model resolution happens at context startup, so invalid configuration
fails during boot.

## Lifetime and concurrency

One adapter is one serial inference pipeline. `fork()` creates another state over shared weights
when the caller supplied the loaded model and owns its arena. Close adapters before closing that
arena. `JinferChatModel`, `JinferEmbeddingModel` and `JinferDocumentPostProcessor` are
`AutoCloseable`; Spring closes managed beans automatically.

Shaded jars must merge `META-INF/services` entries, for example with Maven Shade's
`ServicesResourceTransformer`, so architecture providers remain discoverable.
