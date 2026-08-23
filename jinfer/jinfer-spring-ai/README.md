# Jinfer for Spring AI

[![Java 25+](https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white)](https://openjdk.org/projects/jdk/25/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache)](../LICENSE)
[![GraalVM Native Image](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)

A Spring AI provider backed by the [Jinfer](../README.md) inference engine.

**Local LLM inference inside your JVM. No server. No Python. No external processes.**

## Add the provider

Import the Jinfer BOM, then add the starter and model providers:

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

<dependencies>
  <dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>jinfer-spring-ai-spring-boot-starter</artifactId>
  </dependency>
  <dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>jinfer-models-all</artifactId>
    <type>pom</type>
  </dependency>
</dependencies>
```

Replace `jinfer-models-all` with individual providers such as `jinfer-lfm2`, `jinfer-gemma4`,
`jinfer-qwen3` or `jinfer-inflect2` to keep the classpath small.

Optional runtime backends:

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

Include either backend or both. When both are present, Jinfer tries the native backend first and
uses the Java Vector backend as a fallback. Without them, Jinfer uses its built-in kernels.

### JBang

For scripts that use the core Spring AI provider without Spring Boot:

```java
//JAVA 25
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//DEPS com.qxotic:jinfer-bom:0.1.0@pom
//DEPS com.qxotic:jinfer-spring-ai com.qxotic:jinfer-lfm2
//DEPS com.qxotic:jam-native com.qxotic:jam-vector
```

## Spring Boot quick start

Java 25 is required. Add these options to the application JVM:

```text
--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
```

Configure a model reference or local model path:

```yaml
spring:
  ai:
    model:
      chat: jinfer
    jinfer:
      chat:
        model: hf.co/LiquidAI/LFM2.5-350M-GGUF:Q8_0
        context-length: 4096
        max-tokens: 256
```

Use Spring AI's `ChatClient`:

```java
@Service
final class Assistant {
    private final ChatClient chat;

    Assistant(ChatClient.Builder builder) {
        this.chat = builder.build();
    }

    String ask(String question) {
        return chat.prompt(question).call().content();
    }
}
```

The starter creates beans from these properties:

| Capability | Selection | Model source |
|------------|-----------|--------------|
| Chat | `spring.ai.model.chat=jinfer` | `spring.ai.jinfer.chat.model` |
| Embeddings | `spring.ai.model.embedding=jinfer` | `spring.ai.jinfer.embedding.model` |
| Reranking | configured when a model is present | `spring.ai.jinfer.rerank.model` |
| Speech | configured when a model is present | `spring.ai.jinfer.speech.model` |

Chat is the default when no other chat provider is selected. Model sources may be remote references
or local paths and resolve during application startup. Add companions under
`spring.ai.jinfer.chat.companions`, keyed by capability such as `media` or `speculation`.

## Structured output

Return a Java value from a Jinfer-backed `ChatClient`:

```java
record Incident(String severity, String summary, boolean pageOnCall) {}

Incident incident = chat.prompt("Checkout has returned HTTP 503 for ten minutes.")
        .call()
        .entity(
                Incident.class,
                ChatClient.EntityParamSpec::useProviderStructuredOutput);
```

Spring derives the schema. Jinfer rejects tokens outside that schema during sampling.

## Without Spring Boot

Without Spring Boot, keep the BOM and model provider shown above and replace the starter with the
core integration:

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jinfer-spring-ai</artifactId>
</dependency>
```

### Chat

```java
try (var model = JinferChatModel.builder()
        .model("hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF:Q8_0")
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

Use `model("hf.co/...")` for a model reference and `modelPath(Path.of("model.gguf"))` for a local
file. Companions follow the same pattern with `companion(...)` and `companionPath(...)`. Examples
pin `Q8_0`; a bare repository follows llama.cpp and selects `Q4_K_M`.

`stream(Prompt)` returns a cancellable stream of text deltas and a final response.

## Tools

Spring AI `@Tool` methods run through the same automatic tool loop:

```java
final class WeatherTools {
    @Tool(description = "Get the current weather for a city")
    String weather(String city) {
        return "18C, sunny in " + city;
    }
}

ChatClient chat = ChatClient.builder(model)
        .defaultTools(new WeatherTools())
        .build();

String answer = chat.prompt("What is the weather in Paris?").call().content();
```

## Images, audio and video

Attach the model's projector as the `media` companion and pass Spring `Media` values:

```java
try (var vision = JinferChatModel.builder()
        .model("hf.co/LiquidAI/LFM2.5-VL-3B-GGUF:Q8_0")
        .companion(
                "media",
                "hf.co/LiquidAI/LFM2.5-VL-3B-GGUF/mmproj-LFM2.5-VL-3B-Q8_0.gguf")
        .build()) {

    ChatResponse response = vision.call(new Prompt(UserMessage.builder()
            .text("Read this image.")
            .media(new Media(MimeTypeUtils.IMAGE_PNG, new FileSystemResource("page.png")))
            .build()));

    System.out.println(response.getResult().getOutput().getText());
}
```

Media bytes are decoded locally, then projected into model embeddings. Configure video frame
selection with `videoSampler`. Jinfer does not fetch media during inference.

## Prompt caching

Prefill a long system prompt once:

```java
Path gguf = Path.of("models/LFM2.5-8B-A1B-Q8_0.gguf");

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

[`CachedPrompt.java`](../examples/scripts/CachedPrompt.java) prints the restored token count.

## Embeddings and reranking

```java
try (var embeddings = JinferEmbeddingModel.builder()
        .model("hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0")
        .contextLength(2048)
        .build();
     var reranker = JinferDocumentPostProcessor.builder()
        .model("hf.co/mradermacher/Qwen3-Reranker-0.6B-GGUF:Q8_0")
        .contextLength(2048)
        .topK(5)
        .build()) {

    float[] vector = embeddings.embed("search query");
}
```

Use embeddings for broad retrieval, then rerank the candidates without a vector service or
reranking endpoint.

## Speech synthesis

`JinferSpeechModel` implements Spring AI's `TextToSpeechModel`. Blocking calls return WAV bytes;
streaming emits PCM clips as they are synthesized:

```java
try (var speech = JinferSpeechModel.builder()
        .model("hf.co/remixerdec/Inflect-Nano-v2-GGUF:Q8_0")
        .build()) {

    Files.write(Path.of("hello.wav"), speech.call("Hello from local Java inference."));
}
```

## Complete examples

- [`jinfer-example-local-rag`](../jinfer-example-local-rag/README.md) keeps embedding, retrieval,
  vector storage and grounded chat in one Spring Boot JVM.
- [`jinfer-example-judge-advisor`](../jinfer-example-judge-advisor/README.md) uses a local,
  grammar-constrained judge to reject bad answers and drive Spring AI's self-refine loop.
- [`Narrate.java`](../examples/scripts/Narrate.java) runs two models from one Java file and writes a
  spoken description of an image.

## Lifetime and concurrency

One adapter is one serial inference pipeline. `fork()` adds an independent state that shares the
loaded weights. Spring closes managed adapters automatically.

Shaded JARs must preserve `ServiceLoader` entries. With Maven Shade, add
`ServicesResourceTransformer`.
