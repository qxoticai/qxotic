<h1 align="center">jinfer for Spring AI</h1>

<p align="center"><strong>AI in a jar, wired into Spring Boot with one property.</strong></p>

<p align="center">
  <a href="https://openjdk.org/projects/jdk/25/"><img src="https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white" alt="Java 25+"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache" alt="License: Apache 2.0"></a>
  <a href="https://www.graalvm.org/latest/reference-manual/native-image/"><img src="https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F" alt="GraalVM Native Image"></a>
</p>

## The mission

**AI sovereignty for the JVM.**

An existing `ChatClient` keeps its code. The model moves into the same JVM as the service, so
there is no inference server to run, no Python runtime, no external process and no API key. This is
a Spring AI provider backed by the [jinfer](../README.md) engine. The weights load during
application startup and live in the heap alongside the beans, so the inference runs end-to-end
within the JVM.

**Fast local LLM inference for the JVM. Just a jar.**

## What it does

- **One property.** `spring.ai.model.chat=jinfer` plus a model reference, and `ChatClient` is
  autoconfigured. Embeddings, reranking and speech are configured the same way.
- **Provider-native structured output.** `.entity(Incident.class, …useProviderStructuredOutput)`
  constrains sampling to Spring's derived schema, so the response cannot violate it.
- **Tools.** `@Tool` methods run through Spring AI's automatic tool-calling loop.
- **Multimodal.** Spring `Media` values for images, audio and video, decoded locally.
- **Retrieval in one process.** `EmbeddingModel` plus a `DocumentPostProcessor` reranker, with no
  vector service and no reranking endpoint.
- **Offline after the first run.** Model sources resolve at application startup from a local path
  or the cache, so nothing reaches the network in the request path.
- **Managed lifecycle.** Spring closes the adapters; `fork()` shares loaded weights across pipelines.

## Spring Boot quick start

Java 25 is required. Add these options to the application JVM:

```text
--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
```

Point a property at a model reference or a local model path:

```yaml
spring:
  ai:
    model:
      chat: jinfer
    jinfer:
      chat:
        model: LiquidAI/LFM2.5-350M-GGUF:Q8_0
        context-length: 4096
        max-tokens: 256
```

Then inject Spring AI's `ChatClient`:

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

## Add the provider

Use the Spring Boot parent or BOM for Spring Boot itself. Import the Spring AI and jinfer BOMs,
then add the starter and a model provider:

```xml
<dependencyManagement>
  <dependencies>
    <dependency>
      <groupId>org.springframework.ai</groupId>
      <artifactId>spring-ai-bom</artifactId>
      <version>2.0.1</version>
      <type>pom</type>
      <scope>import</scope>
    </dependency>
    <dependency>
      <groupId>com.qxotic</groupId>
      <artifactId>jinfer-bom</artifactId>
      <version>0.2.0</version>
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
    <artifactId>jinfer-lfm2</artifactId>
  </dependency>
</dependencies>
```

Use `jinfer-models-all` instead of `jinfer-lfm2` for every model family. The BOMs manage versions
only. Without them, pin `0.2.0` on each jinfer dependency and `2.0.1` on each Spring AI one.

Optional runtime backends are `jam-native` (hand-tuned SIMD) and `jam-vector` (Panama Vector API):

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

Include either or both. With both present, jinfer prefers the native backend and falls back to the
Vector one; with neither, it uses its built-in kernels.

### JBang

For scripts that use the core Spring AI provider without Spring Boot:

```java
//JAVA 25
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//DEPS com.qxotic:jinfer-bom:0.2.0@pom
//DEPS com.qxotic:jinfer-spring-ai com.qxotic:jinfer-lfm2
//DEPS com.qxotic:jam-native com.qxotic:jam-vector
```

## Structured output

Return a Java value from a jinfer-backed `ChatClient`:

```java
record Incident(String severity, String summary, boolean pageOnCall) {}

Incident incident = chat.prompt("Checkout has returned HTTP 503 for ten minutes.")
        .call()
        .entity(Incident.class, ChatClient.EntityParamSpec::useProviderStructuredOutput);
```

Spring derives the schema. jinfer rejects tokens outside that schema during sampling, so the record
is well-formed on the first attempt.

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

## Without Spring Boot

Keep the BOM and model provider shown above and replace the starter with the core integration:

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jinfer-spring-ai</artifactId>
</dependency>
```

### Chat

```java
try (var model = JinferChatModel.builder()
        .model("LiquidAI/LFM2.5-8B-A1B-GGUF:Q8_0")
        .contextLength(8192)   // 0 = the model maximum
        .options(JinferChatOptions.builder()
                .temperature(0.7)
                .maxTokens(512)
                .thinking(false)
                .build())
        .build()) {

    System.out.println(model.call("What is the capital of France?"));
}
```

Use `model("...")` for a model reference and `modelPath(Path.of("model.gguf"))` for a local
file. Companions follow the same pattern with `companion(...)` and `companionPath(...)`.

A reference is `owner/repo[:quant]`, which Hugging Face carries. Name a host to reach another
source, as in `modelscope.cn/Qwen/Qwen3-0.6B-GGUF:Q8_0`. Examples pin `Q8_0`; a reference with no
quant follows llama.cpp and selects `Q4_K_M`.

`stream(Prompt)` returns a cancellable stream of text deltas and a final response.

## Images, audio and video

Attach the model's projector as the `media` companion and pass Spring `Media` values:

```java
try (var vision = JinferChatModel.builder()
        .model("LiquidAI/LFM2.5-VL-3B-GGUF:Q8_0")
        .companion("media", "LiquidAI/LFM2.5-VL-3B-GGUF/mmproj-LFM2.5-VL-3B-Q8_0.gguf")
        .build()) {

    ChatResponse response = vision.call(new Prompt(UserMessage.builder()
            .text("Read this image.")
            .media(new Media(MimeTypeUtils.IMAGE_PNG, new FileSystemResource("page.png")))
            .build()));

    System.out.println(response.getResult().getOutput().getText());
}
```

Media bytes are decoded locally, then projected into model embeddings. Configure video frame
selection with `videoSampler`. jinfer does not fetch media during inference.

## Embeddings and reranking

When using reranking without the Spring Boot starter, add the RAG API. Its version comes from the
Spring AI BOM:

```xml
<dependency>
  <groupId>org.springframework.ai</groupId>
  <artifactId>spring-ai-rag</artifactId>
</dependency>
```

```java
try (var embeddings = JinferEmbeddingModel.builder()
                .model("Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0")
                .contextLength(2048)
                .build();
        var reranker = JinferDocumentPostProcessor.builder()
                .model("mradermacher/Qwen3-Reranker-0.6B-GGUF:Q8_0")
                .topK(5)
                .build()) {

    float[] vector = embeddings.embed("search query");
}
```

Use embeddings for broad retrieval, then rerank the candidates. No vector service or reranking
endpoint is involved.

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
        .promptCache(Path.of("personas.jkv"))   // the prefill survives the restart
        .build();
```

[`CachedPrompt.java`](../examples/scripts/CachedPrompt.java) prints the restored token count.

## Speech synthesis

`JinferSpeechModel` implements Spring AI's `TextToSpeechModel`. Blocking calls return WAV bytes;
streaming emits PCM clips as they are synthesized:

```java
try (var speech = JinferSpeechModel.builder()
        .model("remixerdec/Inflect-Nano-v2-GGUF:Q8_0")
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

## License

Apache 2.0
