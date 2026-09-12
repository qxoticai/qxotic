<h1 align="center">jinfer for Spring AI</h1>

<p align="center"><strong>Fast local LLM inference for Spring AI.</strong></p>

<p align="center">
  <a href="https://openjdk.org/projects/jdk/25/"><img src="https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white" alt="Java 25+"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache" alt="License: Apache 2.0"></a>
  <a href="https://www.graalvm.org/latest/reference-manual/native-image/"><img src="https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F" alt="GraalVM Native Image"></a>
</p>

Run AI in your application's JVM through Spring AI, powered by [jinfer](../README.md).
No inference server, Python runtime or API key required.

## What it does

- **Spring Boot auto-configuration.** Pick Jinfer with `spring.ai.model.chat=jinfer` and configure a model to use it through Spring AI's APIs.
- **Constrained generation.** Strictly sample tokens following the JSON schema Spring AI derives from your Java types.
- **Automatic tool calling.** Expose `@Tool` methods and let Spring AI manage tool execution and follow-up model calls.
- **Multimodal support.** Pass Spring `Media` values to models that support images, audio or video.
- **Local embeddings and reranking.** Generate embeddings through `EmbeddingModel` and rerank retrieved documents with `DocumentPostProcessor`.
- **Local inference.** Models download on first use and stay cached; inference runs locally.

## Spring Boot quick start

The examples use Java 25, Spring Boot 4.1.1 and Spring AI 2.0.1.
Add the [dependencies below](#add-the-provider) to your Spring Boot application, then set these JVM options:

```text
--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
```

For `mvn spring-boot:run`, configure the plugin to pass the flags and enable full JIT compilation:

```xml
<plugin>
  <groupId>org.springframework.boot</groupId>
  <artifactId>spring-boot-maven-plugin</artifactId>
  <configuration>
    <optimizedLaunch>false</optimizedLaunch>
    <jvmArguments>--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED</jvmArguments>
  </configuration>
</plugin>
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
        context-capacity: 4096
        max-tokens: 512
```

Then inject a `ChatClient.Builder` and build the client:

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

Spring owns the model bean and closes it at application shutdown.

The starter creates beans from these properties:

| Capability | Selection | Model source |
|------------|-----------|--------------|
| Chat | `spring.ai.model.chat=jinfer` | `spring.ai.jinfer.chat.model` |
| Embeddings | `spring.ai.model.embedding=jinfer` | `spring.ai.jinfer.embedding.model` |
| Reranking | configured when a model is present | `spring.ai.jinfer.rerank.model` |
| Speech | configured when a model is present | `spring.ai.jinfer.speech.model` |

Chat is enabled when its model is configured and no other chat provider is selected.
Model sources may be remote references or local paths and resolve during application startup.
Add companions under `spring.ai.jinfer.chat.companions`, keyed by capability such as `media` or `speculation`.

The [translation table](https://qxotic.ai/jinfer#the-same-knob-at-each-face) maps settings between Spring AI, LangChain4j, the CLI and the server.

## Add the provider

Use the Spring Boot parent or BOM for Spring Boot itself.
With the Spring Boot parent, set `<java.version>25</java.version>`.
Import the Spring AI and jinfer BOMs, then add the starter and model providers:

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
    <artifactId>jinfer-models-all</artifactId>
  </dependency>
</dependencies>
```

`jinfer-models-all` includes the providers used by the examples below, not model weights.
Only the models you load are downloaded.
The BOMs manage versions; they do not add dependencies.
Without them, pin `0.2.0` on each jinfer dependency and `2.0.1` on each Spring AI one.

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

Include either or both.
With both present, jinfer prefers the native backend and falls back to the Vector one; with neither, it uses its built-in kernels.

### JBang

For scripts that use the core Spring AI provider without Spring Boot:

```java
//JAVA 25
//JAVAC_OPTIONS -parameters
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//DEPS com.qxotic:jinfer-bom:0.2.0@pom
//DEPS com.qxotic:jinfer-spring-ai com.qxotic:jinfer-models-all
//DEPS org.springframework.ai:spring-ai-client-chat:2.0.1
//DEPS org.springframework.ai:spring-ai-rag:2.0.1
//DEPS com.qxotic:jam-native com.qxotic:jam-vector
```

## Structured output

Using the `chat` client created above, return a Java record:

```java
record Incident(String severity, String summary, boolean pageOnCall) {}

Incident incident = chat.prompt(
        "Extract this incident as a JSON object with severity (a short string), "
        + "summary (one short sentence), and pageOnCall (a boolean). "
        + "Severity: high. Checkout has returned HTTP 503 for ten minutes. "
        + "Page the on-call engineer.")
        .call()
        .entity(Incident.class, ChatClient.EntityParamSpec::useProviderStructuredOutput);
```

Spring derives the schema, and Jinfer constrains generation to it.
`useProviderStructuredOutput` does not add schema instructions to the prompt; describe the task and expected fields yourself.
A token limit or cancellation can still leave an incomplete response.

## Tools

Given a Spring-injected `ChatModel model`, register a tool with `ChatClient`.
This example returns fixed weather data:

```java
final class WeatherTools {
    @Tool(description = "Get the current weather for a city")
    String weather(@ToolParam(description = "City name, for example Paris") String city) {
        return "18C, sunny in " + city;
    }
}

ChatClient chat = ChatClient.builder(model)
        .defaultTools(new WeatherTools())
        .build();

String answer = chat.prompt("Use the weather tool to check the weather in Paris.").call().content();
System.out.println(answer);
```

Compile tool classes with `-parameters` so Spring can read their parameter names.
The Spring Boot parent enables this; for other Maven projects, set `<parameters>true</parameters>` on `maven-compiler-plugin`.
The JBang header above includes the flag.

## Without Spring Boot

Keep the BOMs and model providers shown above, and replace the starter with the integration and `ChatClient` dependencies:

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jinfer-spring-ai</artifactId>
</dependency>
<dependency>
  <groupId>org.springframework.ai</groupId>
  <artifactId>spring-ai-client-chat</artifactId>
</dependency>
```

### Chat

```java
try (var model = JinferChatModel.builder()
        .model("LiquidAI/LFM2.5-350M-GGUF:Q8_0")
        .options(JinferChatOptions.builder().maxTokens(512).build())
        .build()) {

    System.out.println(model.call("What is the capital of France?"));
}
```

The application owns manually built models; use try-with-resources or close them at shutdown.
Use `model("...")` for a model reference and `modelPath(Path.of("model.gguf"))` for a local file.
Companions follow the same pattern with `companion(...)` and `companionPath(...)`.

A model reference is `owner/repo[:quant]`, resolved from Hugging Face by default.
Name a host to use another source, as in `modelscope.cn/Qwen/Qwen3-0.6B-GGUF:Q8_0`.
Set `JINFER_OFFLINE=1` to require cached models and prohibit downloads.

Chat context capacity defaults to 4096 tokens, or the model maximum when smaller.
Set `contextCapacity(0)` to use the full model maximum.
Negative values and explicit capacities above that maximum are rejected.

`stream(Prompt)` returns a cancellable stream of text deltas and a final response.

## Images, audio and video

This example describes a local image with LFM2.5-VL.
Supply your own `page.png` and attach the model's projector as the `media` companion:

```java
try (var vision = JinferChatModel.builder()
        .model("LiquidAI/LFM2.5-VL-3B-GGUF:Q8_0")
        .companion("media", "LiquidAI/LFM2.5-VL-3B-GGUF/mmproj-LFM2.5-VL-3B-Q8_0.gguf")
        .options(JinferChatOptions.builder().maxTokens(128).build())
        .build()) {

    ChatResponse response = vision.call(new Prompt(UserMessage.builder()
            .text("Describe this image in one sentence.")
            .media(new Media(MimeTypeUtils.IMAGE_PNG, new FileSystemResource("page.png")))
            .build()));

    System.out.println(response.getResult().getOutput().getText());
}
```

Media bytes are decoded locally, then projected into model embeddings.
Audio and video require a model and companion that support those inputs; this LFM2.5-VL example uses images only.
Configure video frame selection with `videoSampler`.
Jinfer does not fetch media during inference.

## Embeddings and reranking

When using reranking without the Spring Boot starter, add the RAG API.
Its version comes from the Spring AI BOM:

```xml
<dependency>
  <groupId>org.springframework.ai</groupId>
  <artifactId>spring-ai-rag</artifactId>
</dependency>
```

```java
try (var embeddings = JinferEmbeddingModel.builder()
                .model("Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0")
                .contextCapacity(2048)
                .build();
        var reranker = JinferDocumentPostProcessor.builder()
                .model("mradermacher/Qwen3-Reranker-0.6B-GGUF:Q8_0")
                .topK(1)
                .build()) {

    String question = "How do I reset my password?";
    float[] vector = embeddings.embed(question);
    System.out.println("Embedding dimensions: " + vector.length);

    List<Document> candidates = List.of(
            new Document("Use the password reset link on the sign-in page."),
            new Document("Standard shipping takes three business days."));
    List<Document> ranked = reranker.process(new Query(question), candidates);
    ranked.forEach(document -> System.out.println(document.getText()));
}
```

The example embeds a query and reranks two supplied documents.
In a RAG application, pass the documents returned by your retriever to the reranker.

## Prompt caching

Define a reusable system prompt, save it, and restore it in a new model instance:

```java
String modelRef = "LiquidAI/LFM2.5-350M-GGUF:Q8_0";
Path catalog = Path.of("personas.jkv");
List<Message> instructions = List.of(new SystemMessage(
        "You are a support assistant for AcmeCloud. Answer in one sentence. "
        + "Password resets are at https://acme.example/reset."));

try (var base = JinferChatModel.builder().model(modelRef).build()) {
    var support = base.withCachedPrompt(instructions, List.of());
    System.out.println(support.call("How do I reset my password?"));
    base.saveCachedPrompts(catalog);
}

try (var restored = JinferChatModel.builder()
        .model(modelRef)
        .promptCache(catalog)
        .build()) {
    var support = restored.withCachedPrompt(instructions, List.of());
    System.out.println(support.call("How do I reset my password?"));
}
```

Use the same model and prefix when restoring the catalog.
Cached-prompt views share their base model's lifetime; close the base, not each view.

The [Spring AI cache integration tests](src/test/java/com/qxotic/jinfer/spring/ai/CachedPromptIT.java) verify restored token counts and catalog round-trips.

## Speech synthesis

`JinferSpeechModel` implements Spring AI's `TextToSpeechModel`.
`call` returns a complete WAV file; `stream` emits ordered, headerless little-endian PCM16 chunks.
This example saves a WAV file, then reports the chunks from a second synthesis:

```java
try (var speech = JinferSpeechModel.builder()
        .model("remixerdec/Inflect-Nano-v2-GGUF:Q8_0")
        .companion("lexicon", "remixerdec/Inflect-Nano-v2-GGUF/lexicon.bin")
        .build()) {

    Files.write(Path.of("hello.wav"), speech.call("Hello from local Java inference."));

    speech.stream(new TextToSpeechPrompt("Hello again."))
            .doOnNext(response -> System.out.printf("%d PCM bytes, %s Hz, %s channels%n",
                    response.getResult().getOutput().length,
                    response.getMetadata().get(JinferSpeechModel.SAMPLE_RATE),
                    response.getMetadata().get(JinferSpeechModel.CHANNELS)))
            .blockLast(); // keep the model alive until the stream finishes
}
```

Chunks may include silence and do not necessarily align with sentences.
In a WebFlux application, inject the Spring-managed `TextToSpeechModel` and return its `Flux` from the handler.
Block only at an imperative boundary such as this standalone example, never on a WebFlux event-loop thread.

Kokoro takes its voice as a companion, the same way; `espeak-ng` must be on `PATH`:

```java
try (var speech = JinferSpeechModel.builder()
        .model("simonfxr/kokoro.cpp-GGUF:Q8_0")
        .companion("voice", "simonfxr/kokoro.cpp-GGUF/voices/kokoro-voice-af_heart.gguf")
        .build()) {
    Files.write(Path.of("kokoro.wav"), speech.call("Hello from Kokoro."));
}
```

A request that names a `voice`, `model` or `format` this instance does not have is refused, never silently given the default.

## Complete examples

- [`jinfer-example-local-rag`](../jinfer-example-local-rag/README.md) keeps embedding, retrieval, vector storage and grounded chat in one Spring Boot JVM.
- [`jinfer-example-judge-advisor`](../jinfer-example-judge-advisor/README.md) uses a local, grammar-constrained judge to reject bad answers and drive Spring AI's self-refine loop.
