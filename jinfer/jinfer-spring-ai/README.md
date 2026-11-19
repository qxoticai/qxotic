<h1 align="center">jinfer for Spring AI</h1>

<p align="center"><strong>Fast local LLM inference for Spring AI.</strong></p>

<p align="center">
  <a href="https://openjdk.org/projects/jdk/25/"><img src="https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white" alt="Java 25+"></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache" alt="License: Apache 2.0"></a>
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
- **Decisions.** Answer yes/no, choice and score questions about text, fields, images, audio and video with probabilities through `DecisionModel`.
- **Local inference.** Models download on first use and stay cached; inference runs locally.

## Spring Boot quick start

The examples use Java 25, Spring Boot 4.2.0 and Spring AI 2.1.0.
A Spring Boot native image (`mvn -Pnative native:compile`), built with GraalVM 25.0.4 or later, needs no extra configuration.
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
| Decisions | `spring.ai.model.decision=jinfer` | `spring.ai.jinfer.decision.model` |
| Reranking | configured when a model is present | `spring.ai.jinfer.rerank.model` |
| Speech | `spring.ai.model.audio.speech=jinfer` | `spring.ai.jinfer.speech.model` |
| Transcription | `spring.ai.model.audio.transcription=jinfer` | `spring.ai.jinfer.transcription.model` |

Chat, speech and transcription are also enabled when their model is configured and no other provider of that type is selected.
Model sources may be remote references or local paths and resolve during application startup.
Add companions under `spring.ai.jinfer.chat.companions`, keyed by capability such as `media` or `speculation`.
`spring.ai.jinfer.chat.output-modalities=text,audio` makes a model with a voice answer in speech, see [Spoken replies](#spoken-replies).
An embedder takes its projector the same way, as `spring.ai.jinfer.embedding.companions.media`.
With a `MeterRegistry` in the application (Spring Boot Actuator, for example), the starter records Spring AI's `gen_ai` metrics for the chat, embedding, decision, speech and transcription models: token usage and operation timings.

The [translation table](https://github.com/qxoticai/qxotic/blob/main/docs/jinfer/index.md#the-same-knob-at-each-face) maps settings between Spring AI, LangChain4j, the CLI and the server.

## Add the provider

These examples target jinfer **0.4.0**.
For an unreleased checkout, first install the artifacts locally from the repository root:

```bash
mvn -pl jinfer/jinfer-bom,jinfer/jinfer-spring-ai-spring-boot-starter,jinfer/jinfer-gemma-embedding2 -am install -DskipTests
```

Use the Spring Boot parent or BOM for Spring Boot itself.
With the Spring Boot parent, set `<java.version>25</java.version>`.
Import the Spring AI and jinfer BOMs, then add the starter and model providers:

```xml
<dependencyManagement>
  <dependencies>
    <dependency>
      <groupId>org.springframework.ai</groupId>
      <artifactId>spring-ai-bom</artifactId>
      <version>2.1.0</version>
      <type>pom</type>
      <scope>import</scope>
    </dependency>
    <dependency>
      <groupId>com.qxotic</groupId>
      <artifactId>jinfer-bom</artifactId>
      <version>0.4.0</version>
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
Without them, pin the versions `jinfer-bom` 0.4.0 lists on each jinfer dependency and `2.1.0` on each Spring AI one.

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
//DEPS com.qxotic:jinfer-bom:0.4.0@pom
//DEPS com.qxotic:jinfer-spring-ai com.qxotic:jinfer-models-all
//DEPS org.springframework.ai:spring-ai-client-chat:2.1.0
//DEPS org.springframework.ai:spring-ai-rag:2.1.0
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

## Conversation memory

The starter configures Spring AI's session memory: a `SessionService` bean, in memory unless a `SessionRepository` such as the JDBC one is on the classpath, and the advisors that attach it to a `ChatClient`.
Name the session on every call:

```java
ChatClient chat = ChatClient.builder(model)
        .defaultAdvisors(SessionAdvisors.builder(sessionService).build())
        .build();

chat.prompt("My name is Ada.").advisors(SessionAdvisors.session("s1")).call().content();
chat.prompt("What is my name?").advisors(SessionAdvisors.session("s1")).call().content();
```

Replies go back into the session as they came, reasoning and tool calls included, so a retained prompt cache keeps hitting across turns.
That holds for a session stored in a database too: a reply carries nothing of jinfer's, and the model remembers its recent replies (up to 16M token ids), so an echo that says the same replays the exact generated tokens.
An older echo, or one after a restart, renders from its text instead, re-tokenized, so its next reply can differ.

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
A URL is refused; the error shows the reference to use instead.
Set `JINFER_OFFLINE=1` to require cached models and prohibit downloads.

Context capacity defaults to 4096 tokens, or the model's maximum when smaller, for chat, embedding and reranking alike.
Set `contextCapacity(PromptCache.Options.MAXIMUM_CONTEXT)` (`-1`) to use the model's maximum; `0` is the default.
Other negative values, and explicit capacities above that maximum, are rejected.

A reply is a message of parts, in the model's order: the reasoning of a thinking model is a `ReasoningPart` (`getReasoning()`), the answer its text (`getText()`), then the tool calls.
Send the reply back as it is to continue the conversation; the reasoning replays with it.
`stream(Prompt)` returns a cancellable stream of deltas, each a text or a reasoning part, and a final response with the tool calls and usage; `MessageAggregator` and `ChatClient` rebuild the whole message from them.

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

A message is a list of parts in the order written, so text and media may interleave: `UserMessage.builder().part(TextPart.of("Before:")).part(MediaPart.of(image)).part(TextPart.of("After?"))` reaches the model in that order.
Media bytes are decoded locally, then projected into model embeddings.
Audio and video require a model and companion that support those inputs; this LFM2.5-VL example uses images only.
Configure video frame selection with `videoSampler`.
Jinfer does not fetch media during inference.

## Spoken replies

A model with a voice, LFM3-Omni with its projector attached, can answer in speech through the chat model.
What a reply is made of is the `outputModalities` option: unset, a reply is text alone and the model never switches into speech; `TEXT` and `AUDIO` is a spoken reply with its transcript; `AUDIO` alone is speech without one.
The option decides, never the prompt.
The speech is a `MediaPart` of `audio/wav` beside the transcript, whole from `call`, and a part per stretch from `stream`:

```java
try (var omni = JinferChatModel.builder()
        .model("LiquidAI/LFM3-Omni-2B-GGUF:Q8_0")
        .companion("media", "LiquidAI/LFM3-Omni-2B-GGUF/mmproj-LFM3-Omni-2B-F16.gguf")
        .options(JinferChatOptions.builder()
                .outputModalities(Modality.TEXT, Modality.AUDIO)
                .build())
        .build()) {

    ChatResponse response = omni.call(new Prompt(new UserMessage("What is the capital of France?")));

    AssistantMessage reply = response.getResult().getOutput();
    System.out.println(reply.getText()); // the transcript
    for (MessagePart part : reply.getParts()) {
        if (part instanceof MediaPart speech) {
            Files.write(Path.of("reply.wav"), speech.media().getDataAsByteArray());
        }
    }
}
```

The same option on a request's `JinferChatOptions` opens or closes the voice for that request alone, and `spring.ai.jinfer.chat.output-modalities=text,audio` sets the default in Spring Boot.
A streamed reply's media chunks are not aggregated: `ChatClient.stream().content()` yields the transcript, and a subscriber to `ChatModel.stream` collects the audio parts, each chunk a WAV of its own.
A spoken reply does not reason, so allowing audio turns `thinking` off, and it cannot offer tools or follow a grammar: such a request is refused in words.
A model without a voice answers `TEXT, AUDIO` in text, and `AUDIO` alone is refused with the remedy.
LiquidAI's voice-chat prompt, "Respond with interleaved text and audio." as the system message, shapes short spoken answers; a text-only reply under it can stop mid-sentence, so it belongs with requests that speak.
There is no voice option: LFM3-Omni's spoken chat replies come in its one chat voice, whatever the system prompt asks for.
The chat voice speaks English, French and German clearly; Spanish and Italian come out accented and Japanese and Chinese partly garbled, so ask for text in those.
For speech in a set language, LFM3-Omni's text-to-speech instruction takes the language explicitly (see [jinfer-lfm3](../jinfer-lfm3/README.md#text-to-speech)).

## Embeddings and reranking

### EmbeddingGemma 2 text

An embedding-only application needs `jinfer-spring-ai` and the Gemma provider below; it can omit `spring-ai-client-chat` and replace `jinfer-models-all` with this provider.
Their versions come from `jinfer-bom:0.4.0`.
Keep the starter instead of `jinfer-spring-ai` when Spring Boot manages the model.

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jinfer-gemma-embedding2</artifactId>
</dependency>
```

In a Maven project with these dependencies, save this as `GemmaText.java` in the project directory.
It takes the local path to `embeddinggemma-2-Q8_0.gguf` as its argument.
The checkpoint is available in [unsloth/embeddinggemma-2-GGUF](https://huggingface.co/unsloth/embeddinggemma-2-GGUF).
Text-only inference does not need the projector.

```java
import com.qxotic.jinfer.spring.ai.JinferEmbeddingModel;
import org.springframework.ai.document.Document;
import java.nio.file.Path;

public class GemmaText {
    public static void main(String[] args) {
        try (var embeddings = JinferEmbeddingModel.builder()
                .modelPath(Path.of(args[0]))
                .contextCapacity(8192)
                .dimensions(256)
                .build()) {
            float[] document = embeddings.embed(new Document("A tabby cat sitting on a windowsill."));
            float[] query = embeddings.embed("a photo of a cat");
            System.out.printf("Document: %d, query: %d%n", document.length, query.length);
        }
    }
}
```

Run it with Java 25's source-file launcher (POSIX shell), replacing the model path:

```bash
mvn dependency:build-classpath -Dmdep.outputFile=classpath.txt
java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  --class-path "$(cat classpath.txt)" GemmaText.java /path/to/embeddinggemma-2-Q8_0.gguf
```

`embed(Document)` and document batches apply document framing; `embed(String)` applies query framing.
String-list batches, `embedForResponse`, and `call(EmbeddingRequest)` stay raw: format batched queries explicitly with `GemmaEmbedding2.searchQuery(text)`.
For titles and other tasks, use the [Gemma helpers](../jinfer-gemma-embedding2/README.md) through a raw request, without passing the formatted text back through a query/document-framing overload.

Vector stores call `add` and `similaritySearch` without options, and some size their table from `dimensions()`.
`dimensions(256)` sets the default width and makes `dimensions()` report 256; request `EmbeddingOptions.dimensions` still wins.
Gemma's [model card](https://huggingface.co/google/embeddinggemma-2) lists widths of 768, 512, 256, and 128, and warns that 128 substantially reduces multimodal quality.
The adapter also accepts intermediate widths, which are not listed in the model card.
Returned shortened vectors are L2-normalized.
The builder's default, `dimensions(0)`, keeps the native width; unsupported widths are refused at build time.
Qwen3-Embedding accepts widths from 32 through its native width, while fixed-width models such as LFM2.5 refuse custom dimensions.
Keep the checkpoint, quantization, output width, and query/document framing consistent when building and searching an index.

`contextCapacity(8192)` selects Gemma's documented shared context, including task framing, boundary tokens, and any media positions.
Each input must fit; Spring's batching tokenizer is not Gemma's tokenizer, and packing several inputs does not split an oversized document.
Keep manually built adapters in try-with-resources, as above; Spring closes managed model beans at shutdown.

### EmbeddingGemma 2 media

Save this as `GemmaMedia.java` in the same project directory.
It takes three local paths: the model GGUF, `mmproj-BF16.gguf` from the same repository, and your PNG image.
The dependency set is the same as for text; the `media` companion enables image, audio, and video inputs.

```java
import com.qxotic.jinfer.spring.ai.JinferEmbeddingModel;
import org.springframework.ai.content.Media;
import org.springframework.ai.document.Document;
import org.springframework.util.MimeTypeUtils;
import java.nio.file.Path;

public class GemmaMedia {
    public static void main(String[] args) {
        try (var embeddings = JinferEmbeddingModel.builder()
                .modelPath(Path.of(args[0]))
                .companionPath("media", Path.of(args[1]))
                .contextCapacity(8192)
                .dimensions(256)
                .build()) {
            float[] photo = embeddings.embed(Document.builder()
                    .media(new Media(MimeTypeUtils.IMAGE_PNG, Path.of(args[2]).toUri()))
                    .build());
            System.out.println("Image: " + photo.length);
        }
    }
}
```

Using the classpath generated above, run it with your model, projector, and image paths:

```bash
java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  --class-path "$(cat classpath.txt)" GemmaMedia.java \
  /path/to/embeddinggemma-2-Q8_0.gguf /path/to/mmproj-BF16.gguf /path/to/photo.png
```

The image vector shares the embedding space used by text queries.
As the model card says, media embeds without a prompt; text documents keep theirs.
A Spring AI `Document` holds text or media, not both, so the mixed text-and-media input the LangChain4j adapter embeds as one vector cannot be expressed here.
Spring AI's vector stores still refuse non-text documents at `add`, so media vectors come from the `EmbeddingModel` directly.

For a starter-managed embedding model, select `spring.ai.model.embedding=jinfer`.
Set `model`, `context-capacity=8192`, and `dimensions=256` under `spring.ai.jinfer.embedding`.
For media, set `spring.ai.jinfer.embedding.companions.media` to the projector path.

### Reranking

The Qwen embedding/reranking example below needs `jinfer-qwen3:0.4.0`, or the `jinfer-models-all` dependency from the installation section.
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
                .contextCapacity(8192)
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

## Decisions

A decision model, such as Liquid AI's d1 or vLLM Semantic Router's d3, answers questions with probabilities in one forward pass, without generating text:

```java
try (var model = JinferDecisionModel.builder()
        .model("LiquidAI/d1-2B-GGUF:Q8_0")
        .companion("media", "LiquidAI/d1-2B-GGUF/mmproj-d1-2B-F16.gguf") // for images and video
        .build()) {

    DecisionResponse response = model.call(
            "I was charged twice for my order and I want my money back today.",
            Map.of(
                    "refund", YesNoQuestion.of("Is the customer asking for a refund?"),
                    "team", ChoiceQuestion.of("Which team should handle this?", "billing", "shipping", "technical"),
                    "urgency", ScoreQuestion.of("How urgent is this?", "Can wait", "This week", "Today", "Right now")));

    response.getYesNo("refund").getProbability();   // 0.99
    response.getChoice("team").getValue();          // "billing"
    response.getScore("urgency").getNearestLevel(); // 2
}
```

The input can be text and media parts (`DecisionInput.content(...)`) or fields (`DecisionInput.fields(...)`).
Fields are put to the model as JSON, the form d1 was trained on; a field holding content parts adds its text (or its media's kinds) to the JSON, and its media to the input, never as base64 in the prompt.
d1 reads images and video, d1-Omni also audio, d3 images.
A conversation (`DecisionInput.conversation(...)`) is put to the model as `{"messages": [...]}`, the transcript Spring AI's System One and Ollama decision models send d1: text, tool calls and tool results; reasoning stays out and media is refused.
Media goes in `DecisionInput.content(...)`; LangChain4j's decision conversation does carry it, as numbered markers like `[image 1]`.

The questions of a call are taken the way the model was trained to, with no isolation between them guaranteed.
On d1 and d1-Omni the input is read once for all of them, several times faster, and an answer can then shift with the questions asked beside it.
`independent(true)` answers each question as if asked alone, on every model; d3 is not trained to read questions together and answers each alone either way.
A score's `getValue()` is the expected level, the probability-weighted mean of the levels, as in LangChain4j's adapter and jinfer's `/v1/systemone`; `getNearestLevel()` is the likeliest level.
The response metadata carries the checkpoint's name, and a prompt naming another model is refused rather than answered by this one.

With the Spring Boot starter, select `spring.ai.model.decision=jinfer` and configure the model:

```properties
spring.ai.model.decision=jinfer
spring.ai.jinfer.decision.model=LiquidAI/d1-2B-GGUF:Q8_0
spring.ai.jinfer.decision.companions.media=LiquidAI/d1-2B-GGUF/mmproj-d1-2B-F16.gguf
spring.ai.jinfer.decision.independent=false
spring.ai.jinfer.decision.context-capacity=0
```

`context-capacity` bounds an input with its questions, a media row counting as a token: `0`, the default, is 4096 or the model's maximum when smaller, and `-1` is that maximum.
The `DecisionModel` bean is then ready to inject.

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

`JinferSpeechModel` implements Spring AI's `TextToSpeechModel` over a model made for it; an LFM3-Omni speaks through the chat model instead, see [Spoken replies](#spoken-replies).
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

## Transcription

`JinferTranscriptionModel` implements Spring AI's `TranscriptionModel`:

```java
try (var transcriber = JinferTranscriptionModel.builder()
        .model("mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf")
        .build()) {

    System.out.println(transcriber.transcribe(new FileSystemResource("speech.wav")));
}
```

The `Resource` may hold any format `jinfer-codecs` decodes.
The typed `transcribe(Path)` and `transcribe(byte[])` doors return the full jinfer
`Transcription`, with per-token spans, confidences and `words()` grouping.

## Complete examples

- [`jinfer-example-local-rag`](../jinfer-example-local-rag/README.md) keeps embedding, retrieval, vector storage and grounded chat in one Spring Boot JVM.
- [`jinfer-example-judge-advisor`](../jinfer-example-judge-advisor/README.md) uses a local, grammar-constrained judge to reject bad answers and drive Spring AI's self-refine loop.
