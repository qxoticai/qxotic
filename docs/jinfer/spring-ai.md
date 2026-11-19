---
sidebar_position: 3
---

# Spring AI integration

**Spring AI, fully local.**
In-process [Spring AI](https://spring.io/projects/spring-ai) chat, embeddings, reranking, decisions, speech synthesis and transcription over GGUF models, with a Spring Boot starter.
No server, no HTTP hop.

## Installation

These examples target jinfer **0.4.0**, Spring AI **2.1.0**, and Java **25**; the starter targets Spring Boot **4.2.0**.
For an unreleased checkout, first install the artifacts locally from the repository root:

```bash
mvn -pl jinfer/jinfer-bom,jinfer/jinfer-spring-ai-spring-boot-starter,jinfer/jinfer-gemma-embedding2 -am install -DskipTests
```

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jinfer-spring-ai</artifactId>
  <version>0.4.0</version>
</dependency>
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jinfer-lfm2</artifactId>
  <version>0.4.0</version>
</dependency>
```

The provider loads nothing by itself: add one model family (`jinfer-lfm2` here), or `jinfer-models-all` for every family.
Spring Boot applications use `jinfer-spring-ai-spring-boot-starter` instead of `jinfer-spring-ai`, see [Spring Boot](#spring-boot) and the [module README](https://github.com/qxoticai/qxotic/blob/main/jinfer/jinfer-spring-ai/README.md).

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

`model(String)` accepts a model ref (`owner/repo:quant`, or `modelscope.cn/...` for another source).
A URL is refused; the error shows the reference to use instead.
`modelPath(Path)` is the local form and never touches the network.
Companions use the same split: `companion(capability, String)` accepts a model reference; `companionPath(capability, Path)` accepts a local file.
Spring Boot properties remain path-or-ref strings and the autoconfiguration routes them to the matching method.

A bare repository reference follows llama.cpp and selects `Q4_K_M`.
Jinfer's best-supported quant is `Q8_0`, so the examples specify it explicitly.

Context capacity defaults to 4096 tokens, or the model's maximum when smaller, for chat, embedding and reranking alike.

A `JinferChatOptions` value on the prompt overrides per request:

```java
ChatResponse response = model.call(new Prompt(
        new UserMessage("Answer in one sentence."),
        JinferChatOptions.builder().maxTokens(64).seed(42L).build()));
```

A reply is a message of parts, in the model's order: the reasoning of a thinking model is a `ReasoningPart` (`getReasoning()`), the answer its text (`getText()`), then the tool calls.
Send the reply back as it is to continue the conversation; the reasoning replays with it.
`stream(Prompt)` returns a cancellable stream of deltas, each a text or a reasoning part, and a final response with finish reason, tool calls, and usage; `MessageAggregator` and `ChatClient` rebuild the whole message from them.
Disposing the subscription cancels generation.

Tools use `ToolCallingChatOptions`; compile tool classes with `-parameters` so Spring can read their parameter names.
Structured output uses `outputSchema` or `ChatClient.entity(..., EntityParamSpec::useProviderStructuredOutput)`.
The schema is enforced by the decoder.
`useProviderStructuredOutput` does not add schema instructions to the prompt; describe the task and expected fields yourself.

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

A message is a list of parts in the order written, so text and media may interleave: `UserMessage.builder().part(TextPart.of("Before:")).part(MediaPart.of(image)).part(TextPart.of("After?"))` reaches the model in that order.
Media bytes are decoded locally, then projected into embeddings.
`videoSampler` configures frame selection.
jinfer never fetches media URLs on an inference path.

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
LiquidAI's voice-chat prompt, "Respond with interleaved text and audio." as the system message, shapes short spoken answers; a text-only reply under it can stop mid-sentence, so it belongs with requests that speak.
There is no voice option: LFM3-Omni's spoken chat replies come in its one chat voice, whatever the system prompt asks for.
The chat voice speaks English, French and German clearly; Spanish and Italian come out accented and Japanese and Chinese partly garbled, so ask for text in those.
For speech in a set language, LFM3-Omni's text-to-speech instruction takes the language explicitly (see [jinfer-lfm3](https://github.com/qxoticai/qxotic/blob/main/jinfer/jinfer-lfm3/README.md#text-to-speech)).

## Prompt caching

Three separate controls:

- `retainedSessions(n)`: keep up to `n` finished states in RAM for append-only reuse
- `withCachedPrompt(messages, tools)`: define a reusable, content-addressed prefix
- `promptCache(path)` / `saveCachedPrompts(path)`: mount or export a persisted catalog

```java
JinferChatModel base = JinferChatModel.builder()
        .modelPath(gguf)
        .retainedSessions(2)
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

The catalog is model-identity checked.
An edited prompt reuses only its matching prefix; a miss recomputes and never changes the answer.
Cached media projection is shared by cached-prompt views.
Cached-prompt views share their base model's lifetime; close the base, not each view.

## Embeddings and reranking

### EmbeddingGemma 2 text

Alongside `jinfer-spring-ai`, add the Gemma provider below instead of `jinfer-lfm2` for an embedding-only application.
Applications already using `jinfer-models-all` have this provider.

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jinfer-gemma-embedding2</artifactId>
  <version>0.4.0</version>
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
For titles and other tasks, use the [Gemma helpers](https://github.com/qxoticai/qxotic/blob/main/jinfer/jinfer-gemma-embedding2/README.md) through a raw request, without passing the formatted text back through a query/document-framing overload.

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
Spring AI's vector stores still refuse non-text documents at `add`, so media vectors come from the `EmbeddingModel` directly.

For a starter-managed embedding model, select `spring.ai.model.embedding=jinfer`.
Set `model`, `context-capacity=8192`, and `dimensions=256` under `spring.ai.jinfer.embedding`.
For media, set `spring.ai.jinfer.embedding.companions.media` to the projector path.

### Reranking

`JinferDocumentPostProcessor` reranks retrieved documents as a Spring AI `DocumentPostProcessor`.
Reranking needs a separate model provider, such as `jinfer-qwen3:0.4.0` for Qwen3-Reranker, and the Spring AI RAG API.
The starter brings in the RAG API; without it, add:

```xml
<dependency>
  <groupId>org.springframework.ai</groupId>
  <artifactId>spring-ai-rag</artifactId>
  <version>2.1.0</version>
</dependency>
```

```java
try (var reranker = JinferDocumentPostProcessor.builder()
        .model("mradermacher/Qwen3-Reranker-0.6B-GGUF:Q8_0")
        .topK(1)
        .build()) {

    List<Document> candidates = List.of(
            new Document("Use the password reset link on the sign-in page."),
            new Document("Standard shipping takes three business days."));
    List<Document> ranked = reranker.process(new Query("How do I reset my password?"), candidates);
    ranked.forEach(document -> System.out.println(document.getText()));
}
```

In a RAG application, pass the documents returned by your retriever to the reranker.
See the [full reranking example](https://github.com/qxoticai/qxotic/blob/main/jinfer/jinfer-spring-ai/README.md#reranking).

## Decisions

`JinferDecisionModel` implements Spring AI's `DecisionModel`.
A decision model, such as Liquid AI's d1 or vLLM Semantic Router's d3, answers yes/no, choice and score questions with probabilities in one forward pass, without generating text:

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
A score's `getValue()` is the expected level, the probability-weighted mean of the levels; `getNearestLevel()` is the likeliest level.
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

## Speech synthesis

`JinferSpeechModel` implements Spring AI's `TextToSpeechModel` over a model made for it; an LFM3-Omni speaks through the chat model instead, see [Spoken replies](#spoken-replies).
`call` returns a complete WAV file; `stream` emits ordered, headerless little-endian PCM16 chunks.

```java
try (var speech = JinferSpeechModel.builder()
        .model("remixerdec/Inflect-Nano-v2-GGUF:Q8_0")
        .companion("lexicon", "remixerdec/Inflect-Nano-v2-GGUF/lexicon.bin")
        .build()) {

    Files.write(Path.of("hello.wav"), speech.call("Hello from local Java inference."));
}
```

Kokoro takes its voice as a companion, the same way (`simonfxr/kokoro.cpp-GGUF/voices/kokoro-voice-af_heart.gguf` as `voice`); `espeak-ng` must be on `PATH`.
A request that names a `voice`, `model` or `format` this instance does not have is refused, never silently given the default.
In a WebFlux application, return the `stream` `Flux` from the handler; never block on an event-loop thread.

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
The typed `transcribe(Path)` and `transcribe(byte[])` doors return the full jinfer `Transcription`, with per-token spans, confidences and `words()` grouping.

## Spring Boot

Use `jinfer-spring-ai-spring-boot-starter`:

```yaml
spring:
  ai:
    model:
      chat: jinfer
    jinfer:
      chat:
        model: LiquidAI/LFM2.5-8B-A1B-GGUF:Q8_0
        context-capacity: 8192
        max-tokens: 512
```

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
Embeddings and decisions need explicit selection, so a chat-only app is never asked for an embedding or decision GGUF.
Add companions under `spring.ai.jinfer.chat.companions` (or `.embedding.companions`, `.decision.companions`), keyed by capability such as `media` or `speculation`.
`spring.ai.jinfer.chat.output-modalities=text,audio` makes a model with a voice answer in speech, see [Spoken replies](#spoken-replies).
Model sources may be remote references or local paths; model resolution happens at startup, and invalid configuration fails the boot.
For `mvn spring-boot:run`, pass the JVM flags above through the plugin's `jvmArguments` and set `optimizedLaunch` to `false` for full JIT compilation.
With a `MeterRegistry` in the application (Spring Boot Actuator, for example), the starter records Spring AI's `gen_ai` metrics for the chat, embedding, decision, speech and transcription models: token usage and operation timings.

The [translation table](index.md#the-same-knob-at-each-face) maps settings between Spring AI, LangChain4j, the CLI and the server.

### Conversation memory

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

## Lifetime and concurrency

One adapter is one serial pipeline.
For another, load the weights once into an arena you own (`Models.load`, `Models.loadEmbedder`, `Models.loadReranker`, `Models.loadDecisionModel`) and build each adapter with `model(loaded)`; adapters share the weights and nothing else.
Close adapters before closing that arena.

`JinferChatModel`, `JinferEmbeddingModel`, `JinferDocumentPostProcessor`, `JinferDecisionModel`, `JinferSpeechModel`, and `JinferTranscriptionModel` are `AutoCloseable`; Spring closes managed beans automatically.

Shaded jars must merge `META-INF/services` entries (e.g. Maven Shade's `ServicesResourceTransformer`) so architecture providers stay discoverable.
