---
sidebar_position: 2
---

# LangChain4j integration

**LangChain4j, fully local.**
[LangChain4j](https://github.com/langchain4j/langchain4j) models backed by jinfer: in-process CPU inference over local GGUF files.
No server, no HTTP hop, no API keys.

The provider covers `ChatModel`, `StreamingChatModel`, `EmbeddingModel`, `ScoringModel`, `DecisionModel`, speech synthesis and transcription.
Ports use jinfer's hand-written, oracle-validated chat-template codecs; unported models fall back to a hardened render of their embedded Jinja template.

## Installation

These examples target jinfer **0.4.0-rc1**, LangChain4j **1.23.0**, and Java **25**.
For an unreleased checkout, first install the artifacts locally from the repository root:

```bash
mvn -pl jinfer/jinfer-bom,jinfer/jinfer-langchain4j,jinfer/jinfer-gemma-embedding2 -am install -DskipTests
```

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jinfer-langchain4j</artifactId>
  <version>0.4.0-rc1</version>
</dependency>
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jinfer-lfm2</artifactId>
  <version>0.4.0-rc1</version>
</dependency>
```

The provider loads nothing by itself: add one model family (`jinfer-lfm2` here), or `jinfer-models-all` for every family.
Importing `jinfer-bom` and `langchain4j-bom` manages these versions for you; see the [module README](https://github.com/qxoticai/qxotic/blob/main/jinfer/jinfer-langchain4j/README.md#add-the-provider).

Run the JVM with:

```text
--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
```

## Chat

```java
try (var model = JinferChatModel.builder()
        .model("LiquidAI/LFM2.5-8B-A1B-GGUF:Q8_0")
        .build()) {

    System.out.println(model.chat("What is the capital of France?"));
}
```

`.model(String)` accepts a model ref (`owner/repo:quant`, or `modelscope.cn/...` for another source); a local path is `.modelPath(Path)`.
Ref downloads happen at `build()` into the shared cache (resumable, sha256-verified; warm builds make no request, `JINFER_OFFLINE=1` forbids the network).

Companions use the same split: `.companion(capability, String)` accepts a model reference; `.companionPath(capability, Path)` accepts a local file.

A bare repository reference follows llama.cpp and selects `Q4_K_M`.
Jinfer's best-supported quant is `Q8_0`, so the examples specify it explicitly.

## Parameters

Builder values are model defaults.
`ChatRequest` values override per request.

```java
JinferChatModel model = JinferChatModel.builder()
        .modelPath(Path.of("models/LFM2.5-8B-A1B-Q8_0.gguf"))
        .contextCapacity(8192)      // 0 = the default, -1 = the model's full context
        .temperature(0.7)
        .topP(0.95)
        .maxOutputTokens(1024)
        .maxReasoningTokens(256)     // this model always reasons: cap the span instead of switching it off
        .seed(42L)                // deterministic sampling
        .build();

ChatResponse response = model.chat(ChatRequest.builder()
        .messages(UserMessage.from("Explain BPE in two sentences."))
        .maxOutputTokens(128)     // this request only
        .build());

System.out.println(response.aiMessage().text());
System.out.println(response.tokenUsage());     // real token counts
System.out.println(response.finishReason());   // STOP | LENGTH | TOOL_EXECUTION
```

Stop sequences, JSON response format (grammar-constrained decoding), and `toolChoice=REQUIRED` are supported.
Unsupported knobs throw `UnsupportedFeatureException` instead of being ignored.
Every knob has the same name on the other faces (CLI, server, Spring AI, Java API); the [translation table](index.md#the-same-knob-at-each-face) maps one to the other.

### Reasoning

`thinkingPolicy()` says how a checkpoint reasons.
`NONE` has no think span, `OPTIONAL` honours `thinking(false)`, and `ALWAYS` (LFM2.5-8B-A1B) has no non-thinking turn, so `thinking(false)` fails at `build()` with an `UnsupportedFeatureException` naming the remedy.
Reasoning always arrives separated in `AiMessage.thinking()`.
`maxReasoningTokens(n)` caps the span on every policy, and a per-request `JinferChatRequestParameters.maxReasoningTokens` wins over the builder's.
A budget of 0 makes the model answer first, which costs accuracy even on simple questions: prefer a small positive budget.

Hidden tokens decode at the same rate as visible ones, so reasoning costs latency before it costs anything else.
`JinferTokenUsage.reasoningTokenCount()` reports the hidden part of every reply, so a slow call explains itself in the usage line (`output=138 (reasoning=129)`).
For a call that needs no reasoning - summarising a tool result, classifying, extracting - set `JinferChatRequestParameters.maxReasoningTokens(0)` on that request and keep the model's default for the rest.

## Structured output

`AiServices` returning a POJO works unchanged.
The provider advertises `RESPONSE_FORMAT_JSON_SCHEMA`, so LangChain4j sends the schema.

```java
record Person(String name, int age) {}

interface PersonExtractor {
    Person extract(String text);
}

Person p = AiServices.create(PersonExtractor.class, model)
        .extract("Johann is 42 years old and lives in Munich.");   // Person[name=Johann, age=42]
```

A local model needs the schema for two jobs:

- The compiled grammar masks logits, so schema-violating output is unrepresentable, not merely unlikely, and no parser retry loop is required.
- A mask constrains shape, not meaning, so one line describing the schema's fields is appended to the last user message.

`describeSchema(false)` on the builder leaves the prompt untouched when it already states the shape.

## Streaming

`streaming()` shares the loaded model.
Reasoning models stream on two lanes: content to `onPartialResponse`, thinking to `onPartialThinking`.

```java
StreamingChatModel streaming = model.streaming();

streaming.chat("Tell me a haiku about rivers.", new StreamingChatResponseHandler() {
    @Override public void onPartialResponse(String token) { System.out.print(token); }
    @Override public void onPartialThinking(PartialThinking t) { /* reasoning lane */ }
    @Override public void onCompleteResponse(ChatResponse done) { System.out.println(); }
    @Override public void onError(Throwable error) { error.printStackTrace(); }
});
```

In `AiServices`, the same model streams through `TokenStream`, including the automatic tool loop:

```java
interface Assistant { TokenStream chat(String message); }

Assistant assistant = AiServices.builder(Assistant.class)
        .streamingChatModel(streaming)
        .build();

assistant.chat("Tell me a haiku about rivers.")
        .onPartialResponse(System.out::print)
        .onCompleteResponse(done -> System.out.println())
        .onError(Throwable::printStackTrace)
        .start();
```

## Tools

`JinferChatModel` is a regular `ChatModel`, so `AiServices` works unchanged, including automatic tool execution.
Tool schemas are welded into the prompt in the canonical JSON the model was trained on.

`AiServices` lives in `dev.langchain4j:langchain4j`; this module needs only `-core`.

```java
class Weather {
    @Tool("Get current weather for a city")
    String weather(@P("city name") String city) {
        return "18C, sunny in " + city;
    }
}

interface Assistant {
    String chat(String message);
}

Assistant assistant = AiServices.builder(Assistant.class)
        .chatModel(model)
        .tools(new Weather())
        .chatMemory(MessageWindowChatMemory.withMaxMessages(20))
        .build();

assistant.chat("What's the weather in Paris?");   // calls Weather.weather, answers grounded
```

## Images, audio and video

Multimodal models take their encoders from a sidecar GGUF (`mmproj`).
Media is decoded locally (base64 or `file://`; the library never fetches over the network) and enters the model as embeddings, not text.

```java
ChatModel gemma = JinferChatModel.builder()
        .model("unsloth/gemma-4-12b-it-GGUF:Q8_0")
        .companion("media", "unsloth/gemma-4-12b-it-GGUF/mmproj-F32.gguf")
        .build();

ChatResponse seen = gemma.chat(ChatRequest.builder()
        .messages(UserMessage.from(
                ImageContent.from(base64Png, "image/png"),
                TextContent.from("What is in this picture?")))
        .build());

ChatResponse heard = gemma.chat(ChatRequest.builder()
        .messages(UserMessage.from(
                AudioContent.from(base64Wav, "audio/wav"),
                TextContent.from("Transcribe this recording.")))
        .build());
```

Video arrives as sampled frames sent as timestamped image content, see [`GemmaVideo.java`](https://github.com/qxoticai/qxotic/blob/main/jinfer/examples/GemmaVideo.java).

## Spoken replies

A model with a voice, LFM3-Omni with its projector attached, can answer in speech through the chat model.
What a reply may be made of is the `outputModalities` request parameter: unset, a reply is text alone and the model never switches into speech; `TEXT` and `AUDIO` let the model choose, steered by the prompt; `AUDIO` alone makes it speak.
The speech is the reply's `audios()`, one `audio/wav` clip beside the transcript:

```java
try (var omni = JinferChatModel.builder()
        .model("LiquidAI/LFM3-Omni-2B-GGUF:Q8_0")
        .companion("media", "LiquidAI/LFM3-Omni-2B-GGUF/mmproj-LFM3-Omni-2B-F16.gguf")
        .defaultRequestParameters(JinferChatRequestParameters.builder()
                .outputModalities(Modality.TEXT, Modality.AUDIO)
                .build())
        .build()) {

    AiMessage reply = omni.chat(
            SystemMessage.from("Respond with interleaved text and audio."),
            UserMessage.from("What is the capital of France?")).aiMessage();

    System.out.println(reply.text()); // the transcript
    for (Audio speech : reply.audios()) {
        Files.write(Path.of("reply.wav"), speech.binaryData());
    }
}
```

The same parameter on a request opens or closes the voice for that request alone.
The streaming model delivers the speech a stretch at a time, as the engine's `Media.Audio` through `onUnmappedRawEvent` (also on an `AiService`'s `TokenStream`), the transcript through `onPartialResponse`, and the whole clip on the complete response.
A spoken reply does not reason, so allowing audio turns `thinking` off, and it cannot offer tools or follow a grammar: such a request is refused in words.
There is no voice option: LFM3-Omni's spoken chat replies come in its one chat voice, whatever the system prompt asks for.
A language clause is a request the model may not honour: it can answer in that language, in English, or in text alone (Japanese replies came back as text in every run).
For speech in a set language, LFM3-Omni's text-to-speech instruction takes the language explicitly (see [jinfer-lfm3](https://github.com/qxoticai/qxotic/blob/main/jinfer/jinfer-lfm3/README.md#text-to-speech)).

## A local multi-model agent

Two GGUFs, one JVM.
LFM2.5 runs LangChain4j's tool loop; Gemma 4 (vision + audio) is exposed as tools.

```java
class Senses {
    final ChatModel gemma;   // gemma-4-12B + mmproj, built with companion("media", ...)

    @Tool("Look at an image file and answer a question about it")
    String lookAt(@P("absolute path of the image file") String path,
                  @P("what to look for or answer") String question) {
        return gemma.chat(ChatRequest.builder()
                .messages(UserMessage.from(
                        ImageContent.from(base64(path), "image/png"),
                        TextContent.from(question)))
                .build()).aiMessage().text();
    }

    @Tool("Listen to an audio file and answer a question about it")
    String listenTo(@P("absolute path of the audio file") String path,
                    @P("what to listen for") String question) { /* same shape, AudioContent */ }
}

interface Agent { String chat(String message); }

Agent agent = AiServices.builder(Agent.class)
        .chatModel(brain)                 // LFM2.5-8B
        .tools(new Senses(gemma))         // gemma-4-12B behind the tools
        .chatMemory(MessageWindowChatMemory.withMaxMessages(20))
        .build();
```

Runnable version: `LocalAgentIT`.

## Embeddings

`JinferEmbeddingModel` runs embedding GGUFs in-process, including EmbeddingGemma 2, Qwen3-Embedding, and LFM2.5-Embedding.
Segments are packed into context-sized ragged batches: one forward pass embeds many segments.

### EmbeddingGemma 2 text

Alongside `jinfer-langchain4j`, add the Gemma provider below instead of `jinfer-lfm2` for an embedding-only application.
Applications already using `jinfer-models-all` have this provider.

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jinfer-gemma-embedding2</artifactId>
  <version>0.4.0-rc1</version>
</dependency>
```

In a Maven project with these dependencies, save this as `GemmaText.java` in the project directory.
It takes the local path to `embeddinggemma-2-Q8_0.gguf` as its argument.
The checkpoint is available in [unsloth/embeddinggemma-2-GGUF](https://huggingface.co/unsloth/embeddinggemma-2-GGUF).
Text-only inference does not need the projector.

```java
import com.qxotic.jinfer.langchain4j.JinferEmbeddingModel;
import dev.langchain4j.model.embedding.request.EmbeddingInputType;
import dev.langchain4j.model.embedding.request.EmbeddingRequest;
import java.nio.file.Path;

public class GemmaText {
    public static void main(String[] args) {
        try (var embeddings = JinferEmbeddingModel.builder()
                .modelPath(Path.of(args[0]))
                .contextCapacity(8192)
                .dimensions(256)
                .build()) {
            var document = embeddings.embed(EmbeddingRequest.builder()
                    .input("A tabby cat sitting on a windowsill.")
                    .inputType(EmbeddingInputType.DOCUMENT)
                    .build()).embeddings().getFirst();
            var query = embeddings.embed(EmbeddingRequest.builder()
                    .input("a photo of a cat")
                    .inputType(EmbeddingInputType.QUERY)
                    .build()).embeddings().getFirst();
            System.out.printf("Document: %d, query: %d%n",
                    document.vector().length, query.vector().length);
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

`dimensions(256)` sets the default width and makes `dimension()` report 256; a request's own `dimensions` still wins.
Gemma's [model card](https://huggingface.co/google/embeddinggemma-2) lists widths of 768, 512, 256, and 128, and warns that 128 substantially reduces multimodal quality.
The adapter also accepts intermediate widths, which are not listed in the model card.
Returned shortened vectors are L2-normalized.
The builder's default, `dimensions(0)`, keeps the native width; unsupported widths are refused at build time.
Qwen3-Embedding accepts widths from 32 through its native width, while fixed-width models such as LFM2.5 refuse custom dimensions.
Keep the checkpoint, quantization, output width, and query/document framing consistent when building and searching an index.

`contextCapacity(8192)` selects Gemma's documented shared context, including task framing, boundary tokens, and any media positions.
Each input must fit; packing several inputs does not split an oversized document.
Keep the concrete adapter in try-with-resources, as above, so its state and owned weights are closed.

For an existing `store`, create the ingestor and retriever inside that same try block and use them before it closes.
This higher-level wiring also needs `dev.langchain4j:langchain4j:1.23.0`:

```java
EmbeddingStoreIngestor.builder()
        .embeddingModel(embeddings)
        .embeddingStore(store)
        .embeddingInputType(EmbeddingInputType.DOCUMENT)
        .build();

EmbeddingStoreContentRetriever.builder()
        .embeddingModel(embeddings)
        .embeddingStore(store)
        .embeddingInputType(EmbeddingInputType.QUERY)
        .build();
```

### EmbeddingGemma 2 media

Save this as `GemmaMedia.java` in the same project directory.
It takes three local paths: the model GGUF, `mmproj-BF16.gguf` from the same repository, and your PNG image.
The dependency set is the same as for text; the `media` companion enables image, audio, and video inputs.

```java
import com.qxotic.jinfer.langchain4j.JinferEmbeddingModel;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.model.embedding.request.EmbeddingRequest;
import java.nio.file.Path;

public class GemmaMedia {
    public static void main(String[] args) {
        try (var embeddings = JinferEmbeddingModel.builder()
                .modelPath(Path.of(args[0]))
                .companionPath("media", Path.of(args[1]))
                .contextCapacity(8192)
                .dimensions(256)
                .build()) {
            var photo = embeddings.embed(EmbeddingRequest.builder()
                    .input(ImageContent.from(Path.of(args[2]).toUri()))
                    .build()).embeddings().getFirst();
            System.out.println("Image: " + photo.vector().length);
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
As the model card says, media embeds without a prompt, so the input type changes nothing for an input that carries media.
Text and media parts in one `EmbeddingInput` embed as one vector, in order.
Without the projector, `supportedContentTypes()` is text only and LangChain4j refuses media before decoding it.
With a listener registered, LangChain4j (1.21 to 1.23) refuses an input that is media alone ("text cannot be null or blank", [langchain4j#6784](https://github.com/langchain4j/langchain4j/issues/6784)).
Adding a text part avoids the error, but the text then becomes part of the vector.

Plain `embed(String)`/`embedAll` embeds raw text; a one-time SLF4J warning points at the role options when the model is prefix-trained and a logging provider is installed.

The input types cover search, the one task LangChain4j has a word for.
For titles and other tasks, format text with the [Gemma helpers](https://github.com/qxoticai/qxotic/blob/main/jinfer/jinfer-gemma-embedding2/README.md) and pass it through a raw call without also selecting QUERY/DOCUMENT, which would add a second prefix.

## Reranking

`JinferScoringModel` runs a reranker GGUF (Qwen3-Reranker, LFM2 ColBERT) as a `ScoringModel` for `ReRankingContentAggregator`.

```java
ScoringModel reranker = JinferScoringModel.builder()
        .model("mradermacher/Qwen3-Reranker-0.6B-GGUF:Q8_0")
        .build();

RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
        .contentRetriever(retriever)                       // wide net from the embedder
        .contentAggregator(ReRankingContentAggregator.builder()
                .scoringModel(reranker)
                .minScore(0.5)
                .build())                                  // precise cut from the reranker
        .build();
```

Runnable version: `RerankRetrievalIT`.

## Decisions

`JinferDecisionModel` implements LangChain4j's `DecisionModel`.
A decision model, such as Liquid AI's d1 or vLLM Semantic Router's d3, answers yes/no, choice and scale questions with probabilities in one forward pass, without generating text:

```java
try (var model = JinferDecisionModel.builder()
        .model("LiquidAI/d1-2B-GGUF:Q8_0")
        .companion("media", "LiquidAI/d1-2B-GGUF/mmproj-d1-2B-F16.gguf")  // for images and video
        .build()) {

    DecisionResponse response = model.decide(DecisionRequest.builder()
            .input("I was charged twice for my order and I want my money back today.")
            .question("refund", YesNoQuestion.from("Is the customer asking for a refund?"))
            .question("team", ChoiceQuestion.from("Which team should handle this?",
                    List.of("billing", "shipping", "technical")))
            .question("urgency", ScaleQuestion.from("How urgent is this?",
                    List.of("Can wait", "This week", "Today", "Right now")))
            .build());

    response.yesNo("refund").probability();   // 0.99
    response.choice("team").value();          // "billing"
    response.scale("urgency").mean();         // 2.08
}
```

The input can be a text, fields or contents.
Fields are put to the model as JSON, the form d1 was trained on; a field holding contents contributes its text (or its media's kinds) to the JSON, and its media to the input, never as base64 in the prompt.
Contents take images, audio and video beside text; d1 reads images and video, d1-Omni also audio, d3 images.
A conversation (`inputMessages`) is not an input and is refused.

The questions of a request are taken the way the model was trained to, with no isolation between them guaranteed.
On d1 and d1-Omni the input is read once for all of them, several times faster, and an answer can then shift with the questions asked beside it.
`independent(true)` answers each question as if asked alone, on every model; d3 is not trained to read questions together and answers each alone either way.
A scale's `mean()` is the expected level, the probability-weighted mean of the levels.
The response names the checkpoint it came from (`modelName()`), and a request naming another model is refused rather than answered by this one.
`listeners(...)` takes LangChain4j `DecisionModelListener`s, notified of every request, response and error, for audit logs, metrics and traces.

## Speech synthesis

`JinferSpeechModel` turns text into WAV bytes:

```java
try (var speech = JinferSpeechModel.builder()
        .model("remixerdec/Inflect-Nano-v2-GGUF:Q8_0")
        .companion("lexicon", "remixerdec/Inflect-Nano-v2-GGUF/lexicon.bin")
        .build()) {

    var audio = speech.synthesize("Hello from local Java inference.").audio();

    Files.write(Path.of("hello.wav"), audio.binaryData());
}
```

Kokoro takes its voice as a companion, the same way (`simonfxr/kokoro.cpp-GGUF/voices/kokoro-voice-af_heart.gguf` as `voice`); `espeak-ng` must be on `PATH`.
A request that names a `voice` is refused rather than answered in this model's only voice.
An LFM3-Omni speaks through the chat model instead, see [Spoken replies](#spoken-replies).

## Speech recognition

`JinferTranscriptionModel` implements LangChain4j's `AudioTranscriptionModel`:

```java
try (var transcriber = JinferTranscriptionModel.builder()
        .model("mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf")
        .build()) {

    System.out.println(transcriber.transcribe(Path.of("speech.wav")).text());
}
```

The typed doors return the full jinfer `Transcription`, with per-token spans, confidences and `words()` grouping; the LangChain4j `transcribe(AudioTranscriptionRequest)` door returns its text.
A request that carries a `prompt`, `temperature` or `language` is refused rather than silently ignored.

## Cached prompts

`withCachedPrompt` prefills a prefix (system prompt, tools, few-shot, even images) into an in-memory KV block tree and returns a view whose requests restore it instead of recomputing it.
Content-addressed, memory-only by default, byte-identical output.

```java
JinferChatModel base = JinferChatModel.builder().modelPath(gguf).build();

// prefill once; every chat on the view pays only the user's text
JinferChatModel support = base.withCachedPrompt(
        List.of(SystemMessage.from(SUPPORT_INSTRUCTIONS)), supportTools);
support.chat("How do I reset my password?");

// several prompts share one tree; common prefixes are stored once
JinferChatModel sales = base.withCachedPrompt(
        List.of(SystemMessage.from(SALES_INSTRUCTIONS)), salesTools);

// persist everything into one artifact...
base.saveCachedPrompts(Path.of("dist/personas.jkv"));

// ...and mount it in the next process: re-declaring a stored prompt costs zero prefill
JinferChatModel base2 = JinferChatModel.builder()
        .modelPath(gguf)
        .promptCache(Path.of("dist/personas.jkv"))        // read-only, model-checked
        .build();
JinferChatModel support2 = base2.withCachedPrompt(
        List.of(SystemMessage.from(SUPPORT_INSTRUCTIONS)), supportTools);  // instant
```

The model retains one completed conversation by default, so append-only follow-ups resume live state.
`.retainedSessions(0)` closes state after every request; higher values retain that many conversations.
This is process-local acceleration, not identity: jinfer matches rendered prompt content.

Views are immutable and composable.
`promptCache(path)` mounts exactly one read-only artifact; a missing, incompatible, or wrong-model artifact fails `build()`.
An edited prompt matches to the divergence point and pays only the tail.

Read cache accounting from the response: `((JinferTokenUsage) response.tokenUsage()).cachedInputTokens()` and `servedFrom()`.
Requires a native template codec; the Jinja fallback makes no prefix-stability promise.

## Parallel pipelines

One instance is one serial pipeline; concurrent calls queue.
For parallel inference, load the weights once into a caller-owned arena and build each pipeline on them.

```java
try (Arena arena = Arena.ofShared()) {
    var path = ModelStore.standard().resolve("LiquidAI/LFM2.5-350M-GGUF:Q8_0");
    var loaded = Models.load(path, arena);
    try (var a = JinferChatModel.builder().model(loaded).contextCapacity(8192).build();
            var b = JinferChatModel.builder().model(loaded).contextCapacity(8192).build()) {
        // Run concurrent requests through a and b.
    }
}                                   // Closing the arena frees the shared weights.
```

The arena must outlive every instance built on it.
Freed weights throw `IllegalStateException` at the forward pass.
Freeing the arena during a request is a data race and can crash the VM.

Pipelines share the weights and nothing else: listeners, defaults and cached prompts are set per build.
To start a pipeline from prompts another one prefilled, `saveCachedPrompts(path)` on the first and `promptCache(path)` on the next.
The same `model(loaded)` seam exists on `JinferEmbeddingModel` (`Models.loadEmbedder`), `JinferScoringModel` (`Models.loadReranker`) and `JinferDecisionModel` (`Models.loadDecisionModel`).

## Notes

- One generation runs at a time per loaded model; concurrent `chat` calls queue fairly.
- The response model name is the GGUF file name.
- `FinishReason.TOOL_EXECUTION` reports whenever the reply carries tool calls.
- Shaded jars need Maven Shade's `ServicesResourceTransformer` (architecture ports register via `ServiceLoader`).
