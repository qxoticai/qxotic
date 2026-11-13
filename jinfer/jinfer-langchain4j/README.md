<h1 align="center">jinfer for LangChain4j</h1>

<p align="center"><strong>AI in a jar, behind the LangChain4j API.</strong></p>

<p align="center">
  <a href="https://openjdk.org/projects/jdk/25/"><img src="https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white" alt="Java 25+"></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache" alt="License: Apache 2.0"></a>
  <a href="https://www.graalvm.org/latest/reference-manual/native-image/"><img src="https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F" alt="GraalVM Native Image"></a>
</p>

## The mission

**AI sovereignty for the JVM.**

An existing LangChain4j application keeps its code. `AiServices`, `@Tool`, `TokenStream`,
`EmbeddingStoreIngestor` and the retrieval augmentor all work unchanged, without an API key. This
is a LangChain4j provider backed by the [jinfer](../README.md) engine, so the inference runs
end-to-end within the JVM, with no server, Python runtime or external process, and no prompt sent
to a third party.

**Fast local LLM inference for the JVM. Just a jar.**

## What it does

- **Full provider.** `ChatModel`, `StreamingChatModel`, `EmbeddingModel`, `ScoringModel` and
  speech, all implemented against weights loaded in the same process.
- **Constrained output.** Return a record from `AiServices` and the sampler is constrained to its
  schema, so the model cannot produce invalid JSON and there is no retry loop.
- **Tools, memory and agents.** `AiServices` runs its usual loop against a local model.
- **Multimodal.** Images, audio and video through standard `ImageContent`, `AudioContent` and
  `VideoContent`, decoded locally.
- **Retrieval.** Embeddings and ColBERT reranking in the same JVM as chat, with no vector service.
- **Decisions.** `DecisionModel` answers yes/no, choice and scale questions about text, fields, images, audio and video with probabilities, without generating text.
- **Accurate reporting.** Real token counts and finish reasons. Unsupported parameters fail
  instead of being silently ignored.
- **Shared weights.** Models built on one `Models.load` run as parallel pipelines over one copy
  of the model.

## Run the demos

Install [JBang](https://www.jbang.dev/), then, from a repository checkout:

```bash
cd jinfer/examples/scripts

jbang Chat.java "Invent a tiny language for talking to houseplants."
jbang Json.java "Grace Hopper, born 1906 in New York City."
jbang Narrate.java photo.jpg
jbang Detect.java street.jpg "person, bicycle, traffic light"
```

They stream text, constrain JSON during sampling, turn an image into a spoken WAV and draw
detections onto a PNG. Models download on first use. `Detect.java` uses a 12B vision model and asks
for a 16 GB heap; the rest are small. The [full gallery](../examples/scripts/README.md) also covers
speech, semantic search, reranking and prompt caching.

Every knob here has the same name on the other faces (CLI, server, Spring AI, Java API); the [translation table](https://github.com/qxoticai/qxotic/blob/main/docs/jinfer/index.md#the-same-knob-at-each-face) maps one to the other.

## Add the provider

These examples target jinfer **0.4.0-rc1**, LangChain4j **1.23.0**, and Java **25**.
For an unreleased checkout, first install the artifacts locally from the repository root:

```bash
mvn -pl jinfer/jinfer-bom,jinfer/jinfer-langchain4j,jinfer/jinfer-gemma-embedding2 -am install -DskipTests
```

```xml
<dependencyManagement>
  <dependencies>
    <dependency>
      <groupId>com.qxotic</groupId>
      <artifactId>jinfer-bom</artifactId>
      <version>0.4.0-rc1</version>
      <type>pom</type>
      <scope>import</scope>
    </dependency>
    <dependency>
      <groupId>dev.langchain4j</groupId>
      <artifactId>langchain4j-bom</artifactId>
      <version>1.23.0</version>
      <type>pom</type>
      <scope>import</scope>
    </dependency>
  </dependencies>
</dependencyManagement>

<dependencies>
  <dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>jinfer-langchain4j</artifactId>
  </dependency>
  <dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>jinfer-lfm2</artifactId>
  </dependency>
</dependencies>
```

Use `jinfer-models-all` instead of `jinfer-lfm2` for every model family.
The `AiServices` examples also need `dev.langchain4j:langchain4j`.
The BOMs manage versions only.
Without them, pin the versions `jinfer-bom` 0.4.0-rc1 lists (0.4.0-rc1 for `jinfer-langchain4j` and `jinfer-lfm2`) and `1.23.0` on each LangChain4j one.

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

```java
//JAVA 25
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//DEPS com.qxotic:jinfer-bom:0.4.0-rc1@pom
//DEPS com.qxotic:jinfer-langchain4j
//DEPS com.qxotic:jinfer-lfm2
//DEPS com.qxotic:jam-native com.qxotic:jam-vector
```

Java 25 is required.

## Chat

```java
try (var model = JinferChatModel.builder()
        .model("LiquidAI/LFM2.5-350M-GGUF:Q8_0")
        .build()) {

    System.out.println(model.chat("Explain virtual threads in one sentence."));
}
```

Use `.model("...")` for a model reference and `.modelPath(Path.of("model.gguf"))` for a local
file. Companions follow the same pattern with `.companion(...)` and `.companionPath(...)`.

A reference is `owner/repo[:quant]`, which Hugging Face carries. Name a host to reach another
source, as in `modelscope.cn/Qwen/Qwen3-0.6B-GGUF:Q8_0`. Examples pin `Q8_0`; a reference with no
quant follows llama.cpp and selects `Q4_K_M`.

## Parameters

Model defaults belong on the builder. Per-request overrides belong on `ChatRequest`, following
LangChain4j conventions.

```java
try (var model = JinferChatModel.builder()
        .modelPath(Path.of("models/LFM2.5-8B-A1B-Q8_0.gguf"))
        .contextCapacity(8192)      // 0 = the default, -1 = the model's full context
        .temperature(0.7)
        .maxOutputTokens(1024)
        .maxReasoningTokens(256)     // cap the think span; 8B-A1B always reasons, so thinking(false) is refused
        .seed(42L)                // deterministic sampling
        .build()) {

    ChatResponse response = model.chat(ChatRequest.builder()
            .messages(UserMessage.from("Explain BPE in two sentences."))
            .maxOutputTokens(128)     // this request only
            .build());

    System.out.println(response.aiMessage().text());
    System.out.println(response.tokenUsage());     // real token counts, not estimates
    System.out.println(response.finishReason());   // STOP | LENGTH | TOOL_EXECUTION
}
```

Stops, JSON response formats and `toolChoice=REQUIRED` are supported. Unsupported parameters fail
instead of being ignored.

`thinkingPolicy()` says how a checkpoint reasons.
`NONE` has no think span, `OPTIONAL` honours `thinking(false)`, and `ALWAYS` (LFM2.5-8B-A1B) has no non-thinking turn, so `thinking(false)` fails at `build()` with an `UnsupportedFeatureException` naming the remedy.
Reasoning always arrives separated in `AiMessage.thinking()`; `maxReasoningTokens(n)` caps the span on every policy (a budget of 0 makes the model answer first, which costs accuracy even on simple questions: prefer a small positive budget), and a per-request `JinferChatRequestParameters.maxReasoningTokens` wins over the builder's.
Reasoning costs latency before it costs anything else.
Hidden tokens decode at the same rate as visible ones, so a one-line question that spends 130 tokens thinking before a 9-token answer takes fifteen times longer than the answer alone - 10 seconds on a laptop that decodes at 13 tokens per second.
That is the model's own default: LFM2.5-2.6B opens its think span in the generation prompt itself, so every client pays those tokens, and `thinking(false)` closes a span the template opened.
`JinferTokenUsage.reasoningTokenCount()` reports the hidden part of every reply, so a slow call explains itself in the usage line (`output=138 (reasoning=129)`).
For a call that needs no reasoning - summarising a tool result, classifying, extracting - set `JinferChatRequestParameters.maxReasoningTokens(0)` on that request and keep the model's default for the rest.

## Structured output

Return a POJO from `AiServices`; jinfer constrains generation to its schema:

```java
record Person(String name, int age) {}

interface PersonExtractor {
    Person extract(String text);
}

Person p = AiServices.create(PersonExtractor.class, model)
        .extract("Johann is 42 years old and lives in Munich.");   // Person[name=Johann, age=42]
```

The sampler rejects tokens outside the generated schema. No parser retry loop is required.
The grammar cannot tell the model which fields exist, so one line describing the shape is appended to the last user message; `describeSchema(false)` on the builder leaves the prompt untouched when it already states the shape.

## Streaming

The streaming view shares the already-loaded weights:

```java
interface Assistant { TokenStream chat(String message); }

Assistant assistant = AiServices.create(Assistant.class, model.streaming());

assistant.chat("Tell me a haiku about rivers.")
        .onPartialResponse(System.out::print)
        .onError(Throwable::printStackTrace)
        .start();
```

## Tool calling

Use LangChain4j tools; `AiServices` runs the loop:

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

Multimodal models load their encoders from a companion model file, following llama.cpp's `mmproj`
convention. Media is decoded from caller-provided base64 data or local file URIs. jinfer does not
fetch media during inference.

```java
try (var gemma = JinferChatModel.builder()
        .model("unsloth/gemma-4-12b-it-GGUF:Q8_0")
        .companion("media", "unsloth/gemma-4-12b-it-GGUF/mmproj-F32.gguf")
        .build()) {

    ChatResponse seen = gemma.chat(UserMessage.from(
            ImageContent.from(Path.of("photo.png").toUri()),
            TextContent.from("What is in this picture?")));

    ChatResponse heard = gemma.chat(UserMessage.from(
            AudioContent.from(Path.of("recording.wav").toUri()),
            TextContent.from("Transcribe this recording.")));

    System.out.println(seen.aiMessage().text());
    System.out.println(heard.aiMessage().text());
}
```

The runnable [`Narrate.java`](../examples/scripts/Narrate.java) uses Gemma to describe an image,
then sends the description to a second model that writes a WAV file.
[`GemmaVideo.java`](../examples/GemmaVideo.java) samples video frames and sends them as timestamped
image content.

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
The streaming model delivers the speech a stretch at a time, as the engine's `Media.Audio` through `onUnmappedRawEvent`, the transcript through `onPartialResponse`, and the whole clip on the complete response:

```java
List<Media.Audio> stretches = new ArrayList<>();
omni.streaming().chat(request, new StreamingChatResponseHandler() {
    @Override public void onPartialResponse(String partial) { System.out.print(partial); }
    @Override public void onUnmappedRawEvent(Object event) { stretches.add((Media.Audio) event); }
    @Override public void onCompleteResponse(ChatResponse response) {
        try {
            Files.write(Path.of("reply.wav"), AudioCodec.wav(Media.Audio.concat(stretches)));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
    @Override public void onError(Throwable error) { error.printStackTrace(); }
});
```

An `AiService`'s `TokenStream` takes the same `onUnmappedRawEvent`.
A reply sent back in history carries its speech and replays exactly; sent back as `AudioContent.from(speech)`, the clip is heard by the model like any other audio input.
A spoken reply does not reason, so allowing audio turns `thinking` off, and it cannot offer tools or follow a grammar: such a request is refused in words.
There is no voice option: LFM3-Omni's spoken chat replies come in its one chat voice, whatever the system prompt asks for.
A language clause is a request the model may not honour: it can answer in that language, in English, or in text alone (Japanese replies came back as text in every run).
For speech in a set language, LFM3-Omni's text-to-speech instruction takes the language explicitly (see [jinfer-lfm3](../jinfer-lfm3/README.md#text-to-speech)).

## Multi-model agent

Two models can share one JVM: a tool-capable model runs LangChain4j's agent loop while a multimodal
model provides vision and audio as Java tools. The complete, tested example is
[`LocalAgentIT.java`](src/test/java/com/qxotic/jinfer/langchain4j/LocalAgentIT.java).

```java
agent.chat("Look at dashboard.png, listen to operator-note.wav, "
        + "then explain whether they describe the same incident.");
```

The agent decides when to call `lookAt(...)` and `listenTo(...)`. Both tools invoke the second
model directly. Pixels, samples, tool results and conversation state remain in the JVM.

## Embeddings

`JinferEmbeddingModel` runs embedding GGUFs in-process, including EmbeddingGemma 2, Qwen3-Embedding, and LFM2.5-Embedding.
Segments are packed into context-sized ragged batches: one forward pass embeds many segments.

### EmbeddingGemma 2 text

Alongside `jinfer-langchain4j`, add the Gemma provider below instead of `jinfer-lfm2` for an embedding-only application.
Its version comes from `jinfer-bom:0.4.0-rc1`; applications already using `jinfer-models-all` have this provider.

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
This higher-level wiring also needs `dev.langchain4j:langchain4j`, managed by the LangChain4j BOM above:

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
For titles and other tasks, format text with the [Gemma helpers](../jinfer-gemma-embedding2/README.md) and pass it through a raw call without also selecting QUERY/DOCUMENT, which would add a second prefix.

## Reranking

Add reranking after embedding retrieval:

```java
ScoringModel reranker = JinferScoringModel.builder()
        .model("mradermacher/Qwen3-Reranker-0.6B-GGUF:Q8_0")
        .build();

RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
        .contentRetriever(retriever)
        .contentAggregator(ReRankingContentAggregator.builder()
                .scoringModel(reranker)
                .minScore(0.5)
                .build())
        .build();
```

Try the smaller [Search.java](../examples/scripts/Search.java) and
[Rerank.java](../examples/scripts/Rerank.java) demos side by side.

## Decisions

A decision model, such as Liquid AI's d1 or vLLM Semantic Router's d3, answers questions with probabilities in one forward pass, without generating text:

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
A scale's `mean()` is the expected level, the probability-weighted mean of the levels, as in Spring AI's adapter and jinfer's `/v1/systemone`.
The response names the checkpoint it came from (`modelName()`), and a request naming another model is refused rather than answered by this one.
`listeners(...)` takes langchain4j `DecisionModelListener`s, notified of every request, response and error, for audit logs, metrics and traces.

## Speech synthesis

Turn text into WAV bytes:

```java
try (var speech = JinferSpeechModel.builder()
        .model("remixerdec/Inflect-Nano-v2-GGUF:Q8_0")
        .companion("lexicon", "remixerdec/Inflect-Nano-v2-GGUF/lexicon.bin")
        .build()) {

    var audio = speech.synthesize("Hello from local Java inference.").audio();

    Files.write(Path.of("hello.wav"), audio.binaryData());
}
```

Kokoro takes its voice as a companion, the same way; `espeak-ng` must be on `PATH`:

```java
JinferSpeechModel.builder()
        .model("simonfxr/kokoro.cpp-GGUF:Q8_0")
        .companion("voice", "simonfxr/kokoro.cpp-GGUF/voices/kokoro-voice-af_heart.gguf")
        .build();
```

A request that names a `voice` is refused rather than answered in this model's only voice.
An LFM3-Omni speaks through the chat model instead, see [Spoken replies](#spoken-replies).

## Speech recognition

`JinferTranscriptionModel` implements langchain4j's `AudioTranscriptionModel`:

```java
try (var transcriber = JinferTranscriptionModel.builder()
        .model("mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf")
        .build()) {

    System.out.println(transcriber.transcribe(Path.of("speech.wav")).text());
}
```

The typed doors return the full jinfer `Transcription`, with per-token spans, confidences
and `words()` grouping; the langchain4j `transcribe(AudioTranscriptionRequest)` door
returns its text. A request that carries a `prompt`, `temperature` or `language` is
refused rather than silently ignored.

## Cached prompts

Prefill a system prompt once, then reuse it:

```java
JinferChatModel base = JinferChatModel.builder().modelPath(gguf).build();

JinferChatModel support = base.withCachedPrompt(
        List.of(SystemMessage.from(SUPPORT_INSTRUCTIONS)), supportTools);
support.chat("How do I reset my password?");

base.saveCachedPrompts(Path.of("dist/personas.jkv"));

JinferChatModel base2 = JinferChatModel.builder()
        .modelPath(gguf)
        .promptCache(Path.of("dist/personas.jkv"))
        .build();
```

[`CachedPrompt.java`](../examples/scripts/CachedPrompt.java) reports how many prompt tokens were
restored.

## Parallel pipelines

One instance is one serial pipeline.
For parallel generation, build each pipeline on the same loaded weights:

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

The arena owns the weights and must outlive both pipelines.
They share the weights and nothing else: listeners, defaults and cached prompts are set per build.

## Notes

- One generation runs at a time per loaded model; concurrent `chat` calls queue fairly.
- The model name in responses is the GGUF file name; `FinishReason.TOOL_EXECUTION` is reported
  whenever the reply carries tool calls.
- Shaded JARs must preserve `ServiceLoader` entries. With Maven Shade, add
  `ServicesResourceTransformer`.

## License

Apache 2.0
