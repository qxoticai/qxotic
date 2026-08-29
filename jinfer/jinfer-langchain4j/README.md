<h1 align="center">jinfer for LangChain4j</h1>

<p align="center"><strong>AI in a jar, behind the LangChain4j API.</strong></p>

<p align="center">
  <a href="https://openjdk.org/projects/jdk/25/"><img src="https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white" alt="Java 25+"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache" alt="License: Apache 2.0"></a>
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
- **Accurate reporting.** Real token counts and finish reasons. Unsupported parameters fail
  instead of being silently ignored.
- **Shared weights.** `fork()` gives a second pipeline over one copy of the model.

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

## Add the provider

```xml
<dependencyManagement>
  <dependencies>
    <dependency>
      <groupId>com.qxotic</groupId>
      <artifactId>jinfer-bom</artifactId>
      <version>0.2.0</version>
      <type>pom</type>
      <scope>import</scope>
    </dependency>
    <dependency>
      <groupId>dev.langchain4j</groupId>
      <artifactId>langchain4j-bom</artifactId>
      <version>1.19.0</version>
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

Use `jinfer-models-all` instead of `jinfer-lfm2` for every model family. The `AiServices` examples
also need `dev.langchain4j:langchain4j`. The BOMs manage versions only. Without them, pin `0.2.0`
on each jinfer dependency and `1.19.0` on each LangChain4j one.

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
//DEPS com.qxotic:jinfer-bom:0.2.0@pom
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
        .contextLength(8192)      // 0 = the model's full context
        .temperature(0.7)
        .maxOutputTokens(1024)
        .thinking(false)          // reasoning scaffold off (models without one ignore it)
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

Use embeddings in the same JVM as chat:

```java
EmbeddingModel embeddings = JinferEmbeddingModel.builder()
        .model("Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0")
        .contextLength(2048)          // packing upper bound; 0 = the model's maximum
        .build();
```

Qwen3 supports LangChain4j's `dimensions` parameter. Query/document framing uses the standard
`EmbeddingInputType`:

```java
EmbeddingStoreIngestor.builder()
        .embeddingModel(embeddings)
        .embeddingStore(store)
        .embeddingInputType(EmbeddingInputType.DOCUMENT)   // card's document framing
        .build();
EmbeddingStoreContentRetriever.builder()
        .embeddingModel(embeddings)
        .embeddingStore(store)
        .embeddingInputType(EmbeddingInputType.QUERY)      // card's query framing
        .build();
```

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

## Speech synthesis

Turn text into WAV bytes:

```java
try (var speech = JinferSpeechModel.builder()
        .model("remixerdec/Inflect-Nano-v2-GGUF:Q8_0")
        .build()) {

    var audio = speech.synthesize("Hello from local Java inference.").audio();

    Files.write(Path.of("hello.wav"), audio.binaryData());
}
```

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

One instance is one serial pipeline. Fork for parallel generation over shared weights:

```java
try (Arena arena = Arena.ofShared()) {
    var path = ModelStore.standard().resolve("LiquidAI/LFM2.5-350M-GGUF:Q8_0");
    var loaded = Models.load(path, arena);
    try (var a = JinferChatModel.builder().model(loaded).contextLength(8192).build();
            var b = a.fork()) {     // second pipeline, shared weights, separate context
        // Run concurrent requests through a and b.
    }
}                                   // Closing the arena frees the shared weights.
```

The arena owns the weights and must outlive both pipelines.

## Notes

- One generation runs at a time per loaded model; concurrent `chat` calls queue fairly.
- The model name in responses is the GGUF file name; `FinishReason.TOOL_EXECUTION` is reported
  whenever the reply carries tool calls.
- Shaded JARs must preserve `ServiceLoader` entries. With Maven Shade, add
  `ServicesResourceTransformer`.

## License

Apache 2.0
