# jinfer-langchain4j

[langchain4j](https://github.com/langchain4j/langchain4j) chat models backed by [jinfer](../README.md): in-process CPU inference over local GGUF files.
No server, no HTTP - the model runs inside your JVM.
Prompts go through jinfer's hand-written, oracle-validated chat-template codecs (token-exact, injection-inert by construction); unported models fall back to a hardened render of their own embedded Jinja template.

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jinfer-langchain4j</artifactId>
  <version>0.1.0</version>
</dependency>
```

Run your JVM with jinfer's flags:

```
--enable-preview --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
```

## Chat

```java
ChatModel model = JinferChatModel.builder()
        .model("hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF:Q8_0")
        .build();

String answer = model.chat("What is the capital of France?");
```

`.model(String)` takes whatever you have: a local GGUF path, a hub ref, or a pasted browser URL.
A remote ref downloads at `build()` into the shared model cache (resumable, sha256-verified; warm builds cost zero requests, `JINFER_OFFLINE=1` forbids the network - see [models from the hub](../README.md#models-from-the-hub)).
`.modelPath(Path)` stays the never-network form.

## Parameters

Model-level defaults on the builder, per-request overrides on the `ChatRequest` - standard langchain4j semantics.

```java
ChatModel model = JinferChatModel.builder()
        .modelPath(Path.of("models/LFM2.5-8B-A1B-Q8_0.gguf"))
        .contextLength(8192)      // 0 = the model's full context
        .temperature(0.7)
        .topP(0.95)
        .maxOutputTokens(1024)
        .thinking(false)          // reasoning scaffold off (models without one ignore it)
        .seed(42)                 // deterministic sampling
        .build();

ChatResponse response = model.chat(ChatRequest.builder()
        .messages(UserMessage.from("Explain BPE in two sentences."))
        .maxOutputTokens(128)     // this request only
        .build());

System.out.println(response.aiMessage().text());
System.out.println(response.tokenUsage());     // real token counts, not estimates
System.out.println(response.finishReason());   // STOP | LENGTH | TOOL_EXECUTION
```

String stop sequences, JSON response format (grammar-constrained decoding), and `toolChoice=REQUIRED` are supported.
Tools compose with a JSON *schema* response format: one selection admits the model's own tool calls while visible text can only be the schema, so the agent may call first and must answer shaped.
Unsupported knobs (penalties, per-request `modelName`, tools with schemaless JSON format, `REQUIRED` with a response format) throw `UnsupportedFeatureException` instead of being silently ignored.

## Structured output

`AiServices` returning a POJO works unchanged - the provider advertises `RESPONSE_FORMAT_JSON_SCHEMA`, so langchain4j sends the schema instead of asking for JSON in prose.

```java
record Person(String name, int age) {}

interface PersonExtractor {
    Person extract(String text);
}

Person p = AiServices.create(PersonExtractor.class, model)
        .extract("Johann is 42 years old and lives in Munich.");   // Person[name=Johann, age=42]
```

A schema does two jobs here, and a local model needs both.
The grammar compiled from it masks the logits, so output that violates the schema is unrepresentable - not merely unlikely.
The schema is *also* stated to the model in the prompt, because a mask constrains shape and says nothing about meaning: a model that never saw the schema answers `{"name": "user_agent", "age": 42}` - valid, and wrong.
Hosted providers hide this behind models trained on a schema channel; a local GGUF has no such channel.

## Streaming

`streaming()` shares the already-loaded model - the GGUF is mapped once, and `blocking()` gets you back (a model built with `buildStreaming()` reaches `fork()`, `withCachedPrompt` and the token estimator through it).
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

In `AiServices`, the same model streams through the idiomatic `TokenStream` - including through the automatic tool loop:

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

## Tools, the langchain4j way

`JinferChatModel` is a regular `ChatModel`, so `AiServices` works unchanged - including automatic tool execution loops.
Tool schemas are welded into the prompt in the exact canonical JSON the model was trained on.

`AiServices` lives in the `dev.langchain4j:langchain4j` artifact (this module only needs `-core`).

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

## Images and audio (Gemma 4)

Multimodal models take their encoders from a sidecar GGUF (llama.cpp's `mmproj` convention).
Media is decoded locally (base64 or `file://` - this library never fetches over the network) and enters the model as embeddings, never as text.

```java
ChatModel gemma = JinferChatModel.builder()
        .model("hf.co/unsloth/gemma-4-12b-it-qat-GGUF:Q4_K_XL")
        .companion("media", "hf.co/unsloth/gemma-4-12b-it-qat-GGUF/mmproj-F32.gguf")
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

## A local multi-model agent

Two GGUFs, one JVM, zero cloud.
LFM2.5 (fast, tool-capable) is the brain running langchain4j's automatic tool loop; Gemma 4 (vision + audio) is its eyes and ears, exposed *as tools*.
The brain never sees pixels or samples - it delegates questions and reasons over the answers, with memory across turns.

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

A real run (both models on one desktop CPU, ~70 s for the whole session):

```
USER>  Look at /tmp/scene/sign.png and tell me the color of the TOP lamp of the traffic light.
  [tool] lookAt(/tmp/scene/sign.png, "the color of the TOP lamp of the traffic light")
AGENT> The color of the top lamp of the traffic light is red.

USER>  Now listen to /tmp/scene/memo.wav - is it speech, music, or something else?
  [tool] listenTo(/tmp/scene/memo.wav, "speech, music, or something else")
AGENT> ... the file contains elements the tool could not definitively classify ...

USER>  Summarize everything you observed for me, one line each.
AGENT> The top lamp of the traffic light is red.
       The memo.wav file's content is ambiguous, with the tool explaining the differences
       between speech and music but not definitively classifying it.
```

Note the last two answers: the recording was a pure sine tone - out of distribution for a speech-tuned audio encoder - and the agent reports the ambiguity instead of inventing content, then recalls both observations from memory.
The runnable version is `LocalAgentIT` in this module's tests.

## Embeddings

`JinferEmbeddingModel` runs an embedding GGUF (Qwen3-Embedding, LFM2.5-Embedding) in-process, so the whole RAG stack - vectors, store, chat - stays in one JVM with zero egress.
Segments are packed into context-sized ragged batches: one forward pass embeds many segments, so ingesting hundreds of chunks costs a handful of prefills, not hundreds.
Usage reports exact token counts.

```java
EmbeddingModel embeddings = JinferEmbeddingModel.builder()
        .model("hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0")
        .contextLength(2048)          // packing upper bound; 0 = the model's maximum
        .build();
```

Qwen3 is Matryoshka-trained, so langchain4j's standard `dimensions` request parameter selects any
width from 32 through the model's native width; the returned prefix is L2-normalized. Fixed-width
models such as LFM2.5 reject that parameter instead of silently slicing their vectors.

Retrieval-tuned embedders are trained with query/document framing (LFM2.5's `query: `/`document: ` pair, Qwen3's instructed query), and embedding bare text instead silently degrades retrieval.
The provider speaks langchain4j's own vocabulary for this - `EmbeddingInputType` - so the framework knobs are all it takes:

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

Typeless traffic (plain `embed`/`embedAll`) embeds raw text as given - a one-time stderr note points at the knobs when the model is prefix-trained.

## Reranking

`JinferScoringModel` runs a reranker GGUF (Qwen3-Reranker, LFM2 ColBERT) in-process: a `ScoringModel` for langchain4j's `ReRankingContentAggregator`, the standard second stage after embedding retrieval.

```java
ScoringModel reranker = JinferScoringModel.builder()
        .model("hf.co/Qwen/Qwen3-Reranker-0.6B-GGUF:Q8_0")
        .build();

RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
        .contentRetriever(retriever)                       // wide net from the embedder
        .contentAggregator(ReRankingContentAggregator.builder()
                .scoringModel(reranker)
                .minScore(0.5)
                .build())                                  // precise cut from the reranker
        .build();
```

The runnable version is `RerankRetrievalIT` in this module's tests.

## Cached prompts

A cached prompt is paid for once and cheap forever after: `withCachedPrompt` prefills the prefix
(system prompt, tools, few-shot, even images) into an in-memory KV block tree and returns a model
view whose requests restore it instead of recomputing it - users only pay for their own data.
Content-addressed (no names), memory-only by default, byte-identical output to the uncached path.

```java
JinferChatModel base = JinferChatModel.builder().modelPath(gguf).build();

// prefill once; every chat on the view pays only the user's text
JinferChatModel support = base.withCachedPrompt(
        List.of(SystemMessage.from(SUPPORT_INSTRUCTIONS)), supportTools);
support.chat("How do I reset my password?");

// several prompts share one tree - common prefixes are stored once
JinferChatModel sales = base.withCachedPrompt(
        List.of(SystemMessage.from(SALES_INSTRUCTIONS)), salesTools);

// optional persistence: freeze everything into one artifact...
base.saveCachedPrompts(Path.of("dist/personas.jkv"));

// ...and mount it in the next process: re-declaring a stored prompt costs zero prefill
JinferChatModel base2 = JinferChatModel.builder()
        .modelPath(gguf)
        .promptCache(Path.of("dist/personas.jkv"))        // read-only, model-checked
        .build();
JinferChatModel support2 = base2.withCachedPrompt(
        List.of(SystemMessage.from(SUPPORT_INSTRUCTIONS)), supportTools);  // instant
```

The model retains one completed conversation by default, so an append-only follow-up can resume its
live state. Set `.retainSessions(0)` to close the state after every request; each subsequent request
then starts with a fresh state and may still restore matching blocks. Values above one retain that
many recent conversations. This is bounded process-local acceleration, not conversation identity:
jinfer matches the rendered prompt content, so reconstructing equivalent messages still resumes.

Cached-prompt views are immutable and composable (`withCachedPrompt` on a view branches on its
prefix). `promptCache(path)` mounts exactly one read-only artifact; a missing, incompatible, or
wrong-model artifact fails `build()` rather than silently running cold.
A view's tools are its DEFAULT tool set, request over defaults like every other parameter: a request stating the same set (what AiServices does) serves from the cache, a different set (or `toolChoice NONE`) serves correctly at full prefill - byte-identical output either way, with a one-time stderr warning naming the override.
Every response accounts for the cache: `((JinferTokenUsage) response.tokenUsage()).cachedInputTokens()` is the read, `servedFrom()` the tier - 0 and `FRESH` on a view mean you are paying full prefill, and the warning says why.
An edited prompt matches to the divergence point and pays only the tail; a wrong-model artifact fails at `build()`.
Requires a model with a native template codec (the Jinja fallback makes no prefix-stability promise).

## Parallel pipelines

One instance is one serial pipeline; concurrent calls queue fairly on it.
For real parallelism, load the weights once into YOUR arena and fork - every builder has a `model(loaded)` seam and every model a `fork()`:

```java
try (Arena arena = Arena.ofShared()) {
    var loaded = Models.load(ModelStore.standard().resolve("hf.co/...:Q4_K_M"), arena);
    var a = JinferChatModel.builder().model(loaded).contextLength(8192).build();
    var b = a.fork();               // second pipeline, same weights, a context's price
    // ... concurrent chat on a and b ...
    a.close(); b.close();
}                                   // the owner frees the weights, at a brace
```

The block structure is the ownership story: your arena outlives every instance built on it.
A sequential violation is caught fail-fast - a safety canary at the forward pass throws `IllegalStateException` on freed weights - while freeing the arena DURING a request is a data race and can still crash the VM.
`fork()` on a model that loaded its own weights refuses with that exact recipe - it frees its weights at `close()`, and a fork would dangle.
The same seam and `fork()` exist on `JinferEmbeddingModel` (via `Models.loadEmbedder`) and `JinferScoringModel` (via `Models.loadReranker`).

## Notes

- One generation runs at a time per loaded model; concurrent `chat` calls queue fairly.
- The model name in responses is the GGUF file name; `FinishReason.TOOL_EXECUTION` is reported whenever the reply carries tool calls.
- Shaded/fat-jar consumers need Maven Shade's `ServicesResourceTransformer` (the architecture ports register via `ServiceLoader`).
