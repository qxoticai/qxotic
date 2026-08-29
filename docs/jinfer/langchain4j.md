---
sidebar_position: 2
---

# LangChain4j integration

**LangChain4j, fully local.** [LangChain4j](https://github.com/langchain4j/langchain4j) models backed by jinfer: in-process CPU inference over local GGUF files. No server, no HTTP hop, no API keys.

Ports use jinfer's hand-written, oracle-validated chat-template codecs; unported models fall back to a hardened render of their embedded Jinja template.

## Installation

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jinfer-langchain4j</artifactId>
  <version>0.2.0</version>
</dependency>
```

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

`.model(String)` accepts a model ref (`owner/repo:quant`, or `modelscope.cn/...` for another source); a local path is
`.modelPath(Path)`. Ref downloads happen at `build()` into the shared cache (resumable,
sha256-verified; warm builds make no request, `JINFER_OFFLINE=1` forbids the network).

Companions use the same split: `.companion(capability, String)` accepts a model reference;
`.companionPath(capability, Path)` accepts a local file.

A bare repository reference follows llama.cpp and selects `Q4_K_M`. Jinfer's best-supported quant
is `Q8_0`, so the examples specify it explicitly.

## Parameters

Builder values are model defaults. `ChatRequest` values override per request.

```java
ChatModel model = JinferChatModel.builder()
        .modelPath(Path.of("models/LFM2.5-8B-A1B-Q8_0.gguf"))
        .contextLength(8192)      // 0 = the model's full context
        .temperature(0.7)
        .topP(0.95)
        .maxOutputTokens(1024)
        .thinking(false)          // reasoning off; models without it ignore it
        .seed(42)                 // deterministic sampling
        .build();

ChatResponse response = model.chat(ChatRequest.builder()
        .messages(UserMessage.from("Explain BPE in two sentences."))
        .maxOutputTokens(128)     // this request only
        .build());

System.out.println(response.aiMessage().text());
System.out.println(response.tokenUsage());     // real token counts
System.out.println(response.finishReason());   // STOP | LENGTH | TOOL_EXECUTION
```

Stop sequences, JSON response format (grammar-constrained decoding), and `toolChoice=REQUIRED` are supported. Unsupported knobs throw `UnsupportedFeatureException` instead of being ignored.

## Structured output

`AiServices` returning a POJO works unchanged. The provider advertises `RESPONSE_FORMAT_JSON_SCHEMA`, so LangChain4j sends the schema.

```java
record Person(String name, int age) {}

interface PersonExtractor {
    Person extract(String text);
}

Person p = AiServices.create(PersonExtractor.class, model)
        .extract("Johann is 42 years old and lives in Munich.");   // Person[name=Johann, age=42]
```

A local model needs the schema for two jobs:

- The compiled grammar masks logits, so schema-violating output is unrepresentable, not merely unlikely.
- The schema is also stated in the prompt. A mask constrains shape, not meaning; hosted providers hide this behind models trained on a schema channel, which a local GGUF lacks.

## Streaming

`streaming()` shares the loaded model. Reasoning models stream on two lanes: content to `onPartialResponse`, thinking to `onPartialThinking`.

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

`JinferChatModel` is a regular `ChatModel`, so `AiServices` works unchanged, including automatic tool execution. Tool schemas are welded into the prompt in the canonical JSON the model was trained on.

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

## Images and audio (Gemma 4)

Multimodal models take their encoders from a sidecar GGUF (`mmproj`). Media is decoded locally (base64 or `file://`; the library never fetches over the network) and enters the model as embeddings, not text.

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

## A local multi-model agent

Two GGUFs, one JVM. LFM2.5 runs LangChain4j's tool loop; Gemma 4 (vision + audio) is exposed as tools.

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

`JinferEmbeddingModel` runs an embedding GGUF (Qwen3-Embedding, LFM2.5-Embedding) in-process, so vectors, store, and chat stay in one JVM.

Segments are packed into context-sized ragged batches: one forward pass embeds many segments.

```java
EmbeddingModel embeddings = JinferEmbeddingModel.builder()
        .model("Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0")
        .contextLength(2048)          // packing upper bound; 0 = the model's maximum
        .build();
```

Qwen3 is Matryoshka-trained; `dimensions` selects any width from 32 through the native width, and the returned prefix is L2-normalized. Fixed-width models such as LFM2.5 reject `dimensions` instead of slicing vectors.

Retrieval-tuned embedders use query/document framing (LFM2.5's `query: `/`document: ` pair, Qwen3's instructed query). Embedding bare text degrades retrieval. Set `EmbeddingInputType`:

```java
EmbeddingStoreIngestor.builder()
        .embeddingModel(embeddings)
        .embeddingStore(store)
        .embeddingInputType(EmbeddingInputType.DOCUMENT)   // document framing
        .build();

EmbeddingStoreContentRetriever.builder()
        .embeddingModel(embeddings)
        .embeddingStore(store)
        .embeddingInputType(EmbeddingInputType.QUERY)      // query framing
        .build();
```

Plain `embed`/`embedAll` embeds raw text; a one-time stderr note points at the knobs when the model is prefix-trained.

## Reranking

`JinferScoringModel` runs a reranker GGUF (Qwen3-Reranker, LFM2 ColBERT) as a `ScoringModel` for `ReRankingContentAggregator`.

```java
ScoringModel reranker = JinferScoringModel.builder()
        .model("Qwen/Qwen3-Reranker-0.6B-GGUF:Q8_0")
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

## Cached prompts

`withCachedPrompt` prefills a prefix (system prompt, tools, few-shot, even images) into an in-memory KV block tree and returns a view whose requests restore it instead of recomputing it. Content-addressed, memory-only by default, byte-identical output.

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

The model retains one completed conversation by default, so append-only follow-ups resume live state. `.retainSessions(0)` closes state after every request; higher values retain that many conversations. This is process-local acceleration, not identity: jinfer matches rendered prompt content.

Views are immutable and composable. `promptCache(path)` mounts exactly one read-only artifact; a missing, incompatible, or wrong-model artifact fails `build()`. An edited prompt matches to the divergence point and pays only the tail.

Read cache accounting from the response: `((JinferTokenUsage) response.tokenUsage()).cachedInputTokens()` and `servedFrom()`. Requires a native template codec; the Jinja fallback makes no prefix-stability promise.

## Parallel pipelines

One instance is one serial pipeline; concurrent calls queue. For parallel inference, load the weights once into a caller-owned arena and fork.

```java
try (Arena arena = Arena.ofShared()) {
    var loaded = Models.load(ModelStore.standard().resolve("...:Q4_K_M"), arena);
    var a = JinferChatModel.builder().model(loaded).contextLength(8192).build();
    var b = a.fork();               // second pipeline, same weights
    // ... concurrent chat on a and b ...
    a.close(); b.close();
}                                   // the owner frees the weights at a brace
```

The arena must outlive every instance built on it. Freed weights throw `IllegalStateException` at the forward pass. Freeing the arena during a request is a data race and can crash the VM.

`fork()` on a model that loaded its own weights refuses. The same seam and `fork()` exist on `JinferEmbeddingModel` (`Models.loadEmbedder`) and `JinferScoringModel` (`Models.loadReranker`).

## Notes

- One generation runs at a time per loaded model.
- The response model name is the GGUF file name. `FinishReason.TOOL_EXECUTION` reports whenever the reply carries tool calls.
- Shaded jars need Maven Shade's `ServicesResourceTransformer` (architecture ports register via `ServiceLoader`).
