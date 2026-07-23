# langchain4j-jinfer

[langchain4j](https://github.com/langchain4j/langchain4j) chat models backed by [jinfer](../README.md): in-process CPU inference over local GGUF files.
No server, no HTTP - the model runs inside your JVM.
Prompts go through jinfer's hand-written, oracle-validated chat-template codecs (token-exact, injection-inert by construction); unported models fall back to a hardened render of their own embedded Jinja template.

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>langchain4j-jinfer</artifactId>
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
        .modelPath(Path.of("models/LFM2.5-8B-A1B-Q8_0.gguf"))
        .build();

String answer = model.chat("What is the capital of France?");
```

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

Unsupported knobs (`topK`, penalties, string stop sequences, JSON response format, `toolChoice=REQUIRED`) throw `UnsupportedFeatureException` instead of being silently ignored.

## Streaming

`streaming()` shares the already-loaded model - the GGUF is mapped once.
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
        .modelPath(Path.of("models/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"))
        .mediaProjector(Path.of("models/mmproj-F32.gguf"))   // vision + audio encoders
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
    final ChatModel gemma;   // gemma-4-12B + mmproj, built with mediaProjector(...)

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

## Notes

- One generation runs at a time per loaded model; concurrent `chat` calls queue fairly.
- The model name in responses is the GGUF file name; `FinishReason.TOOL_EXECUTION` is reported whenever the reply carries tool calls.
- Shaded/fat-jar consumers need Maven Shade's `ServicesResourceTransformer` (the architecture ports register via `ServiceLoader`).
