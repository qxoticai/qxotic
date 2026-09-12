<h1 align="center">jinfer</h1>

<p align="center"><strong>AI, in a jar</strong></p>

<div align="center">
  <a href="https://openjdk.org/projects/jdk/25/"><img src="https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white" alt="Java 25+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache" alt="License: Apache 2.0"></a>
  <a href="https://www.graalvm.org/latest/reference-manual/native-image/"><img src="https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F" alt="GraalVM Native Image"></a>

`jinfer` stands for "**J**VM **Infer**ence": a low-level AI inference engine for the JVM.  
No sidecar process, Docker containers, Python, ONNX or HTTP requests involved.  
AI on the JVM, just a Maven dependency away.

</div>

## Highlights

- **Multimodal support.** Vision, audio, video, embeddings for RAG, text-to-speech.
- **Supports popular Java AI frameworks.** [LangChain4j](jinfer-langchain4j/README.md) and
  [Spring AI](jinfer-spring-ai/README.md) providers and an [OpenAI-compatible server](./jinfer-server).
- **Top performance.** Efficient prompt caching, speculative decoding, Matryoshka embeddings and optional hand-tuned native kernels from [JAM](../jam), with a performant Vector API fallback.
- **Constrained generation.** Models can only emit tokens that follow the specified schema.
- **First-class support for GraalVM's Native Image.** Self-contained binaries with millisecond startup.

## Supported architectures

| Family | Capabilities | Artifact |
|--------|--------------|----------|
| Google Gemma 4 | chat, vision, audio, MTP | `jinfer-gemma4` |
| Liquid AI LFM 2.5 | chat, vision, embeddings, reranking | `jinfer-lfm2` |
| OpenAI gpt-oss | chat | `jinfer-gptoss` |
| Poolside Laguna XS 2.1 | chat | `jinfer-laguna` |
| JetBrains Mellum 2 | chat | `jinfer-mellum` |
| Meta Llama 3+ | chat | `jinfer-llama` |
| IBM Granite 4.1+ | chat | `jinfer-llama` |
| Mistral AI Ministral 3 | chat | `jinfer-llama` |
| Hugging Face SmolLM3 | chat | `jinfer-llama` |
| inclusionAI Ling 3 | chat | `jinfer-bailingmoe3` |
| OpenBMB MiniCPM5 | chat | `jinfer-llama` |
| Alibaba Qwen 3 | embeddings, reranking | `jinfer-qwen3` |
| Alibaba Qwen 3.5+ | chat, vision, MTP | `jinfer-qwen35` |
| NVIDIA Nemotron-H | chat | `jinfer-nemotronh` |
| Owen Song's Inflect | speech synthesis | `jinfer-inflect2` |
| Kokoro | speech synthesis | `jinfer-kokoro` |

Supported quantizations: `Q4_0`, `Q5_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `MXFP4` and the dense `F32`, `F16`, `BF16`.  
Jinfer recommends `Q8_0` for its balance of quality and performance.

## Run the demos

The demos and examples use [JBang](https://www.jbang.dev/).

```bash
cd jinfer/examples/scripts

jbang Chat.java "Invent a tiny language for talking to houseplants."   # streaming text
jbang Json.java "Ada Lovelace, born 1815 in London."                   # schema-perfect JSON
jbang Narrate.java photo.jpg                                           # vision, then speech
jbang Detect.java street.jpg "person, bicycle, traffic light"          # annotated PNG
```

Start with `Chat.java`, which downloads a 1B model. `Detect.java` uses a 12B vision model, so
expect a longer download. The [full gallery](examples/scripts/README.md) also covers semantic
search, reranking, logic puzzles and prompt-cache accounting.

## Use it from Java

Import the BOM once:

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
  </dependencies>
</dependencyManagement>
```

Add one API binding and one model family (`jinfer-models-all` for all model architectures):

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jinfer-langchain4j</artifactId>
</dependency>
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jinfer-lfm2</artifactId>
</dependency>
```

To spawn and use a model:

```java
try (var model = JinferChatModel.builder()
        .model("LiquidAI/LFM2.5-350M-GGUF:Q8_0")
        .build()) {
    System.out.println(model.chat("What is the answer to the ultimate question of life, the universe, and everything?"));
}
```

A model reference is defined as `[provider.com/]owner/repository[@revision][/path][:quant]`, downloaded once from Hugging Face and cached.  
Other providers and hosts are supported, for example `modelscope.cn/Qwen/Qwen3-0.6B-GGUF:Q8_0`.
Use `.modelPath(Path modelPath)` to specify a model file already on disk.

Run with `--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED`, and add
`jam-native` at runtime scope to accelerate matrix multiplications.  
If native libraries cannot be used/loaded, use `jam-vector` instead to accelerate matrix multiplications, this provides a pure Java execution, end-to-end.

## Examples

The snippets below use the [LangChain4j](jinfer-langchain4j) integration. The [Spring AI](jinfer-spring-ai/README.md) integration covers the same features.
For the `AiServices` examples, also add `dev.langchain4j:langchain4j:1.19.0` to your dependencies.

**Streaming.**

```java
interface Assistant { TokenStream chat(String message); }

Assistant assistant = AiServices.create(Assistant.class, model.streaming());

assistant.chat("Tell me a haiku about rivers.")
        .onPartialResponse(System.out::print)
        .onError(Throwable::printStackTrace)
        .start();
```

**Structured output.** The generation is constrained to the schema, so the result cannot come back
malformed:

```java
record Person(String name, int age, String city) {}

interface PersonExtractor { Person extract(String text); }

Person p = AiServices.create(PersonExtractor.class, model)
        .extract("Johann is 42 and lives in Munich."); // Person[name=Johann, age=42, city=Munich]
```

To constrain the output to a specific shape instead of a POJO, pass a [GBNF grammar](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md):

```java
var response = model.chat(ChatRequest.builder()
        .messages(UserMessage.from("Extract the person as JSON: " + text))
        .parameters(JinferChatRequestParameters.builder().grammar(GRAMMAR).build())
        .build());
```

**Tool calling.**

```java
class Weather {
    @Tool("Current weather for a city")
    String weather(@P("city") String city) { return "18C, sunny in " + city; }
}

interface WeatherAssistant { String chat(String message); }

WeatherAssistant assistant = AiServices.builder(WeatherAssistant.class)
        .chatModel(model)
        .tools(new Weather())
        .build();

assistant.chat("What's the weather in Zurich?");   // calls weather("Zurich"), then answers
```

**Vision.** Multimodal models attach their encoder as a companion, following llama.cpp's `mmproj`
convention:

```java
try (var gemma = JinferChatModel.builder()
        .model("unsloth/gemma-4-12b-it-GGUF:Q8_0")
        .companion("media", "unsloth/gemma-4-12b-it-GGUF/mmproj-F32.gguf")
        .build()) {

    var answer = gemma.chat(UserMessage.from(
            ImageContent.from(Path.of("photo.png").toUri()),
            TextContent.from("What is in this picture?")));

    System.out.println(answer.aiMessage().text());
}
```

Media is decoded locally and projected into embeddings. `jinfer` does not fetch media during
inference.

**Embeddings and reranking.** No vector service or reranking endpoint is required.

```java
EmbeddingModel embeddings = JinferEmbeddingModel.builder()
        .model("Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0")
        .build();

ScoringModel reranker = JinferScoringModel.builder()
        .model("mradermacher/Qwen3-Reranker-0.6B-GGUF:Q8_0")
        .build();
```

**Prompt caching.** Prefill a long system prompt once and reload it after a restart:

```java
JinferChatModel support = base.withCachedPrompt(List.of(SystemMessage.from(INSTRUCTIONS)), tools);

support.chat("How do I reset my password?");   // the instructions are already in the KV cache
base.saveCachedPrompts(Path.of("personas.jkv"));
```

**Speculative decoding.** Models that ship a draft head (e.g. Gemma 4, Qwen 3.5) can attach it as the
`speculation` companion:

```java
try (var gemma = JinferChatModel.builder()
        .model("unsloth/gemma-4-E2B-it-GGUF:Q8_0")
        .companion("speculation", "unsloth/gemma-4-E2B-it-GGUF/MTP/mtp-gemma-4-E2B-it-Q8_0.gguf")
        .build()) { ... }
```

This provides a speed-up only when the draft head guesses well, and that depends on the text, not the model.
Measured on Gemma 4 E2B Q8_0 on a 16-core CPU: lists 2.0x, code 1.7x, prose 0.9x; chat prose on a Q4_K_M checkpoint fell to 0.4x.

**Text-to-speech.**

```java
try (var speech = JinferSpeechModel.builder()
        .model("remixerdec/Inflect-Nano-v2-GGUF:Q8_0")
        .companion("lexicon", "remixerdec/Inflect-Nano-v2-GGUF/lexicon.bin")
        .build()) {

    var audio = speech.synthesize("Hello from local Java inference.").audio();

    Files.write(Path.of("hello.wav"), audio.binaryData());
}
```

## Chat CLI

To test different models, a simple CLI is bundled, can chat with all the supported models. 

```bash
mvn -pl jinfer/jinfer-cli -am package -DskipTests

java \
  --add-modules jdk.incubator.vector \
  -jar jinfer/jinfer-cli/target/jinfer.jar \
  --model LiquidAI/LFM2.5-350M-GGUF:Q8_0 \
  --chat
```

## OpenAI-compatible server

A simple OpenAI-compatible server is also provided.  
Multimodal models can attach their audio/image projector with `--mmproj <clip.gguf>`. Pass `--help` for more details.

```bash
mvn -pl jinfer/jinfer-cli -am package -DskipTests

java \
  --add-modules jdk.incubator.vector \
  -jar jinfer/jinfer-cli/target/jinfer.jar \
  --model LiquidAI/LFM2.5-2.6B-GGUF:Q8_0 \
  --context-capacity 65536 \
  --server
```

The server runs by default at `localhost:54154`, to verify it works:
```shell
curl -s http://127.0.0.1:54154/v1/chat/completions \
    -d '{"messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

## GraalVM Native Image

```bash
make -C jinfer native
./bin/jinfer --model ./model.gguf --chat
```

One self-contained binary, instant startup. Requires GraalVM Native Image 25.0.3+.
