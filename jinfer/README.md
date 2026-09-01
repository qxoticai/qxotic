<h1 align="center">jinfer</h1>

<p align="center"><strong>AI in a jar</strong></p>

<p align="center">
  <a href="https://openjdk.org/projects/jdk/25/"><img src="https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white" alt="Java 25+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache" alt="License: Apache 2.0"></a>
  <a href="https://www.graalvm.org/latest/reference-manual/native-image/"><img src="https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F" alt="GraalVM Native Image"></a>
</p>

**A fast, local AI inference engine for the JVM.**

`jinfer` runs chat, vision, audio, embeddings, reranking and speech models end-to-end within the JVM: weights stay local, everything happens in one
process, no sidecar server, Python runtime, ONNX or HTTP hop involved. Add a
dependency, call a builder, and the model is loaded into the heap and freed when it is closed.

## What it does

- **In-process.** The model loads into the running JVM. There is no separate service to deploy or
  keep running between requests.
- **Modalities.** Chat and streaming, tool calling, vision, audio, video, embeddings, reranking and speech synthesis, all from one engine.
- **Constrained output.** Compiled grammars mask logits during sampling, so the model cannot emit
  output that violates the schema. There is no parse-and-retry loop.
- **Framework bindings.** [LangChain4j](jinfer-langchain4j/README.md) and
  [Spring AI](jinfer-spring-ai/README.md) providers, plus an OpenAI-compatible server.
- **Performance.** Prompt caching that survives restarts, MTP speculative decoding, Matryoshka
  embeddings and hand-tuned kernels from [JAM](../jam), falling back to a portable Vector API
  path when JAM is absent.
- **Native image.** `make native` produces a self-contained binary with millisecond startup.
- **Apache 2.0.** No API key and no quota. Observability is local JFR events.

## Run the demos

Install [JBang](https://www.jbang.dev/). It resolves the dependencies, so there is nothing to
build first:

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

Add one API binding and one model family (`jinfer-models-all` grabs every architecture):

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

Then generate:

```java
try (var model = JinferChatModel.builder()
        .model("LiquidAI/LFM2.5-350M-GGUF:Q8_0")
        .build()) {

    System.out.println(model.chat("Explain virtual threads in one sentence."));
}
```

A model reference is `owner/repo[:quant]`, which Hugging Face carries. Name a host to reach
another source, as in `modelscope.cn/Qwen/Qwen3-0.6B-GGUF:Q8_0`. Use `modelPath(...)` for a file
already on disk.

Run with `--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED`, and add
`jam-native` at runtime scope for the fast matmul path.

## Examples

The snippets below use the LangChain4j binding. The [Spring AI](jinfer-spring-ai/README.md)
binding covers the same features.

**Streaming.**

```java
interface Assistant { TokenStream chat(String message); }

Assistant assistant = AiServices.create(Assistant.class, model.streaming());

assistant.chat("Tell me a haiku about rivers.")
        .onPartialResponse(System.out::print)
        .onError(Throwable::printStackTrace)
        .start();
```

**Structured output.** The sampler is constrained to the schema, so the result cannot come back
malformed:

```java
record Person(String name, int age, String city) {}

interface PersonExtractor { Person extract(String text); }

Person p = AiServices.create(PersonExtractor.class, model)
        .extract("Johann is 42 and lives in Munich.");   // Person[name=Johann, age=42, city=Munich]
```

To constrain the output to a specific shape instead of a POJO, pass a GBNF grammar:

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
    String weather(String city) { return "18C, sunny in " + city; }
}

interface WeatherAssistant { String chat(String message); }

WeatherAssistant assistant = AiServices.builder(WeatherAssistant.class)
        .chatModel(model)
        .tools(new Weather())
        .build();

assistant.chat("What's the weather in Paris?");   // calls weather("Paris"), then answers
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


**Speech synthesis.**

```java
try (var speech = JinferSpeechModel.builder()
        .model("remixerdec/Inflect-Nano-v2-GGUF:Q8_0")
        .build()) {

    var audio = speech.synthesize("Hello from local Java inference.").audio();

    Files.write(Path.of("hello.wav"), audio.binaryData());
}
```

## Choose an API

| Use case | Start here                                                                                                |
|----------|-----------------------------------------------------------------------------------------------------------|
| LangChain4j applications | [`jinfer-langchain4j`](jinfer-langchain4j/README.md)                                                      |
| Spring AI / Spring Boot | [`jinfer-spring-ai`](jinfer-spring-ai/README.md) (`jinfer-spring-ai-spring-boot-starter`)                 |
| Terminal and HTTP | [CLI and OpenAI-compatible server](#cli-and-server)                                                       |
| Single-file demos | [JBang demos](examples/scripts/README.md)                                                                 |
| Full applications | [Local RAG](jinfer-example-local-rag/README.md) · [Judge advisor](jinfer-example-judge-advisor/README.md) |

## What runs

Architecture dispatch comes from the providers on the classpath. Add an artifact to support a
family:

| Family | Capabilities | Artifact |
|--------|--------------|----------|
| Gemma 4 | chat, vision, audio, MTP | `jinfer-gemma4` |
| Qwen 3 / 3.5 | chat, embeddings, reranking, vision, MTP | `jinfer-qwen3`, `jinfer-qwen35` |
| LFM 2.5 | chat, embeddings, ColBERT reranking, vision | `jinfer-lfm2` |
| Laguna XS 2.1 | chat | `jinfer-laguna` |
| Llama family | chat (Llama, Ministral, MiniCPM, SmolLM, Granite) | `jinfer-llama` |
| gpt-oss · Nemotron-H · Maple | chat | `jinfer-gptoss`, `jinfer-nemotronh`, `jinfer-maple` |
| Inflect | speech synthesis | `jinfer-inflect2` |

Supported quantizations: Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, MXFP4, NVFP4, plus dense F32/F16/BF16.
`Q8_0` is the best-supported quant. A reference with no `:quant` follows llama.cpp and selects
`Q4_K_M`. Custom architectures plug in through `ModelProvider`.

## CLI and server

```bash
mvn -pl jinfer/jinfer-cli -am package -DskipTests

java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  -jar jinfer/jinfer-cli/target/jinfer.jar \
  --model LiquidAI/LFM2.5-350M-GGUF:Q8_0 --chat
```

Swap `--chat` for `--prompt "..."` (one-shot) or `--server --port 54154`. The server implements
`/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`, `/v1/tokenize`,
`/health` and Prometheus `/metrics`, so any OpenAI client can use it. Loopback is the default;
non-loopback binding requires `--api-key`. Multimodal models attach their projector with
`--mmproj <clip.gguf>`. Run `--help` for the complete contract.

## GraalVM Native image

```bash
make -C jinfer native
./jinfer/jinfer --model ./model.gguf --chat
```

One self-contained binary, millisecond startup. Requires GraalVM Native Image 25.0.3+.

## Documentation

The [documentation](https://qxotic.ai/docs/jinfer) covers the topics this README omits: hub
model references and download knobs, prompt caching internals,
embedding and reranking framing, parallel pipelines over shared weights, JFR observability, and
benchmarking with [`jinfer-bench`](jinfer-bench/README.md).

Part of [Quixotic AI](../README.md) project umbrella, an open stack for local AI on the JVM. `jinfer` is the inference engine; the
layers under it include [jam](../jam) (quantized matmul), [jota](../jota) (tensors),
[toknroll](../toknroll) (tokenizers) and [gguf](../gguf) (model files).
