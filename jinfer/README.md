<h1 align="center">jinfer</h1>

<p align="center"><strong>AI, in a jar</strong></p>

<p align="center">
  <a href="https://openjdk.org/projects/jdk/25/"><img src="https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white" alt="Java 25+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache" alt="License: Apache 2.0"></a>
  <a href="https://www.graalvm.org/latest/reference-manual/native-image/"><img src="https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F" alt="GraalVM Native Image"></a>
</p>

`jinfer` stands for "**J**VM **Infer**ence": a low-level AI engine for the JVM.  
No sidecar process, Docker container, Python, ONNX or HTTP requests involved.  
AI on the JVM, just a Maven dependency away.

## Highlights

- **Multi-modal support.** Vision, audio, video, embeddings for RAG, text-to-speech.
- **Supports popular Java AI frameworks.** [LangChain4j](jinfer-langchain4j/README.md) and
  [Spring AI](jinfer-spring-ai/README.md) providers and an OpenAI-compatible server.
- **Top performance.** Efficient prompt caching, speculative decoding, Matryoshka embeddings and optional hand-tuned native kernels from [JAM](../jam), with a performant Vector API fallback.
- **Constrained generation.** Models can only generate tokens that follow the schema.
- **First-class support for GraalVM's Native Image.** Self-contained binaries with millisecond startup.


## Supported models

| Family | Capabilities | Artifact |
|--------|--------------|----------|
| Google Gemma 4 | chat, vision, audio, MTP | `jinfer-gemma4` |
| Liquid AI LFM 2.5 | chat, vision, embeddings, reranking | `jinfer-lfm2` |
| OpenAI gpt-oss  Nemotron-H | chat | `jinfer-gptoss` |
| Poolside Laguna XS 2.1 | chat | `jinfer-laguna` |
| Meta Llama 3+ | chat | `jinfer-llama` |
| IBM Granite 4.1+ | chat | `jinfer-llama` |
| Mistral Ministral 3 | chat | `jinfer-llama` |
| Hugging Face SmolLM 3 | SmolLM | `jinfer-llama` |
| inflectionAI Ling 3 | chat | `jinfer-bailingmoe3` |
| MiniCPM 5 | chat | `jinfer-llama` |
| Alibaba Qwen 3 | embeddings, reranking | `jinfer-qwen3` |
| Alibaba Qwen 3.5+ | chat, vision, MTP | `jinfer-qwen35` |
| NVIDIA Nemotron | chat | `jinfer-nemotronh` |
| Owen Song's Inflect | speech synthesis | `jinfer-inflect2` |
| Kokoro 82M | speech synthesis | `jinfer-kokoro` |

Supported quantizations: `Q4_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `MXFP4` and dense `F32`, `F16`, `BF16`.  
Jinfer recommends the `Q8_0` quant, top-quality with good performance.

Kokoro also needs a voice GGUF companion and `espeak-ng` or `espeak` on `PATH`.
Inflect models pronounce from their `lexicon.bin`, with `espeak-ng` covering unknown words when it is installed.
Both expose their front end as `phonemizer()`, and `synthesize` takes the phoneme ids directly, the way a language model takes tokens.

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

Then generate:

```java
try (var model = JinferChatModel.builder()
        .model("LiquidAI/LFM2.5-350M-GGUF:Q8_0")
        .build()) {
    System.out.println(model.chat("What is the answer to the ultimate question of life, the universe, and everything?"));
}
```

A model reference is defined as `[provider.com/]owner/repository[/path][@revision][:quant]`, downloaded once from Hugging Face and cached.  
Supports other model providers/hosts e.g. `modelscope.cn/Qwen/Qwen3-0.6B-GGUF:Q8_0`.  
Use `.modelPath(Path modelPath)` to specify a model file already on disk.

Run with `--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED`, and add
`jam-native` at runtime scope to accelerate matrix multiplications.

## Examples

The snippets below use the [LangChain4j](jinfer-langchain4j) integration. The [Spring AI](jinfer-spring-ai/README.md) integration cover the same features.

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

A `.jkv` is content-addressed twice over.
Its blocks are keyed by the exact prompt tokens, so two programs share a cache only when they frame the prompt identically; a script that assembles its own history misses one written by the CLI and prefills from scratch, reporting nothing restored.
The artifact itself is keyed by the model file, every attached companion, and - for a model that projects media - that modality's decoder and projector plan; opening it under a different identity fails loudly rather than serving the wrong KV.
That identity is a printable line (`jinfer-cache/1 model=sha256:... companion:media=sha256:... imageDecoder=imageio imagePlan="..."`), recorded in the artifact; a refused open shows the artifact's line and the current load's and names the first field that differs.
The digest an artifact stores is the line's SHA-256, so `printf %s "<line>" | sha256sum` reproduces it.
`jinfer cache-info file.jkv` prints everything the file says about itself with no model in hand: format, identity and digest, both commit slots and which one serves, and every block with its span, size, parent and checksum - all checksums verified.
A JVM and a native image resolve different media decoders, so a media model's cache carries between them only when one is pinned with `-Djinfer.imageDecoder` or `-Djinfer.audioDecoder`.
Text-only and speculation-only caches carry between builds as they are.


**Speculative decoding.** A model that ships a draft head (Gemma 4, Qwen 3.5) attaches it as the
`speculation` companion and then verifies several drafted tokens per forward pass:

```java
try (var gemma = JinferChatModel.builder()
        .model("unsloth/gemma-4-E2B-it-GGUF:Q8_0")
        .companion("speculation", "unsloth/gemma-4-E2B-it-GGUF/MTP/mtp-gemma-4-E2B-it-Q8_0.gguf")
        .build()) { ... }
```

This is a speed-up only when the draft head guesses well, and that depends on the text, not the model.
Measured on Gemma 4 E2B Q8_0 on a 16-core CPU: lists 2.0x, code 1.7x, prose 0.9x; chat prose on a Q4_K_M checkpoint fell to 0.4x.
Every draft costs a full forward pass to verify, so below roughly one accepted draft in three the head is pure overhead.
Attach it for code, JSON, lists and grammar-constrained output; leave it off for conversation.
The accepted-draft ratio is reported per response (the `jinfer.Speculation` JFR event, the `mtp` cell of `jinfer-bench`), so the decision can be measured rather than guessed.

**Text-to-Speech.**

```java
try (var speech = JinferSpeechModel.builder()
        .model("remixerdec/Inflect-Nano-v2-GGUF:Q8_0")
        .build()) {

    var audio = speech.synthesize("Hello from local Java inference.").audio();

    Files.write(Path.of("hello.wav"), audio.binaryData());
}
```

## CLI and server

```bash
mvn -pl jinfer/jinfer-cli -am package -DskipTests

java \
  --add-modules jdk.incubator.vector \
  -jar jinfer/jinfer-cli/target/jinfer.jar \
  --model LiquidAI/LFM2.5-350M-GGUF:Q8_0 \
  --chat
```

Swap `--chat` for `--prompt "..."` (one-shot) or `--server --port 54154`. The server implements
`/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`, `/v1/tokenize`,
`/v1/detokenize`, `/health` and Prometheus `/metrics`, so any OpenAI client can use it. Loopback is the default;
non-loopback binding requires `--api-key`. Multimodal models attach their projector with
`--mmproj <clip.gguf>`. Run `--help` for the complete contract.

## Testing

`mvn test` runs the default suite; it needs no model files and no network.
The model-backed suites are tagged `integration` and skip unless the model is in a local cache:

```bash
mvn -pl jinfer/jinfer-lfm2 test -Dsurefire.excludedGroups= -Dgroups=integration
```

`scripts/download-models.sh` fetches them into `../models` next to the checkout, where the suites look by default.
`JINFER_MODELS` points them at another cache, and `TestModels` in the testkit names the file each suite wants when it skips.

## GraalVM Native image

```bash
make -C jinfer native
./bin/jinfer --model ./model.gguf --chat
```

One self-contained binary, instant startup. Requires GraalVM Native Image 25.0.3+.
