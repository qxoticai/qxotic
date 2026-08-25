# jinfer

[![Java 25+](https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white)](https://openjdk.org/projects/jdk/25/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache)](LICENSE)
[![GraalVM Native Image](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)

**Local LLM inference for the JVM. No server. No Python. No ONNX. Just a jar.**

jinfer runs large language models *inside* your Java process: chat, streaming, tool calling,
schema-perfect JSON, vision, audio, embeddings, reranking and speech synthesis. Models download
from Hugging Face on first use and stay cached. Vector API kernels provide the portable floor;
[JAM](../jam) accelerates matmul when present. Everything compiles to a single GraalVM native
binary.

## Why jinfer

- **In-process.** The model lives in your JVM. No sidecar, no HTTP hop, no Python to babysit.
- **One builder.** `JinferChatModel.builder().model("hf.co/…:Q8_0").build()` — that's the setup.
- **Speaks your framework.** Drop-in [LangChain4j](jinfer-langchain4j/README.md) and
  [Spring AI](jinfer-spring-ai/README.md) models, plus an OpenAI-compatible server for everything else.
- **Structured output that can't break the schema.** Compiled grammars mask logits — invalid JSON
  is unrepresentable, not merely unlikely.
- **Fast.** Prompt caching, Matryoshka embeddings, ColBERT reranking, MTP speculation, and
  hand-tuned SIMD matmul under the hood.

## See it in 30 seconds

Install [JBang](https://www.jbang.dev/), then from a repository checkout:

```bash
cd jinfer/examples/scripts

jbang Chat.java "Invent a tiny language for talking to houseplants."  # streaming text
jbang Json.java "Ada Lovelace, born 1815 in London."                  # constrained JSON
jbang Narrate.java photo.jpg                                         # vision and speech
jbang Detect.java street.jpg "person, bicycle, traffic light"         # annotated PNG
```

Models download on first use and remain in the jinfer cache. `Detect.java` uses a 12B vision
model; start with `Chat.java` for a small download. More demos — semantic search, reranking,
prompt-cache accounting — in the [gallery](examples/scripts/README.md).

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

Add one API and one model provider (`jinfer-models-all` grabs every architecture):

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

Generate:

```java
try (var model = JinferChatModel.builder()
        .model("hf.co/LiquidAI/LFM2.5-350M-GGUF:Q8_0")
        .build()) {
    System.out.println(model.chat("Explain virtual threads in one sentence."));
}
```

Run with `--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED`, and add
`jam-native` (runtime scope) for the fast matmul path. JBang users need three `//DEPS` lines —
see the [demo scripts](examples/scripts/README.md).

## Choose an API

| Use case | Start here |
|----------|------------|
| LangChain4j applications | [`jinfer-langchain4j`](jinfer-langchain4j/README.md) |
| Spring AI / Spring Boot | [`jinfer-spring-ai`](jinfer-spring-ai/README.md) (`jinfer-spring-ai-spring-boot-starter`) |
| Terminal and HTTP | [CLI and OpenAI-compatible server](#cli-and-server) |
| Single-file demos | [JBang gallery](examples/scripts/README.md) |
| Full applications | [Local RAG](jinfer-example-local-rag/README.md) · [Judge advisor](jinfer-example-judge-advisor/README.md) |

## CLI and server

```bash
mvn -pl jinfer/jinfer-cli -am package -DskipTests

java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
  -jar jinfer/jinfer-cli/target/jinfer.jar \
  --model hf.co/LiquidAI/LFM2.5-350M-GGUF:Q8_0 --chat
```

Swap `--chat` for `--prompt "..."` (one-shot) or `--server --port 54154`. The server implements
`/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`, `/v1/tokenize`,
`/health` and Prometheus `/metrics`. Loopback is the safe default; non-loopback binding requires
`--api-key`. Run `--help` for the complete contract.

## What runs

Architecture dispatch comes from the providers on the classpath — add an artifact, get a family:

| Family | Capabilities | Artifact |
|--------|--------------|----------|
| Gemma 4 | chat, vision, audio, MTP | `jinfer-gemma4` |
| Qwen 3 / 3.5 | chat, embeddings, reranking, vision, MTP | `jinfer-qwen3`, `jinfer-qwen35` |
| LFM 2.5 | chat, embeddings, ColBERT reranking, vision | `jinfer-lfm2` |
| Llama family | chat (Llama, Ministral, MiniCPM, SmolLM, Granite) | `jinfer-llama` |
| gpt-oss · Nemotron-H · Maple | chat | `jinfer-gptoss`, `jinfer-nemotronh`, `jinfer-maple` |
| Inflect | speech synthesis | `jinfer-inflect2` |

Supported quantizations: Q4_0 through Q8_0, the k-quants, MXFP4, NVFP4, plus dense F32/F16/BF16.
`Q8_0` is the best-supported path; a bare `hf.co/org/repo` reference follows llama.cpp and picks
`Q4_K_M`. Multimodal models attach their projector with `--mmproj <clip.gguf>`. Custom
architectures plug in through `ModelProvider`.

## Native image

```bash
make -C jinfer native
./jinfer/jinfer --model ./model.gguf --chat
```

One self-contained binary, millisecond startup. Requires GraalVM Native Image 25.0.3+.

## Going deeper

The [documentation](https://qxotic.ai/docs/jinfer) covers what this README deliberately omits:
hub model references and download knobs, prompt caching (`withCachedPrompt`, persisted `.jkv`
catalogs), embeddings and reranking framing, parallel pipelines, JFR observability, and
benchmarking with [`jinfer-bench`](jinfer-bench/README.md).

## Scope

jinfer performs inference. It does not train, fine-tune or quantize models.

## License

Apache 2.0
