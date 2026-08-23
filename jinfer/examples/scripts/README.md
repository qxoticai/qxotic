# Jinfer JBang demos

[![Java 25+](https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white)](https://openjdk.org/projects/jdk/25/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache)](../../LICENSE)

**Local LLM inference inside your JVM. No server. No Python. No external processes.**

## Run

Each script declares the Jinfer BOM, its API, the required model provider and optional runtime
backends:

```java
//JAVA 25
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//DEPS com.qxotic:jinfer-bom:0.1.0@pom
//DEPS com.qxotic:jinfer-langchain4j com.qxotic:jinfer-llama
//DEPS com.qxotic:jam-native com.qxotic:jam-vector
```

The demos use hand-picked providers: Llama for chat, Gemma for vision, Qwen3 for retrieval and
Inflect for speech. Use `jinfer-models-all` in a Maven application when classpath size does not
matter.

Install [JBang](https://www.jbang.dev/), then run a script from a repository checkout:

```bash
cd jinfer/examples/scripts
jbang Chat.java "Explain HTTP/3 in two sentences."
```

The scripts request Java 25 and download their default model on first use. Models remain in the
Jinfer cache. Pass another model reference as the final argument to override a default.

## Demos

### Streaming chat

```bash
jbang Chat.java "Invent a tiny language for talking to houseplants."
```

Tokens stream to the terminal as they are generated.

### Constrained JSON

```bash
jbang Json.java "Ada Lovelace, born 1815 in London, wrote the first algorithm."
```

```json
{"name": "Ada Lovelace", "year": 1815, "city": "London"}
```

The sampler rejects tokens that do not match the grammar.

### Vision and speech

```bash
jbang Narrate.java photo.jpg
```

Gemma describes the image, then Inflect synthesizes the description into `narration.wav`.

### Object detection

```bash
jbang Detect.java street.jpg "person, bicycle, traffic light"
```

The model returns normalized boxes. Java2D scales and paints them into `detected.png`.

| Script | Demonstrates | Model |
|---|---|---|
| `Chat.java` | streaming chat, token by token | Llama-3.2-1B |
| `Json.java` | grammar-constrained JSON | Llama-3.2-1B |
| `Speak.java` | text to speech from a 4 MB model | Inflect-Nano-v2 |
| `Narrate.java` | image to description to spoken WAV in one JVM | Gemma 4 E2B + Inflect |
| `Search.java` | semantic search, no vector database | Qwen3-Embedding-0.6B |
| `Rerank.java` | cross-encoder reranking, the second stage of RAG | Qwen3-Reranker-0.6B |
| `CachedPrompt.java` | prompt caching, with restored-token accounting | Llama-3.2-1B |
| `Detect.java` | object detection with boxes drawn on the image | Gemma 4 12B + projector |
| `Logic.java` | yes/no logic puzzles, scored by exact match | Llama-3.2-1B |

## Logic puzzle scoring

```bash
jbang Logic.java
```

Each puzzle describes truth-tellers and liars, then asks three yes/no questions. A GBNF grammar
restricts the response to a comma-separated answer. The script normalizes whitespace, checks the
answer directly and prints a final score. No judge model is involved.

## Semantic retrieval and reranking

```bash
jbang Search.java "what causes coffee bitterness?"
jbang Rerank.java "what causes coffee bitterness?"
```

`Search.java` ranks documents by embedding similarity. `Rerank.java` scores each query and document
together. Neither requires a vector database or reranking service.

## Prompt-cache reuse

```bash
jbang CachedPrompt.java
```

Each answer reports how many prompt tokens were restored from the cache.

## Notes

`Models.java` holds the shared model references. The scripts include `jam-native` and `jam-vector`;
Jinfer selects the native backend when supported and otherwise uses the Java Vector backend.
