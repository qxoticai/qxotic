# Jinfer JBang demos

[![Java 25+](https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white)](https://openjdk.org/projects/jdk/25/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache)](../../LICENSE)

**Local LLM inference inside your JVM. No server. No Python. No external processes.**

## Run

Every demo is one self-contained Java file. Its JBang header declares Java 25, the Jinfer BOM, the
required model provider and the optional runtime backends:

```java
//JAVA 25
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//DEPS com.qxotic:jinfer-bom:0.3.1@pom
//DEPS com.qxotic:jinfer-langchain4j com.qxotic:jinfer-llama
//DEPS com.qxotic:jam-native com.qxotic:jam-vector
```

Install [JBang](https://www.jbang.dev/), then run a script from a repository checkout:

```bash
cd jinfer/examples/scripts
jbang Chat.java "Explain HTTP/3 in two sentences."
```

Models download on first use and remain in the Jinfer cache. Each source file declares its defaults
near the top and accepts alternative references through its trailing arguments.

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

### Support triage

```bash
jbang Triage.java "I was charged twice and I want my money back today."
```

A decision model (Liquid's d1) answers routing, refund and urgency questions in one forward pass,
no text generated, and the script prints the probability behind each answer.

```text
team     billing    0.9960
         billing    0.9960
         shipping   0.0021
         technical  0.0010
         account    0.0009
refund   yes        0.9993
urgency  Today      0.8662  (mean 2.12, median Today)

sure enough to auto-route: true
```

## More examples

| Task | Command | Result |
|---|---|---|
| Speech synthesis | `jbang Speak.java "Hello from Java."` | Writes `hello.wav` |
| Speech recognition | `jbang Transcribe.java speech.wav` | Prints the transcript, word timing on stderr |
| Semantic search | `jbang Search.java "what causes coffee bitterness?"` | Ranks documents by embedding similarity |
| Reranking | `jbang Rerank.java "what causes coffee bitterness?"` | Scores each query and document pair |
| Prompt caching | `jbang CachedPrompt.java` | Reports restored prompt tokens |
| Logic scoring | `jbang Logic.java` | Checks constrained answers against exact solutions |
| Decisions | `jbang Triage.java "..."` | Routes a support ticket, with probabilities |

The scripts include `jam-native` and `jam-vector`. Jinfer selects the native backend when supported
and otherwise uses the Java Vector backend.
