# Local RAG with Spring AI

[![Java 25+](https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white)](https://openjdk.org/projects/jdk/25/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache)](../LICENSE)

This example runs retrieval-augmented generation with two models in one Spring Boot JVM. Inference
requires no model server or API key. Models download on first use and can run offline from the
cache afterward.

- Qwen3-Embedding creates vectors through `JinferEmbeddingModel`.
- A chat model answers through `JinferChatModel`.
- Spring AI connects them with `RetrievalAugmentationAdvisor` and `SimpleVectorStore`.

The sample corpus contains facts about store credit, shipping and warranties. The questions do not
contain those facts, so the answers exercise retrieval rather than model recall.

## Run

From the repository root, build the example and its reactor dependencies:

```bash
mvn -Pexamples -pl jinfer/jinfer-example-local-rag -am install -DskipTests
```

Then run the application from this directory:

```bash
export JINFER_CHAT_MODEL=hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF:Q8_0
export JINFER_EMBEDDING_MODEL=hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0
mvn spring-boot:run
```

## Example run

```text
>>> ingested 4 documents in 0.4s
>>> Q: How will I get my refund?
>>> A: Refunds are issued as store credit to your account.
>>> Q: I ordered something on Saturday. When does it ship?
>>> A: Weekend orders ship on Monday; orders before 3pm ship the next business day.
>>> Q: How long is the warranty on my appliance?
>>> A: Every appliance carries a two-year warranty covering parts and labor.
```

## Tests

`LocalRagIT` checks that answers contain facts from the corpus. It uses the repository's
`TestModels` cache lookup and runs only when the required models are available:

```bash
mvn test -Dsurefire.excludedGroups= -Dgroups=integration
```
