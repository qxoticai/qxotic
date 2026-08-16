# jinfer-example-local-rag

Fully-local RAG in one JVM - two GGUFs, no server, no API key, no network.

- **Vectors**: Qwen3-Embedding (0.6B) via `JinferEmbeddingModel` - corpus chunks packed into
  context-sized ragged batches, one forward pass per group (~2k tokens/s on a 32-core CPU).
- **Chat**: any chat GGUF (e.g. LFM2.5-8B) via `JinferChatModel`, grounded by Spring AI's
  `RetrievalAugmentationAdvisor` over an in-memory `SimpleVectorStore`.

The corpus facts (store credit, next business day, two-year warranty) exist ONLY in the
documents - a grounded answer proves the retrieve-then-generate flow, not model memory.

## Run

```
export JINFER_CHAT_MODEL=/path/to/chat.gguf          # e.g. LFM2.5-8B-A1B-Q8_0.gguf
export JINFER_EMBEDDING_MODEL=/path/to/emb.gguf      # e.g. Qwen3-Embedding-0.6B-Q8_0.gguf
mvn spring-boot:run
```

## What a run looks like

```
>>> ingested 4 documents in 0.4s
>>> Q: How will I get my refund?
>>> A: Refunds are issued as store credit to your account.
>>> Q: I ordered something on Saturday - when does it ship?
>>> A: Weekend orders ship on Monday; orders before 3pm ship the next business day.
>>> Q: How long is the warranty on my appliance?
>>> A: Every appliance carries a two-year warranty covering parts and labor.
```

## Tests

`mvn test -Dsurefire.excludedGroups= -Dgroups=integration` - `LocalRagIT` asserts the answers
are grounded in the corpus facts (model-gated through the repo's `TestModels` cache lookup).
