# jinfer architecture

One implementation stack built on `MemoryView`: model boundaries, kernels, chat orchestration and
the prompt cache.

## Modules

```text
jinfer-core       primitives, model contracts, media vocabulary and telemetry
jinfer-codecs     image/audio/video decoding (bytes <-> Media)
jinfer-kernels    GGUF loading and CPU kernels over MemoryView
jinfer-cache      retained sessions and content-addressed checkpoint storage
jinfer-chat       sampling, grammar constraints, generation, chat, templates and tools
jinfer-<model>     one ServiceLoader provider per supported GGUF architecture
jinfer-server     OpenAI-compatible HTTP/SSE server
jinfer-cli        chat, completion, pull and server executable
jinfer-bench      generic language-model and embedding benchmarks

jinfer-hub         model references, local cache and downloads
jinfer-jinja       whole-template fallback renderer
jinfer-testkit     shared model-fixture discovery for tests
```

`jinfer-models-all` is the convenience aggregate. Footprint-sensitive applications can depend on
only the architecture modules they need; `Models.load` discovers the available providers and gives
an actionable error when a matching provider is absent.

Dependencies point toward the small substrate: core at the bottom, then codecs, kernels, cache
and chat, then model providers, integrations and executables. The CLI is a leaf.

## Model boundary

```text
Model<C, W, S extends RuntimeState>
├── ContextModel             incremental bounded ingestion
│   ├── LanguageModel        logits projection
│   └── EmbeddingModel       pooled embedding projection
└── SpeechSynthesisModel     text-to-audio generation

Reranker<S extends ContextState>   relevance projection - a sibling of Model, not a subtype
```

Configuration and weights are immutable. Runtime state owns mutable inference memory and admits one
serial operation at a time. A caller-supplied arena remains caller-owned; a state-created arena is
released with the state. Returned `MemoryView`s are borrowed unless an API explicitly returns a
copy, and callback APIs delimit borrowed-view validity.

## Threading

One pool: `Parallel.shared()`, `-Djinfer.threads` participants (the caller is one), sized to the physical cores by default.
Every kernel, every model port and every jam backend runs its parallel regions on it; jam backends receive it through `JAM.Provider.create(Parallel)` and own no threads (the native library creates none inside a JVM).
A region body gets `(index, slot)`; the slot indexes per-participant scratch and is unique within one loop.
Regions of one pool are serialized, a loop from inside a region runs inline, a body that throws ends the loop early, and `MatMul.mm` refuses a call from inside a region.
Around the pool: one fair engine lock per `ChatEngine` serializes generations while prepares run beside them on a read lock (so a stream can be enqueued without waiting for the reply in flight); the server has one generation worker and an admission gate; the streaming driver is one lazy thread.
The audit and the refactor record live in `THREADING.md` at the repository root.

## Chat flow

```text
Conversation(role, Content...)
  -> ChatTemplate.encode(..., Consumer<Batch>)
  -> PromptWriter streams token or projected-media batches
  -> PromptCache serves the longest compatible prefix
  -> ContextModel.ingest(state, batch)
  -> LanguageModel.logits(state)
  -> Generator samples and streams reply tokens
  -> ReplyParser produces the assistant message and tool calls
```

Native templates emit trusted control tokens directly while user text is always plain-tokenized.
Media projectors stream borrowed embedding rows; `MediaEncodingCache` can replay a previous
projection by `ContentKey`. Models without a native codec use the hardened Jinja whole-render
fallback.

## Prompt cache

`PromptCache` is the production entry point over three mechanisms:

- retained sessions keep a bounded number of live, appendable conversation states;
- block storage checkpoints complete prompt batches under a byte budget;
- one optional JKVF catalog supplies persistent read-only prefixes and can receive exported ones.

Block identities chain model identity with token IDs or media content keys, so incompatible models
and prompts cannot match. A restore stops one position short and re-ingests the last position to
produce fresh logits. Misses, eviction or unavailable checkpoints always degrade to recomputation;
caching changes cost, never inference semantics.

## Testing

Fast unit contracts cover state lifecycle, cache laws, kernels, templates, grammar, media and
server protocols. Fixture-gated integration tests exercise real GGUF models, framework TCKs,
checkpoint restore, tools, structured output, vision, speech, embeddings, reranking and MTP.
Oracle coupling to the removed tensor implementation is deliberately absent.
