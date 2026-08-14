# jinfer-xspring-ai: port plan

Spring AI integration over the x stack (MemoryView/xchat), twinning the old trio (`jinfer-spring-ai`, `-autoconfigure`, `-spring-boot-starter`).
The precedent is `jinfer-xlangchain4j`: the same integration breadth already proven on xchat, so this port copies that architecture and swaps framework types; it does not invent one.

## Verified anchors

- Old trio: `jinfer-spring-ai` 2622 LOC (ChatModel 883, Mappings 317, EmbeddingModel 340, SpeechModel 278, DocumentPostProcessor 261, ChatOptions 203), autoconfigure ~270 LOC (7 files), starter = pom only; 28 test files.
- `jinfer-xlangchain4j`: ChatModel 759, Mappings 475, EmbeddingModel 282, SpeechModel 204, ScoringModel 208, StreamingChatModel 182, ChatRequestParameters 148, TokenUsage 142, CachedPrompt 75.
- Spring AI 2.0.0, Spring Boot 4.1.0 - both already in dependencyManagement **and present in the offline ~/.m2** (spring-ai-bom 2.0.0, spring-boot 4.x artifacts confirmed; the `-o` build is feasible).
- xchat surface (HEAD): `ChatEngine.prepare(...) -> Prepared` (AutoCloseable), `complete(Prepared, ReplySink)`, `Conversation/Message/Tool`, `TextStops`, `PromptCache.Options`, `Models.load/loadEmbedder/loadReranker/loadSpeech`, xboundary `SpeechModel/SpeechOptions/AudioCodec`, `x.llm.Grammar`, `ChatEngine.speculationReady()` + `speculationDepth(int)`.
- Fixture refs already in `scripts/models.txt`: `hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf`, `hf.co/mradermacher/Qwen3-Reranker-0.6B-GGUF/Qwen3-Reranker-0.6B.Q8_0.gguf`.

## Decisions

**D1 - three modules, same as old.** `jinfer-xspring-ai`, `jinfer-xspring-ai-autoconfigure`, `jinfer-xspring-ai-spring-boot-starter`.
Package stays `com.qxotic.jinfer.spring.ai` with the same class names (twin convention from xlangchain4j: classpath-exclusive, swap by dependency).
Autoconfigure+starter are ~300 LOC total and are the entire reason Boot users adopt; not worth collapsing.
The starter pom copies old's verbatim with artifactIds swapped.

**D2 - full old surface, embeddings and reranker first-class (see below).**
No scope cuts from the old module set; it is already tight.

**D3 - x-only additions, minimal:**
- `speculationDepth` builder knob + `spring.ai.jinfer.chat.speculation-depth` property (default 4, 0 = off). The old stack cannot offer this; it is the x port's differentiator.
- `JinferUsage` gains draft acceptance counters (drafted/accepted) when `Completion.speculated()` is present; keeps old's `servedFrom`/nanos.

**D4 - firm drops (parity-checked, not laziness):**
- `Estimators`/token counting: Spring AI has no mandatory estimator hook for `ChatModel` (langchain4j needed it for `AiServices`; Spring's `ChatClient` does not). Add when a user asks.
- Retry wiring: old never wires `spring-ai-retry` into the model; Boot users compose their own `RetryTemplate`. Parity = skip.
- Reactive embedding/rerank variants: old is blocking-only.

## Embeddings + reranker (first-class scope)

These are the RAG half of the port and the most mechanical: the xlangchain4j twins (`JinferEmbeddingModel` 282 LOC, `JinferScoringModel` 208 LOC) already solved every hard problem on xchat. The Spring port is an interface swap over the same cores.

- `JinferEmbeddingModel implements EmbeddingModel` (spring-ai-model): port from the xlang twin, keeping old spring's laws: ragged batches bounded by the state's batch capacity, ONE reusable state reset per group, and the model card's task framing mapped onto Spring's types - `embed(Document)` is the ingestion framing, `embed(String)` the query framing - so vector stores get document framing and search gets query framing without configuration.
- `JinferDocumentPostProcessor implements DocumentPostProcessor` (spring-ai-rag): port from old 261 + the xlang reranker core (`Models.loadReranker`/`LoadedReranker`); Spring's retrieval-augmentation chain takes a `DocumentPostProcessor` as-is.
- Both get the builder/fork/ownsWeights shape from the xlang twins (load once into a caller arena, `fork()` per pipeline).
- Autoconfigure: `spring.ai.jinfer.embedding.*` and `spring.ai.jinfer.rerank.*` properties + one-bean AutoConfigurations, as old.
- Tests: `JinferEmbeddingModelIT` on Qwen3-Embedding-0.6B (framing visible in embeddings, batch>capacity regrouping, fork concurrency), `JinferDocumentPostProcessorIT` on Qwen3-Reranker-0.6B with ORDERING-based assertions (relevant doc outranks distractors - never score values, scores are engine-tuning-sensitive), plus the wrong-checkpoint refusal test (an LFM2.5 checkpoint must name the family's reranker in the error).

## File-by-file (jinfer-xspring-ai)

| File | Source | Shape of the port | Est. LOC |
|---|---|---|---|
| `JinferChatModel` | old 883 + xlang 759 | Spring-facing half (call/stream/resolveOptions/validate/observation/toFinishReason/Builder) copies old - it already targets Spring AI 2.0.0. jinfer-facing half swaps old-chat -> x-chat: `Prepared` in try-with-resources (xcli idiom), `ReplySink` bridge to `FluxSink` via `Flux.push` (single producer - not `Flux.create`), `Conversation` from Mappings. fork/withCachedPrompt/saveCachedPrompts follow xlang's `CachedPrompt`. | ~750 |
| `JinferChatOptions` | old 203 | Verbatim-ish (Spring types only). NO speculationDepth here - engine-level, not per-request. | ~200 |
| `Mappings` | old 317 + xlang 475 | Spring `UserMessage/AssistantMessage/ToolResponseMessage/SystemMessage` + media <-> x `Conversation/Message/Tool`/`Media`. Logic from old, media idioms from xlang. | ~350 |
| `JinferEmbeddingModel` | xlang 282 + old 340 | See section above. | ~300 |
| `JinferSpeechModel` | xlang 204 | Swap to Spring `SpeechModel`; same xboundary core. | ~210 |
| `JinferDocumentPostProcessor` | old 261 + xlang 208 | See section above. | ~220 |
| `package-info` | old 51 + xlang 92 | Entry points, one-pipeline concurrency law, fork idiom, cache tiers, JVM flags, split-package twin note. | ~60 |

## Autoconfigure + starter

- Properties records under `spring.ai.jinfer.{chat,embedding,rerank,speech}`: old keys verbatim, chat gains `speculationDepth` (default 4).
- AutoConfigurations one-bean-per-file as old, including the `ObjectProvider<ChatModelObservationConvention>` override seam old already uses.
- Boot 4 registration: `META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports`, copied from old.

## Tests (target ~30 files)

- Unit (no model): `JinferChatOptionsTest`, `JinferMappingsTest`, `JinferChatModelTest`, `JinferSpeechModelTest`, builder-contract, usage record (incl. acceptance counters).
- ITs (`TestModels.require`, lookup-only): `JinferChatModelIT`, `StructuredOutputIT`, `CachedPromptIT`, `JinferLifecycleIT`, embedding + reranker ITs as specced above, per-family capability ITs (gemma4/qwen35/gptoss/lfm2) on the `AbstractCapabilityIT` pattern.
- NEW `SpeculationIT`: gemma4 E2B + MTP sidecar - identical reply at depth 4 vs 0, acceptance counters on usage, property wiring through `ApplicationContextRunner`.
- Autoconfigure: `ApplicationContextRunner` tests (binding, bean presence/absence by property, companion resolution failure names the flag).

## Execution order (each step ends green: `mvn -o install`, then its tests)

1. **Scaffold**: 3 modules, poms, parent `<modules>` + dependencyManagement; empty compile. Offline repo already verified for spring artifacts.
2. **Embeddings + reranker** (most mechanical, unblocks RAG users first): both models + ITs.
3. **Mappings + ChatOptions** + unit tests.
4. **JinferChatModel blocking** + `JinferChatModelIT` (gemma4 E2B).
5. **Streaming**: `Flux.push` over `ReplySink` + stream tests.
6. **Tool calling + structured output** (`Grammar`) + ITs.
7. **fork + cached-prompt views** + CachedPromptIT/LifecycleIT.
8. **Speech** + test.
9. **Autoconfigure + starter** + ApplicationContextRunner tests.
10. **speculationDepth** (builder + property + usage counters) + SpeculationIT.
11. **Reactor sweep**: full `mvn -o install` from `jinfer/`, spotless, old trio untouched (twin law).

Steps 2, 3 and 8 are independent tracks; everything chat-side needs 3.

## Risks / resolve at step 1

- `ReplySink` exact method surface at xchat HEAD (request lifecycle was refactored this week) - read the source then, not from this plan.
- Spring `Usage` extensibility for acceptance counters: if no custom-key map, they live only on `JinferUsage` (old precedent: `servedFrom` already does).
- `DocumentPostProcessor` package moved between spring-ai 1.x milestones - confirm the 2.0.0 FQCN from the local jar before writing the class.
