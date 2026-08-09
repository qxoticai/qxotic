/**
 * langchain4j provider backed by jinfer: in-process CPU inference over a local GGUF, no server.
 * {@link com.qxotic.jinfer.langchain4j.JinferChatModel} is the entry point (blocking; {@code
 * streaming()} for the twin), {@link com.qxotic.jinfer.langchain4j.JinferEmbeddingModel} covers
 * embeddings, and {@link com.qxotic.jinfer.langchain4j.JinferScoringModel} reranks retrieved
 * segments (a {@code ScoringModel}, so langchain4j's {@code ReRankingContentAggregator} takes it
 * as-is).
 *
 * <p>Every builder takes the model as ONE string - {@code .model("hf.co/unsloth/x-GGUF:Q4_K_M")} -
 * a local path, a hub ref, or a pasted browser URL, downloaded into the shared model cache when
 * absent (resumable, checksum-verified; {@code JINFER_OFFLINE=1} forbids the network). {@code
 * .modelPath(Path)} stays the never-network form.
 *
 * <p>One model instance is ONE serial inference pipeline: concurrent calls queue fairly on it. For
 * parallel pipelines, load the weights once into YOUR arena and fork - every builder has a {@code
 * model(loaded)} seam and every model a {@code fork()}:
 *
 * <pre>{@code
 * try (Arena arena = Arena.ofShared()) {
 *     var loaded = Models.loadEmbedder(ModelStore.resolve("hf.co/...:Q8_0"), arena);
 *     var a = JinferEmbeddingModel.builder().model(loaded).build();
 *     var b = a.fork();               // second pipeline, same weights, a context's price
 *     // ... parallel ingestion on a and b ...
 *     a.close(); b.close();
 * }                                   // the owner frees the weights, at a brace
 * }</pre>
 *
 * <p>The block structure IS the ownership story: your arena outlives every instance built on it. A
 * sequential violation is caught fail-fast - a safety canary at the forward pass throws {@code
 * IllegalStateException} on freed weights - while freeing the arena DURING a request is a data race
 * and can still crash the VM. {@code fork()} on a model that loaded its OWN weights refuses with
 * the load-once recipe.
 *
 * <h2>Structured output</h2>
 *
 * <p>Grammar-constrained decoding makes structured output a guarantee, not a prompt: the sampler
 * can only emit tokens the target language admits. Three escalating ways in:
 *
 * <pre>{@code
 * // 1. Schemaless JSON mode: any valid JSON document
 * model.chat(ChatRequest.builder()
 *         .messages(UserMessage.from("Describe Paris as JSON."))
 *         .responseFormat(ResponseFormat.JSON)
 *         .build());
 *
 * // 2. JSON schema: exactly the documents the schema admits (no extra keys, right types)
 * ResponseFormat cityFormat = ResponseFormat.builder()
 *         .type(ResponseFormatType.JSON)
 *         .jsonSchema(JsonSchema.builder()
 *                 .rootElement(JsonObjectSchema.builder()
 *                         .addStringProperty("city")
 *                         .addNumberProperty("population")
 *                         .required("city", "population")
 *                         .build())
 *                 .build())
 *         .build();
 *
 * // 3. AiServices POJO extraction rides the same machinery automatically: the model reports
 * //    RESPONSE_FORMAT_JSON_SCHEMA, so a typed service method just works
 * interface CityFacts { City describe(String prompt); }
 * City city = AiServices.create(CityFacts.class, model).describe("Describe Paris.");
 * }</pre>
 *
 * <p>Raw GBNF ({@link com.qxotic.jinfer.langchain4j.JinferChatRequestParameters#grammar}) is the
 * generalization for non-JSON shapes (label sets, numeric formats); grammar and JSON format are
 * mutually exclusive per request, and both reject tools loudly. On reasoning models the constraint
 * binds only the OUTPUT channel - think spans sample free, so structured output does not cost
 * reasoning quality. Expect grammar-valid output always; expect FIELD QUALITY to track the model -
 * a constrained small model produces well-formed JSON with weak content.
 *
 * <h2>Prompt caching and speed, diagnosable</h2>
 *
 * <p>{@code withCachedPrompt(messages, tools)} pins a view's prefix and default tools, prefilled
 * once and restored per request; caching changes latency, never behavior. Every response accounts
 * for what the engine did - cache read (the OpenAI {@code cached_tokens} pattern) and phase timings
 * ride the usage, so "is it the model or my code" needs no profiler:
 *
 * <pre>{@code
 * var usage = (JinferTokenUsage) response.tokenUsage();
 * usage.cachedInputTokens();  // ~prefix size = warm; 0 = the prefill was paid in full
 * usage.servedFrom();         // SESSION | BLOCKS | FRESH - which cache tier served
 * usage.promptNanos();        // prefill wall time; usage.predictedNanos() the decode time
 * log.info("{}", usage);      // renders it all: cached=1180/1204 BLOCKS, decode=41.9 tok/s
 * }</pre>
 *
 * <p>If TTFT looks cold on a view, read those two numbers first; the usual cause is a request
 * overriding the welded tool set (a one-time stderr warning names the two sets), then {@code
 * -Djinfer.promptCache=false}, or a missing {@code loadCachedPrompts} artifact (also warned).
 */
package com.qxotic.jinfer.langchain4j;
