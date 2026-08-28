/**
 * langchain4j provider backed by jinfer: in-process CPU inference over a local GGUF, no server.
 * {@link com.qxotic.jinfer.langchain4j.JinferChatModel} is the entry point (blocking; {@code
 * streaming()} for the twin), {@link com.qxotic.jinfer.langchain4j.JinferEmbeddingModel} covers
 * embeddings, and {@link com.qxotic.jinfer.langchain4j.JinferScoringModel} reranks retrieved
 * segments (a {@code ScoringModel}, so langchain4j's {@code ReRankingContentAggregator} takes it
 * as-is).
 *
 * <p>Every builder takes the model as a MODEL REF - {@code .model("hf.co/owner/model-GGUF:Q4_K_M")}
 * - downloaded into the shared model cache when absent (resumable, checksum-verified; {@code
 * JINFER_OFFLINE=1} forbids the network). {@code .modelPath(Path)} is the explicit local form - the
 * two doors never overlap, and a URL is not a model ref: download it first, then pass the path.
 *
 * <p>Resolution of a remote {@code model(String)} goes through the ambient {@code ModelStore};
 * {@code .modelPath(Path)} is the preferred, explicit local form and touches neither the cache nor
 * the network.
 *
 * <p>One model instance is ONE serial inference pipeline: concurrent calls queue fairly on it. For
 * parallel pipelines, load the weights once into YOUR arena and fork - every builder has a {@code
 * model(loaded)} seam and every model a {@code fork()}:
 *
 * <pre>{@code
 * try (Arena arena = Arenas.newCrossThread()) {
 *     var loaded = Models.loadEmbedder(ModelStore.standard().resolve("hf.co/...:Q8_0"), arena);
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
 * mutually exclusive per request. Tools COMPOSE with a schema format - the schema rides the
 * family's reply language, so calls stay the family's own syntax while visible text can only be the
 * schema; raw GBNF and schemaless JSON still reject tools loudly. On reasoning models the
 * constraint binds only the OUTPUT channel - think spans sample free, so structured output does not
 * cost reasoning quality. Expect grammar-valid output always; expect FIELD QUALITY to track the
 * model - a constrained small model produces well-formed JSON with weak content.
 *
 * <h2>Constrained output: the prompt still has a job</h2>
 *
 * <p>THE GRAMMAR GUARANTEES THE FORM; THE PROMPT SETS THE PLAN. A mask says, at each step, "not
 * that token" - it never says "here is the target". The model decides WHAT to write from the prompt
 * alone, and the mask then edits how that decision is spelled, one token at a time. When the two
 * agree the grammar never fires and costs nothing. When they disagree the grammar wins on syntax
 * and the CONTENT pays, because a greedy decoder cannot revise a prefix it has already committed
 * to.
 *
 * <p>What that looks like: asked to extract "Apollo 11 launched on 16 July 1969" into a record with
 * a {@code LocalDate}, and given the schema ONLY as a grammar, LFM2.5-2.6B answered {@code
 * "1677-11-19"}. It began writing the DAY, as the sentence spells it; the date rule admits only
 * digits; that 16 was stranded in the year field with no way back. The output is perfectly valid
 * JSON against the schema - which is the danger, since nothing throws and nothing logs.
 *
 * <p>This does NOT improve with model size. On the langchain4j POJO battery, stating the schema in
 * the prompt is worth 36 vs 32 of 39 on LFM2.5-2.6B and 34 vs 28 on gemma-4-26B: the larger model
 * loses MORE, because it has stronger ideas of its own for the mask to fight. So state the shape
 * you are about to enforce - this provider will not do it for you, and neither will langchain4j
 * once a provider declares {@code RESPONSE_FORMAT_JSON_SCHEMA}, as this one must to get grammars at
 * all.
 *
 * <p>SAY IT IN PLAIN WORDS, not schema jargon. "Write dates as YYYY-MM-DD" fixed the case above;
 * the same schema carrying {@code "format": "date"} did not, reliably - that spelling asks the
 * model to know a JSON Schema convention, the sentence asks it to know nothing. An example of the
 * output you want is better still. And prefer to CONSTRAIN WHAT YOU ALREADY ASKED FOR: "answer yes
 * or no" with a two-way choice grammar cannot diverge, because the plan and the form are the same
 * thing.
 *
 * <p>None of which makes the grammar redundant - it buys what a prompt cannot promise. A
 * well-prompted UNCONSTRAINED run of the same request returned the right date wrapped in a {@code
 * ```json} fence; elsewhere it returned prose. The grammar makes a fence, a preamble, an invented
 * field, a truncated string and malformed JSON unrepresentable rather than unlikely. Prompt for
 * CORRECT, constrain for WELL-FORMED; neither substitutes for the other.
 *
 * <p>Where to put it: in your system prompt, welded with {@link
 * com.qxotic.jinfer.langchain4j.JinferChatModel#withCachedPrompt} - a fixed schema in a fixed
 * prefix is prefilled ONCE per process (or zero times, mounted from {@code Builder.promptCache}),
 * where the same text on each request's user message is re-prefilled every time. In multi-turn it
 * also matters that a prefix stays byte-identical: text appended to the newest user message moves
 * each turn, so the reusable prefix ends at the previous turn and every earlier assistant answer is
 * recomputed.
 *
 * <h2>Tool declarations</h2>
 *
 * <p>DESCRIBE EVERY TOOL. A local model decides from the declaration alone whether a tool applies,
 * and a small checkpoint skips an undescribed one and answers from its own knowledge instead.
 * langchain4j builds the declaration by reflecting the method, so a tool written without
 * {@code @Tool("...")}, without {@code @P("...")} on its parameters, or compiled without {@code
 * -parameters} (which leaves the argument names as {@code arg0}, {@code arg1}) reaches the model
 * with nothing to reason about.
 *
 * <p>Measured, same model and question: given {@code add(arg0, arg1)} with no descriptions,
 * LFM2.5-8B-A1B answered "the provided tools are not applicable for basic arithmetic calculations"
 * and called nothing; given a described {@code add(a, b)} it called the tool on the first try.
 * Hosted frontier models tolerate bare declarations because they were trained to - local ones
 * should not be asked to. The same law as structured output: the wire is always well-formed, the
 * DECISION tracks the model.
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
 * overriding the welded tool set (a one-time stderr warning names the two sets). {@code
 * Builder.promptCache(path)} mounts one read-only artifact and rejects a missing or incompatible
 * file at build time.
 */
package com.qxotic.jinfer.langchain4j;
