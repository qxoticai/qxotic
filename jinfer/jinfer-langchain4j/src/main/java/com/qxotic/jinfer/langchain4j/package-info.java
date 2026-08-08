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
 * <p>One model instance is ONE serial inference pipeline: concurrent calls queue fairly on it, and
 * a second pipeline means a second model over the same GGUF (the weight pages are shared by the OS
 * page cache, so the added cost is one context plus one load).
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
 */
package com.qxotic.jinfer.langchain4j;
