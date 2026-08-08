/**
 * Spring AI provider backed by jinfer: in-process CPU inference over a local GGUF, no server.
 * {@link com.qxotic.jinfer.spring.ai.JinferChatModel} is the entry point (blocking {@code call} and
 * reactive {@code stream} on one object); configure per request via {@link
 * com.qxotic.jinfer.spring.ai.JinferChatOptions}. {@link
 * com.qxotic.jinfer.spring.ai.JinferEmbeddingModel} covers embeddings and {@link
 * com.qxotic.jinfer.spring.ai.JinferDocumentPostProcessor} reranks retrieved documents in a RAG
 * pipeline's post-retrieval stage (Spring AI models reranking as a step, not as a model type).
 *
 * <p>Every builder takes the model as ONE string - {@code .model("hf.co/unsloth/x-GGUF:Q4_K_M")} -
 * a local path, a hub ref, or a pasted browser URL, downloaded into the shared model cache when
 * absent (resumable, checksum-verified; {@code JINFER_OFFLINE=1} forbids the network). {@code
 * .modelPath(Path)} stays the never-network form; the Boot starter's {@code
 * spring.ai.jinfer.*.model} properties take the same string and resolve at context startup, so a
 * typo fails the boot, not the first request.
 *
 * <p>One model instance is ONE serial inference pipeline: concurrent calls queue fairly on it, and
 * a second pipeline means a second model over the same GGUF (the weight pages are shared by the OS
 * page cache, so the added cost is one context plus one load).
 *
 * <h2>Structured output</h2>
 *
 * <p>Grammar-constrained decoding makes structured output a guarantee, not a prompt: {@code
 * outputSchema} compiles to a grammar and the sampler can only emit documents the schema admits.
 * Two ways to populate it:
 *
 * <pre>{@code
 * // 1. Manual: a JSON Schema string on the options
 * ChatResponse r = model.call(new Prompt(
 *         new UserMessage("Describe Paris."),
 *         JinferChatOptions.builder()
 *                 .outputSchema("""
 *                     {"type":"object","properties":{"city":{"type":"string"},
 *                      "population":{"type":"number"}},"required":["city","population"]}""")
 *                 .build()));
 *
 * // 2. ChatClient entity(): Spring AI derives the schema from the target type and this model
 * //    (a StructuredOutputChatOptions implementation) receives it as outputSchema
 * City city = ChatClient.create(model).prompt()
 *         .user("Describe Paris.")
 *         .call()
 *         .entity(City.class);
 * }</pre>
 *
 * <p>On reasoning models the constraint binds only the OUTPUT channel - think spans sample free, so
 * structured output does not cost reasoning quality (the schema-bound text follows the closed
 * span). Expect schema-valid output always; expect FIELD QUALITY to track the model - a constrained
 * small model produces well-formed JSON with weak content. Tools and {@code outputSchema} on one
 * request reject loudly: a grammar mask cannot admit tool-call syntax.
 */
package com.qxotic.jinfer.spring.ai;
