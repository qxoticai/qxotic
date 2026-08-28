/**
 * Spring AI adapters for in-process jinfer models.
 *
 * <p>{@link com.qxotic.jinfer.spring.ai.JinferChatModel} supplies blocking and streaming chat,
 * tools, structured output and multimodal prompts; configure generation through {@link
 * com.qxotic.jinfer.spring.ai.JinferChatOptions}. {@link
 * com.qxotic.jinfer.spring.ai.JinferEmbeddingModel} supplies embeddings, and {@link
 * com.qxotic.jinfer.spring.ai.JinferDocumentPostProcessor} reranks retrieved documents.
 *
 * <p>DESCRIBE EVERY TOOL. A local model decides from the declaration alone whether a tool applies,
 * and a small checkpoint skips an undescribed one and answers from its own knowledge instead.
 * Spring AI builds the declaration by reflecting the method, so a tool without {@code description},
 * without {@code @ToolParam(description = "...")} on its parameters, or compiled without {@code
 * -parameters} (which leaves the argument names as {@code arg0}, {@code arg1}) reaches the model
 * with nothing to reason about. Measured, same model and question: given {@code add(arg0, arg1)}
 * with no descriptions, LFM2.5-8B-A1B answered "the provided tools are not applicable for basic
 * arithmetic calculations" and called nothing; given a described {@code add(a, b)} it called the
 * tool on the first try.
 *
 * <p>Builders accept a local path, a model reference, or an already loaded model. One adapter is
 * one serial inference pipeline. Chat caching stays intentionally small: {@code retainSessions}
 * keeps recent conversations live, {@code promptCache} mounts one persisted catalog, and {@code
 * withCachedPrompt} defines a reusable prefix.
 */
package com.qxotic.jinfer.spring.ai;
