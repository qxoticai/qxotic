/**
 * Spring AI adapters for in-process jinfer models.
 *
 * <p>{@link com.qxotic.jinfer.spring.ai.JinferChatModel} supplies blocking and streaming chat,
 * tools, structured output and multimodal prompts; configure generation through {@link
 * com.qxotic.jinfer.spring.ai.JinferChatOptions}. {@link
 * com.qxotic.jinfer.spring.ai.JinferEmbeddingModel} supplies embeddings, and {@link
 * com.qxotic.jinfer.spring.ai.JinferDocumentPostProcessor} reranks retrieved documents.
 *
 * <p>Builders accept a local path, a model reference, or an already loaded model. One adapter is
 * one serial inference pipeline. Chat caching stays intentionally small: {@code retainSessions}
 * keeps recent conversations live, {@code promptCache} mounts one persisted catalog, and {@code
 * withCachedPrompt} defines a reusable prefix.
 */
package com.qxotic.jinfer.spring.ai;
