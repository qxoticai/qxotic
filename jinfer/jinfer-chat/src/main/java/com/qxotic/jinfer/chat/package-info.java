/**
 * Conversation framing, reply parsing and the high-level chat engine.
 *
 * <p>{@link com.qxotic.jinfer.chat.Conversation} and {@link com.qxotic.jinfer.chat.Content}
 * describe portable messages, tools and media. A model's {@link
 * com.qxotic.jinfer.chat.ChatTemplate} streams the conversation as batches and returns a seeded
 * {@link com.qxotic.jinfer.chat.ReplyParser}; {@link com.qxotic.jinfer.chat.PromptWriter} is the
 * small shared substrate used by native templates. Template-authored control text is trusted, while
 * user content is always tokenized as plain text.
 *
 * <p>{@link com.qxotic.jinfer.chat.ChatEngine} owns generation policy, prompt caching, media
 * projection caching, timeouts and cancellation. {@link com.qxotic.jinfer.chat.LoadedModel} binds
 * it to the tokenizer, stop tokens, cache identity and template facts loaded from a GGUF.
 */
package com.qxotic.jinfer.chat;
