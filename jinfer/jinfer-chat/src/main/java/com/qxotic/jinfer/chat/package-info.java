/**
 * The chat codec: both directions between conversation structure and the token stream, and nothing
 * else.
 *
 * <p>{@link com.qxotic.jinfer.chat.Message} (role + ordered {@link com.qxotic.jinfer.chat.Part}s -
 * a span tree of text, reasoning, tool calls, tool results, media) is the portable high-level view;
 * {@link com.qxotic.jinfer.chat.Conversation} adds the conversation-scoped inputs (tools, effort).
 * {@link com.qxotic.jinfer.chat.ChatTemplate} is the per-model codec: {@code encode(Conversation)}
 * lowers to {@code List<Batch>}, {@code decoder()} parses the generated reply stream back into
 * Parts ({@link com.qxotic.jinfer.chat.ReplyDecoder} - subsumes tool-call detection and think-span
 * demuxing). Implementations are hand-written per model and validated token-exact against the
 * model's own GGUF Jinja template offline (the oracle tests). {@link
 * com.qxotic.jinfer.chat.TurnTemplate} is the transitional per-turn substrate, bridged by {@link
 * com.qxotic.jinfer.chat.TurnTemplateAdapter} until each model ports natively.
 *
 * <p>Two tokenization domains, always: turn scaffolding is emitted as trusted special-token ids;
 * conversation text is tokenized plainly, so content can never mint control tokens. Everything else
 * deliberately lives elsewhere: stop tokens on the model, batching policy on {@link
 * com.qxotic.jinfer.Batch#prepare}, session state in {@code cache.CachedSession}, the raw
 * generation loop in {@code llm.Generator} (which knows nothing of this package).
 */
package com.qxotic.jinfer.chat;
