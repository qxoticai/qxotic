/**
 * Chat layout: lowering a conversation to ingest-ready batches, and nothing else.
 *
 * <p>{@link com.qxotic.jinfer.chat.Message} (role + ordered {@link com.qxotic.jinfer.chat.Part}s,
 * text or media) is the portable high-level view. {@link com.qxotic.jinfer.chat.ChatTemplate}
 * lowers a whole conversation to {@code List<Batch>}; {@link com.qxotic.jinfer.chat.TurnTemplate}
 * is the curated-model refinement - per-turn, deterministic, turn-stable - which is exactly the
 * property that makes incremental ingestion and exact prompt caching sound. Implementations are
 * hand-written per model and validated token-exact against the model's own GGUF Jinja template
 * offline (the oracle tests); {@code instanceof TurnTemplate} is the capability test.
 *
 * <p>Two tokenization domains, always: turn scaffolding is emitted as trusted special-token ids;
 * conversation text is tokenized plainly, so content can never mint control tokens. Everything else
 * deliberately lives elsewhere: stop tokens on the model, batching policy on {@link
 * com.qxotic.jinfer.Batch#prepare}, session state in {@code cache.CachedSession}.
 */
package com.qxotic.jinfer.chat;
