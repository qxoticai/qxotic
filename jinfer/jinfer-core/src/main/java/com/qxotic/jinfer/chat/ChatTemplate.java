package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.Batch;

import java.util.List;

/** The universal chat contract: lower a whole conversation to ingest-ready batches — text as
 *  {@link Batch.Input.Tokens}, media as encoder-projected {@link Batch.Input.Embeddings}. Stateless
 *  and deterministic: the same conversation must always produce the same batches (the prompt cache
 *  keys on them).
 *
 *  <p>Implementations are constructed with the model's tokenizer (and, for multimodal models, its
 *  embedders) so the output needs no further encoding: {@code encode(...) -> ingest} is the whole
 *  pipeline. Stop tokens are the model's ({@code stopTokens()}), not the format's; batching policy
 *  ({@code Batch.prepare}) is the caller's.
 *
 *  <p>Curated models implement the stronger {@link TurnTemplate}, which adds per-turn encoding and
 *  unlocks exact, incremental prompt caching. */
public interface ChatTemplate {

    /** The whole conversation, lowered to batches (generation prompt not included). */
    List<Batch> encode(List<Message> conversation);
}
