package com.qxotic.llm;

import com.qxotic.jinfer.FloatTensor;

import java.util.function.Consumer;

/** A model-paired converter from a non-text {@link Media} source to model-dim rows (fed back as
 *  {@link Batch.Input.Embeddings}) — the continuous-modality sibling of the tokenizer. Obtained from the
 *  model via {@link MultiModal#embedder}, so it is wired to the model's weights and dim and owns its own
 *  scratch. It streams output through {@code sink} in chunks of at most {@code maxChunkSize} rows; each
 *  chunk is an ephemeral, model-dim-wide view (do not retain). The row count is dynamic — a longer clip or
 *  larger image yields more rows; the model-specific frontend (resample/resize, channel collapse,
 *  normalize) all happens inside, so the caller passes only the faithfully-decoded {@link Media}. */
public interface Embedder<R extends Media> {
    void embed(R source, int maxChunkSize, Consumer<FloatTensor> sink);
}
