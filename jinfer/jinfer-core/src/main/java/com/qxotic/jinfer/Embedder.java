package com.qxotic.jinfer;

import java.util.function.Consumer;

/**
 * A model-paired converter from a non-text {@link Media} source to model-dim rows (fed back as
 * {@link Batch.Input.Embeddings}) — the continuous-modality sibling of the tokenizer. Obtained from
 * the model via {@link MultiModal#embedder}, so it is wired to the model's weights and dim and owns
 * its own scratch. It streams output through {@code sink} in chunks of at most {@code maxChunkSize}
 * rows; each chunk is an ephemeral, model-dim-wide view (do not retain). The row count is dynamic —
 * a longer clip or larger image yields more rows; the model-specific frontend (resample/resize,
 * channel collapse, normalize) all happens inside, so the caller passes only the faithfully-decoded
 * {@link Media}.
 */
public interface Embedder<R extends Media> {
    void embed(R source, int maxChunkSize, Consumer<FloatTensor> sink);

    /**
     * Best-effort count of the model-dim rows {@link #embed} would emit for {@code source} -
     * computed from the PREPROCESSING PLAN (resize/tier selection, frame arithmetic), never by
     * encoding. Exact when the encoding is plan-determined (all current ports); a content-dependent
     * encoder returns its honest closest number and documents the accuracy.
     */
    default int positions(R source) {
        throw new UnsupportedOperationException(
                getClass().getSimpleName() + " does not plan media positions");
    }
}
