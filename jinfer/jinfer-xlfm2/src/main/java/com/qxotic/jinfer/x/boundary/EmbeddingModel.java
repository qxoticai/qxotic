package com.qxotic.jinfer.x.boundary;

import com.qxotic.jota.memory.MemoryView;
import java.lang.ref.Reference;
import java.util.function.Consumer;

/**
 * An encoder: a {@link Model} backbone whose head produces a pooled representation of the ingested
 * sequence (a sentence/document embedding), not vocabulary logits.
 *
 * <p>Mirrors {@link LanguageModel}: the indexed method is the primitive and the no-arg is the last
 * retained row. {@code embedding} pools + normalizes the {@code index}-th retained hidden state
 * exactly as {@code logits} projects it — the two are the same computation up to the head. The
 * boundary speaks {@link MemoryView}: pooled rows are FP32 {@code [dim]}, REUSED per-state buffers.
 */
public interface EmbeddingModel<C extends Config, W, S extends RuntimeState>
        extends Model<C, W, S> {

    /**
     * Pool (+ L2-normalize) the {@code index}-th retained hidden state of the last ingest into an
     * embedding.
     */
    default MemoryView<?> embedding(S state, int index) {
        BaseState base = (BaseState) state;
        base.enter();
        MemoryView<?> embedding;
        try {
            embedding = pool(state, index);
        } finally {
            base.exit();
        }
        Reference.reachabilityFence(this);
        return embedding;
    }

    /** The pooling head behind {@link #embedding} - the implementation seam. */
    MemoryView<?> pool(S state, int index);

    /** The last retained row — the pooled embedding of a single ingested sequence. */
    default MemoryView<?> embedding(S state) {
        return embedding(state, state.outputCount() - 1);
    }

    /**
     * Embed packed ragged sequences (see {@link Batch.Input.Sequences}), streaming each sequence's
     * pooled vector to {@code sink} in input order. The view handed to the sink may be a REUSED
     * per-state buffer: it is valid only until the next sink call - copy it out before returning.
     */
    void embed(S state, Batch.Input.Sequences seqs, Consumer<MemoryView<?>> sink);
}
