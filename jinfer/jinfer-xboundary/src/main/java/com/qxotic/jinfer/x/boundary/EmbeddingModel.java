package com.qxotic.jinfer.x.boundary;

import com.qxotic.jota.memory.MemoryView;
import java.lang.ref.Reference;
import java.util.Arrays;
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
     *
     * <p>The default is the CAUSAL streaming law: the packed stream is ingested in {@code
     * batchCapacity}-sized chunks over one KV context (a sequence may span chunks - the model's
     * segmented attention carries its KV), and each sequence's LAST row is pooled as it completes.
     * Bidirectional families (LFM2) override it outright: a sequence must be forwarded WHOLE, so
     * they re-group on sequence boundaries instead.
     */
    default void embed(S state, Batch.Input.Sequences seqs, Consumer<MemoryView<?>> sink) {
        // Claimed across the WHOLE operation, not per chunk: reset() below mutates the cursor, and
        // releasing between chunks would let two concurrent embeds interleave their chunks into
        // one KV context - corrupting both, with no CME to say so. Reentrant, so the per-chunk
        // ingest/embedding claims nest inside this one.
        BaseState base = (BaseState) state;
        base.enter();
        try {
            embed0(state, seqs, sink);
        } finally {
            base.exit();
        }
    }

    private void embed0(S state, Batch.Input.Sequences seqs, Consumer<MemoryView<?>> sink) {
        int[] len = seqs.seqLen();
        int[] ids = seqs.tokens().ids();
        int n = ids.length;
        if (n > state.contextCapacity())
            throw new IllegalArgumentException(
                    "state contextCapacity "
                            + state.contextCapacity()
                            + " < packed length "
                            + n
                            + " (batchCapacity may be smaller; it only bounds the chunk)");
        int bc = state.batchCapacity();
        state.reset();
        int j = 0, seqStart = 0;
        for (int cs = 0; cs < n; cs += bc) {
            int ce = Math.min(cs + bc, n);
            int[] chunkIds = Arrays.copyOfRange(ids, cs, ce);
            // seqLen stays the FULL stream layout - the segmented attention resolves which
            // segments intersect this chunk from the cursor (a sequence may span chunks)
            ingest(
                    state,
                    new Batch(
                            new Batch.Input.Sequences(new Batch.Input.Tokens(chunkIds), len),
                            Batch.Outputs.ALL));
            while (j < len.length && seqStart + len[j] - 1 < ce) {
                sink.accept(
                        embedding(state, (seqStart + len[j] - 1) - cs)); // index within this chunk
                seqStart += len[j];
                j++;
            }
        }
    }
}
