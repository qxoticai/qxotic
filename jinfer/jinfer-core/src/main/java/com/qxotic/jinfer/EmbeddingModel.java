package com.qxotic.jinfer;

import java.util.function.Consumer;

/**
 * An encoder: a {@link Model} backbone whose head produces a pooled representation of the ingested
 * sequence (a sentence/document embedding), not vocabulary logits. Distinct from {@link Embedder},
 * which projects a non-text modality into model-dim input rows — this consumes tokens and outputs an
 * embedding.
 *
 * <p>Mirrors {@link LanguageModel}: the indexed method is the primitive and the no-arg is the last
 * retained row. {@code embedding} pools + normalizes the {@code index}-th retained hidden state exactly
 * as {@code logits} projects it — the two are the same computation up to the head.
 */
public interface EmbeddingModel<C extends Config, W, S extends RuntimeState> extends Model<C, W, S> {

    /** Pool (+ L2-normalize) the {@code index}-th retained hidden state of the last ingest into an embedding. */
    FloatTensor embedding(S state, int index);

    /** The last retained row — the pooled embedding of a single ingested sequence. */
    default FloatTensor embedding(S state) { return embedding(state, state.outputCount() - 1); }

    /**
     * Embed packed ragged sequences (see {@link Batch.Input.Sequences}). Ingests the packed stream in
     * {@code batchCapacity}-sized chunks over one KV context, and streams each sequence's pooled vector to
     * {@code sink} in input order. The whole packed context must fit in {@code contextCapacity}
     * ({@code batchCapacity} may be smaller - it only bounds the per-chunk forward). The {@code FloatTensor}
     * handed to the sink is freshly allocated per call.
     */
    default void embed(S state, Batch.Input.Sequences seqs, Consumer<FloatTensor> sink) {
        int[] len = seqs.seqLen();
        int[] ids = seqs.tokens().ids();
        int n = ids.length;
        if (n > state.contextCapacity())
            throw new IllegalArgumentException("state contextCapacity " + state.contextCapacity()
                    + " < packed length " + n + " (batchCapacity may be smaller; it only bounds the chunk)");
        int bc = state.batchCapacity();
        state.reset();
        int j = 0, seqStart = 0;
        for (int cs = 0; cs < n; cs += bc) {
            int ce = Math.min(cs + bc, n);
            int[] chunkIds = java.util.Arrays.copyOfRange(ids, cs, ce);
            ingest(state, new Batch(new Batch.Input.Sequences(new Batch.Input.Tokens(chunkIds), len), Batch.Outputs.ALL));
            while (j < len.length && seqStart + len[j] - 1 < ce) {
                sink.accept(embedding(state, (seqStart + len[j] - 1) - cs));   // index within this chunk
                seqStart += len[j];
                j++;
            }
        }
    }
}
