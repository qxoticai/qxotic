package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.EmbeddingModel;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.telemetry.InferenceEvent;
import com.qxotic.toknroll.Tokenizer;

/**
 * An embedding port's loaded bundle - the {@link Models#loadEmbedder} counterpart of {@link
 * LoadedModel}, carrying exactly what a provider integration needs: the model, its tokenizer, the
 * port's per-sequence pooling convention ({@code sequenceSuffix} - tokens appended to every encoded
 * sequence, e.g. Qwen3's last-token pooling wants a trailing EOS), and the embedding width (static,
 * so callers never probe with a forward pass).
 */
public record LoadedEmbedder<S extends RuntimeState>(
        EmbeddingModel<?, ?, S> model,
        Tokenizer tokenizer,
        int[] sequenceSuffix,
        int dimension,
        String name) {

    public LoadedEmbedder {
        if (model == null) throw new IllegalArgumentException("null model");
        if (name == null) throw new IllegalArgumentException("null name");
        if (tokenizer == null) throw new IllegalArgumentException("null tokenizer");
        if (sequenceSuffix == null) throw new IllegalArgumentException("null sequenceSuffix");
        if (dimension <= 0) throw new IllegalArgumentException("dimension " + dimension);
        sequenceSuffix = sequenceSuffix.clone();
    }

    /**
     * The provider-integration workhorse: encode each text (plus {@link #sequenceSuffix}), pack the
     * sequences greedily into context-bounded ragged batches, and embed - ONE forward pass per
     * group, so RAG ingestion of hundreds of chunks costs a handful of prefills. {@code sink}
     * receives one pooled vector per text, in order; it may be a REUSED per-state buffer - copy it
     * out before returning. A text longer than {@code contextLength} throws {@link
     * IllegalArgumentException}. Returns the exact total token count. Serialize calls per state
     * (the state is one pipeline); {@code state} must come from {@link #model()}.
     */
    public int embedAll(
            RuntimeState state,
            int contextLength,
            java.util.List<String> texts,
            java.util.function.Consumer<com.qxotic.jinfer.FloatTensor> sink) {
        int[][] seqs = new int[texts.size()][];
        int total = 0;
        for (int i = 0; i < texts.size(); i++) {
            int[] ids = tokenizer.encode(texts.get(i)).toArray();
            int[] seq = java.util.Arrays.copyOf(ids, ids.length + sequenceSuffix.length);
            System.arraycopy(sequenceSuffix, 0, seq, ids.length, sequenceSuffix.length);
            if (seq.length > contextLength) {
                throw new IllegalArgumentException(
                        "text "
                                + i
                                + " is "
                                + seq.length
                                + " tokens, over the "
                                + contextLength
                                + "-token context - raise contextLength(...) or chunk smaller");
            }
            seqs[i] = seq;
            total += seq.length;
        }
        InferenceEvent event =
                InferenceEvent.started(name, InferenceEvent.EMBEDDINGS, InferenceEvent.TEXT);
        long startNanos = System.nanoTime();
        try {
            int tokens = embedPacked(state, contextLength, seqs, total, sink);
            event.inputTokens = tokens;
            return tokens;
        } catch (RuntimeException | Error failure) {
            event.errorType = failure.getClass().getSimpleName();
            throw failure;
        } finally {
            // an encode runs no decode loop: outputTokens and decodeTime are a true zero here
            event.prefillTime = System.nanoTime() - startNanos;
            event.end();
            event.commit();
        }
    }

    private int embedPacked(
            RuntimeState state,
            int contextLength,
            int[][] seqs,
            int total,
            java.util.function.Consumer<com.qxotic.jinfer.FloatTensor> sink) {
        @SuppressWarnings("unchecked") // states of this embedder's model ARE S, by construction
        S s = (S) state;
        // greedy packing: each group fills the context, one forward pass per group
        int start = 0;
        while (start < seqs.length) {
            int end = start, packed = 0;
            while (end < seqs.length && packed + seqs[end].length <= contextLength) {
                packed += seqs[end].length;
                end++;
            }
            int[] ids = new int[packed];
            int[] len = new int[end - start];
            int at = 0;
            for (int i = start; i < end; i++) {
                System.arraycopy(seqs[i], 0, ids, at, seqs[i].length);
                at += seqs[i].length;
                len[i - start] = seqs[i].length;
            }
            model.embed(
                    s,
                    new com.qxotic.jinfer.Batch.Input.Sequences(
                            new com.qxotic.jinfer.Batch.Input.Tokens(ids), len),
                    sink);
            start = end;
        }
        return total;
    }
}
