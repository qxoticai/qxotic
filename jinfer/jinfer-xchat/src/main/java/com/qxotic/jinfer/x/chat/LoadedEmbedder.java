package com.qxotic.jinfer.x.chat;

import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.boundary.EmbeddingModel;
import com.qxotic.jinfer.x.boundary.RuntimeState;
import com.qxotic.jinfer.x.telemetry.InferenceEvent;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.Tokenizer;
import java.util.List;
import java.util.function.Consumer;

/**
 * An embedding port's loaded bundle - the {@link Models#loadEmbedder} counterpart of {@link
 * LoadedModel}, carrying exactly what a provider integration needs: the model, its tokenizer, the
 * port's per-sequence framing convention ({@code sequencePrefix}/{@code sequenceSuffix} - tokens
 * wrapped around every encoded sequence: Qwen3's last-token pooling wants a trailing EOS, LFM2.5's
 * CLS pooling reads a leading BOS), the embedding width (static, so callers never probe with a
 * forward pass), and the model card's retrieval TEXT framing ({@code queryPrefix}/{@code
 * documentPrefix} - prepended before tokenizing when the caller states a retrieval role; {@code ""}
 * = the card prescribes none for that side). Retrieval-tuned embedders are trained WITH these
 * prefixes - LFM2.5's {@code "query: "}/{@code "document: "} pair, Qwen3's instructed query - and
 * embedding bare text instead silently degrades retrieval quality.
 */
public record LoadedEmbedder<S extends RuntimeState>(
        EmbeddingModel<?, ?, S> model,
        Tokenizer tokenizer,
        int[] sequencePrefix,
        int[] sequenceSuffix,
        int dimension,
        String name,
        String queryPrefix,
        String documentPrefix) {

    public LoadedEmbedder {
        if (model == null) throw new IllegalArgumentException("null model");
        if (name == null) throw new IllegalArgumentException("null name");
        if (tokenizer == null) throw new IllegalArgumentException("null tokenizer");
        if (sequencePrefix == null) throw new IllegalArgumentException("null sequencePrefix");
        if (sequenceSuffix == null) throw new IllegalArgumentException("null sequenceSuffix");
        if (dimension <= 0) throw new IllegalArgumentException("dimension " + dimension);
        if (queryPrefix == null) throw new IllegalArgumentException("null queryPrefix");
        if (documentPrefix == null) throw new IllegalArgumentException("null documentPrefix");
        sequencePrefix = sequencePrefix.clone();
        sequenceSuffix = sequenceSuffix.clone();
    }

    /** Whether the card prescribes retrieval framing at all (either side non-empty). */
    public boolean prefixTrained() {
        return !queryPrefix.isEmpty() || !documentPrefix.isEmpty();
    }

    /**
     * The provider-integration workhorse: encode each text (plus {@link #sequencePrefix} and {@link
     * #sequenceSuffix}), pack the sequences greedily into context-bounded ragged batches, and embed
     * - ONE forward pass per group, so RAG ingestion of hundreds of chunks costs a handful of
     * prefills. {@code sink} receives one pooled vector per text, in order; it may be a REUSED
     * per-state buffer - copy it out before returning. A text longer than {@code contextLength}
     * throws {@link IllegalArgumentException}. Returns the exact total token count. Serialize calls
     * per state (the state is one pipeline); {@code state} must come from {@link #model()}.
     */
    public int embedAll(
            RuntimeState state,
            int contextLength,
            List<String> texts,
            Consumer<MemoryView<?>> sink) {
        int[][] seqs = new int[texts.size()][];
        int total = 0;
        for (int i = 0; i < texts.size(); i++) {
            int[] ids = tokenizer.encode(texts.get(i)).toArray();
            int[] seq = new int[sequencePrefix.length + ids.length + sequenceSuffix.length];
            System.arraycopy(sequencePrefix, 0, seq, 0, sequencePrefix.length);
            System.arraycopy(ids, 0, seq, sequencePrefix.length, ids.length);
            System.arraycopy(
                    sequenceSuffix,
                    0,
                    seq,
                    sequencePrefix.length + ids.length,
                    sequenceSuffix.length);
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
            Consumer<MemoryView<?>> sink) {
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
            model.embed(s, new Batch.Input.Sequences(new Batch.Input.Tokens(ids), len), sink);
            start = end;
        }
        return total;
    }
}
