package com.qxotic.jinfer.x.boundary;

import java.util.List;

/**
 * One forward call's worth of work: what to feed ({@link Input}) and which final hidden states to
 * retain ({@link Outputs}). Position-agnostic: a batch is always ingested at the state's cursor
 * ({@link RuntimeState#position()}), which then advances by {@link #count()}.
 *
 * <p>The slice's minimal clone of jinfer-core {@code Batch}: text only — {@link Input.Embeddings}
 * (the multi-modal seam) lands with the cycle that needs it.
 */
public record Batch(Input input, Outputs outputs) {

    /**
     * Which rows' final hidden state to retain for projection. LAST = generation; ALL = scoring.
     */
    public enum Outputs {
        LAST,
        ALL
    }

    /** Text inputs: token ids (embedded internally), or packed ragged multi-sequence text. */
    public sealed interface Input {
        record Tokens(int[] ids) implements Input {}

        /**
         * Packed (ragged) multi-sequence text: {@code tokens.ids()} is this chunk's slice of the
         * packed stream; {@code seqLen[j]} is the FULL length of sequence j across the whole packed
         * stream (not just this chunk). Each token attends only within its own sequence, causally,
         * positions restart at 0 per sequence. Used for batched embedding (no padding).
         */
        record Sequences(Tokens tokens, int[] seqLen) implements Input {}
    }

    /** Prefill a prompt span, projecting only the last row (the next-token distribution). */
    public static Batch prefill(int[] ids) {
        return new Batch(new Input.Tokens(ids), Outputs.LAST);
    }

    /** Decode one sampled token. */
    public static Batch step(int id) {
        return new Batch(new Input.Tokens(new int[] {id}), Outputs.LAST);
    }

    /** Score a span, retaining every row (e.g. perplexity / speculative verify). */
    public static Batch score(int[] ids) {
        return new Batch(new Input.Tokens(ids), Outputs.ALL);
    }

    /**
     * Pack ragged sequences into one batch (concatenate ids, record per-sequence lengths); retains
     * every row so each sequence's pooled position is addressable. No padding.
     */
    public static Batch pack(int[][] seqs) {
        int total = 0;
        for (int[] s : seqs) total += s.length;
        int[] ids = new int[total];
        int[] seqLen = new int[seqs.length];
        int off = 0;
        for (int j = 0; j < seqs.length; j++) {
            System.arraycopy(seqs[j], 0, ids, off, seqs[j].length);
            off += seqs[j].length;
            seqLen[j] = seqs[j].length;
        }
        return new Batch(new Input.Sequences(new Input.Tokens(ids), seqLen), Outputs.ALL);
    }

    /**
     * The token ids a batch list ingests, flattened in order. Token batches only — the shared
     * currency between chat encoding, cache fingerprints and the test harnesses, so the server and
     * the testkit stay byte-compatible.
     */
    public static int[] tokenIds(List<Batch> batches) {
        int n = 0;
        for (Batch b : batches) n += ((Input.Tokens) b.input()).ids().length;
        int[] ids = new int[n];
        int i = 0;
        for (Batch b : batches) {
            int[] part = ((Input.Tokens) b.input()).ids();
            System.arraycopy(part, 0, ids, i, part.length);
            i += part.length;
        }
        return ids;
    }

    /** Total positions a batch list ingests - {@link #count} summed. */
    public static int positions(List<Batch> batches) {
        int total = 0;
        for (Batch b : batches) total += b.count();
        return total;
    }

    /** Rows this batch ingests. */
    public int count() {
        return switch (input) {
            case Input.Tokens t -> t.ids().length;
            case Input.Sequences s -> s.tokens().ids().length;
        };
    }
}
