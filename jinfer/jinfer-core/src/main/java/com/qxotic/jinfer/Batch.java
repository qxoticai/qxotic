package com.qxotic.jinfer;

/** One forward call's worth of work: what to feed ({@link Input}) and which final hidden states to
 *  retain ({@link Outputs}). Position-agnostic: a batch is always ingested at the state's cursor
 *  ({@link RuntimeState#position()}), which then advances by {@link #count()}. */
public record Batch(Input input, Outputs outputs) {

    /** Which rows' final hidden state to retain for projection. LAST = generation; ALL = scoring. */
    public enum Outputs { LAST, ALL }

    /** The multi-modal seam: text as token ids (embedded internally — the ids also drive features like
     *  per-layer embeddings), other modalities as rows an encoder already projected to model dim. */
    public sealed interface Input {
        record Tokens(int[] ids)                       implements Input {}
        /** Encoder-projected rows. {@code bidirectional} = the modality's soft tokens attend to each other
         *  non-causally within the chunk (gemma image: true; gemma audio: false / causal). */
        record Embeddings(FloatTensor rows, int count, boolean bidirectional) implements Input {}
        /** Packed (ragged) multi-sequence text: {@code tokens.ids()} is this chunk's slice of the packed
         *  stream; {@code seqLen[j]} is the FULL length of sequence j across the whole packed stream (not
         *  just this chunk). Each token attends only within its own sequence, causally, positions restart
         *  at 0 per sequence. Used for batched embedding (no padding). */
        record Sequences(Tokens tokens, int[] seqLen)  implements Input {}
    }

    /** Prefill a prompt span, projecting only the last row (the next-token distribution). */
    public static Batch prefill(int[] ids) { return new Batch(new Input.Tokens(ids), Outputs.LAST); }

    /** Decode one sampled token. */
    public static Batch step(int id) { return new Batch(new Input.Tokens(new int[]{id}), Outputs.LAST); }

    /** Score a span, retaining every row (e.g. perplexity / speculative verify). */
    public static Batch score(int[] ids) { return new Batch(new Input.Tokens(ids), Outputs.ALL); }

    /** Ingest {@code count} encoder-projected rows (a modality's soft tokens) inline, attending
     *  bidirectionally within the chunk (image soft tokens). */
    public static Batch embeddings(FloatTensor rows, int count) { return new Batch(new Input.Embeddings(rows, count, true), Outputs.LAST); }

    /** Ingest {@code count} encoder-projected rows; {@code bidirectional=false} for causal modalities (audio). */
    public static Batch embeddings(FloatTensor rows, int count, boolean bidirectional) { return new Batch(new Input.Embeddings(rows, count, bidirectional), Outputs.LAST); }

    /** Pack ragged sequences into one batch (concatenate ids, record per-sequence lengths); retains every
     *  row so each sequence's pooled position is addressable. No padding. */
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

    /** Rows this batch ingests, regardless of modality. */
    public int count() {
        return switch (input) {
            case Input.Tokens t     -> t.ids().length;
            case Input.Embeddings e -> e.count();
            case Input.Sequences s  -> s.tokens().ids().length;
        };
    }
}
