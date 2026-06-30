package com.qxotic.llm;

import com.qxotic.jinfer.FloatTensor;

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
        record Embeddings(FloatTensor rows, int count) implements Input {}
    }

    /** Prefill a prompt span, projecting only the last row (the next-token distribution). */
    public static Batch prefill(int[] ids) { return new Batch(new Input.Tokens(ids), Outputs.LAST); }

    /** Decode one sampled token. */
    public static Batch step(int id) { return new Batch(new Input.Tokens(new int[]{id}), Outputs.LAST); }

    /** Score a span, retaining every row (e.g. perplexity / speculative verify). */
    public static Batch score(int[] ids) { return new Batch(new Input.Tokens(ids), Outputs.ALL); }

    /** Ingest {@code count} encoder-projected rows (a modality's soft tokens) inline. */
    public static Batch embeddings(FloatTensor rows, int count) { return new Batch(new Input.Embeddings(rows, count), Outputs.LAST); }

    /** Rows this batch ingests, regardless of modality. */
    public int count() {
        return switch (input) {
            case Input.Tokens t     -> t.ids().length;
            case Input.Embeddings e -> e.count();
        };
    }
}
