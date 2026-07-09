package com.qxotic.jinfer;

/**
 * The live, mutable per-run state of a model — the dynamic third of the Config/Weights/State
 * triple. A model owns none and produces many (concurrent runs, forked snapshots); it is
 * caller-owned.
 */
public interface RuntimeState {
    int contextCapacity(); // KV ring this state allocated (≤ config.maxContextLength())

    int batchCapacity(); // max rows (tokens) a single ingest into this state may carry

    int position(); // tokens ingested so far

    int outputCount(); // hidden states the last ingest retained (1 after LAST, n after ALL)

    /**
     * Advances the cursor past an ingested batch: {@code rows} rows landed, retaining every row's
     * hidden state under {@link Batch.Outputs#ALL} or just the last one under LAST. Called by the
     * model's ingest tail - the one place the cursor moves forward.
     */
    default void advance(int rows, Batch.Outputs outputs) {
        throw new UnsupportedOperationException(
                "advance() not supported by " + getClass().getName());
    }

    /**
     * Makes the state resumable at {@code position} after its KV/recurrent contents were restored
     * externally (prompt cache): sets the cursor and resets per-batch scratch invariants. Transient
     * output state (logits) is NOT restored - ingest before reading logits.
     */
    default void resumeAt(int position) {
        throw new UnsupportedOperationException(
                "resumeAt() not supported by " + getClass().getName());
    }

    /**
     * Rewind the cursor to 0 so the state can be reused for a fresh sequence/batch; the next ingest
     * overwrites the KV from position 0. Optional - only states used with {@link
     * EmbeddingModel#embed} need it; generative states leave the default.
     */
    default void reset() {
        throw new UnsupportedOperationException("reset() not supported by " + getClass().getName());
    }
}
