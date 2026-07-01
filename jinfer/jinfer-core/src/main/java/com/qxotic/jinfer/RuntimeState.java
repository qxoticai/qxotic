package com.qxotic.jinfer;

/** The live, mutable per-run state of a model — the dynamic third of the Config/Weights/State triple.
 *  A model owns none and produces many (concurrent runs, forked snapshots); it is caller-owned. */
public interface RuntimeState {
    int contextCapacity();   // KV ring this state allocated (≤ config.maxContextLength())
    int batchCapacity();     // max rows (tokens) a single ingest into this state may carry
    int position();        // tokens ingested so far
    int outputCount();     // hidden states the last ingest retained (1 after LAST, n after ALL)

    /** Rewind the cursor to 0 so the state can be reused for a fresh sequence/batch; the next ingest
     *  overwrites the KV from position 0. Optional - only states used with {@link EmbeddingModel#embed}
     *  need it; generative states leave the default. */
    default void reset() { throw new UnsupportedOperationException("reset() not supported by " + getClass().getName()); }
}
