// Per-sequence inference state, opaque to the engine beyond the running cursor. Each model owns
// its concrete state (KV cache layout, activation scratch); the engine only needs to read and
// advance the most recently ingested token.
package com.llama4j;

/** Mutable per-sequence state produced by {@link Model#createNewState()}. */
interface InferenceState {

    /** The most recently ingested token (BOS on a fresh state). */
    int latestToken();

    void latestToken(int token);
}
