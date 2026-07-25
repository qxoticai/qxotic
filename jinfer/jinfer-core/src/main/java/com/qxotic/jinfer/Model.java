// The jinfer LLM model API, in its own package so it depends only on the public FloatTensor kernels
// from com.qxotic.jinfer — never on that package's internals. See Gemma4 for a LanguageModel impl.
package com.qxotic.jinfer;

/**
 * The headless backbone: ingest input into a runtime state and advance it. It has no opinion on the
 * output — a {@link com.qxotic.jinfer.LanguageModel} adds a vocab-logits head, an {@code
 * EmbeddingModel} a pooled head. Weight-bearing (the role HuggingFace calls {@code XxxModel});
 * weights are captured so {@link #ingest} never threads them, and exposed so a model can be cheaply
 * cloned over shared weights ({@code new Impl(config(), weights())}). State is caller-owned,
 * forkable, and many run at once.
 *
 * <p>Lifetime: weights and state buffers release with reachability - drop the last reference and GC
 * unmaps/frees them; there is no close(). Kernels read those buffers via raw addresses the GC
 * cannot see ({@code FloatTensor.GLOBAL_SEGMENT}), so every public entry point that runs kernels
 * ({@link #ingest}, a head projection) is a default wrapper whose trailing reachability fence pins
 * the model across the call - implementations override the unfenced seam ({@link #forward}, ...)
 * and callers never think about it.
 */
public interface Model<C extends Config, W, S extends RuntimeState> {

    C config();

    W weights();

    /**
     * Allocate a state: a KV ring sized to {@code contextCapacity} and scratch for batches up to
     * {@code batchCapacity} rows. {@code contextCapacity} must not exceed {@code
     * config.maxContextLength()}.
     */
    S newState(int contextCapacity, int batchCapacity);

    /**
     * Scratch width {@link #newState(int)} allocates when the caller doesn't pick one: a prefill of
     * up to this many tokens ingests in a single batch; longer prompts are re-chunked by the
     * caller. Defaults to 512; override with {@code -Djinfer.batchCapacity} (read at run time, JVM
     * or native).
     */
    default S newState(int contextCapacity) {
        return newState(contextCapacity, RuntimeFlags.BATCH_CAPACITY);
    }

    /**
     * Ingest one batch at the state's cursor ({@link RuntimeState#position()}), advancing it, and
     * retain the final hidden states selected by {@link Batch#outputs()}. The {@link Batch.Input}
     * union is the multi-modal seam.
     */
    default void ingest(S state, Batch batch) {
        forward(state, batch);
        java.lang.ref.Reference.reachabilityFence(this);
    }

    /** The forward pass behind {@link #ingest} - the implementation seam, never called directly. */
    void forward(S state, Batch batch);
}
