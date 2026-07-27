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
 * <p>Lifetime: every buffer comes from a {@link java.lang.foreign.Arena}, and who provides the
 * arena owns it. Weights map into the arena given at load; states allocate from the arena given at
 * {@link #newState(int, int, java.lang.foreign.Arena)}, from an internal owned arena that {@code
 * state.close()} frees deterministically ({@link #newState(int, int)}), or from a caller-created
 * arena the state adopts and frees as its own ({@link #newState(int, int, java.lang.foreign.Arena,
 * boolean)}). Two laws the code cannot enforce: an arena must outlive every read from it (kernels
 * read raw addresses via {@code FloatTensor.GLOBAL_SEGMENT} - the JDK's close handshake cannot save
 * a raw read, so a violation is a crash, not an exception), and the weights arena must outlive
 * every model sharing those weights. Every public entry point that runs kernels ({@link #ingest}, a
 * head projection) is a default wrapper: it claims the state (fail-fast single-pipeline contract,
 * see {@link BaseState#enter()}) and its trailing reachability fence pins the model across the call
 * - implementations override the unfenced seam ({@link #forward}, ...) and callers never think
 * about it.
 */
public interface Model<C extends Config, W, S extends RuntimeState> {

    C config();

    W weights();

    /**
     * Allocate a state from {@code arena} (BORROWED: the caller owns the arena's lifetime; {@code
     * state.close()} never touches it): a KV ring sized to {@code contextCapacity} and scratch for
     * batches up to {@code batchCapacity} rows. {@code contextCapacity} must not exceed {@code
     * config.maxContextLength()}. Use {@code Arena.ofShared()} when another thread may compute on
     * the state (a streaming driver); a confined arena fails loudly there.
     */
    S newState(int contextCapacity, int batchCapacity, java.lang.foreign.Arena arena);

    /**
     * Allocate a state that OWNS its memory: an internal shared arena freed deterministically by
     * {@code state.close()} (blocking until any in-flight computation returns), with a Cleaner
     * backstop so a dropped unclosed state degrades to GC-eventually rather than leaking.
     */
    default S newState(int contextCapacity, int batchCapacity) {
        java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofShared();
        try {
            return newState(contextCapacity, batchCapacity, arena, true);
        } catch (RuntimeException | Error e) {
            arena.close(); // a leaked ofShared arena has no Cleaner: free before failing
            throw e;
        }
    }

    /**
     * As {@link #newState(int, int, java.lang.foreign.Arena)}, but when {@code adopt} is true the
     * state takes ownership of {@code arena}: {@code state.close()} (and its Cleaner backstop)
     * frees it. For fusing lifetimes deliberately - a single-state owner loads weights into the
     * arena and the state's close frees everything at once - so adopt only when nothing in the
     * arena outlives the state. A non-closeable arena ({@code ofAuto}, {@code global}) may be
     * adopted: it manages itself, and close stays a valid no-op on the memory.
     */
    default S newState(
            int contextCapacity, int batchCapacity, java.lang.foreign.Arena arena, boolean adopt) {
        S state = newState(contextCapacity, batchCapacity, arena);
        if (adopt) ((BaseState) state).adoptArena();
        return state;
    }

    /**
     * Scratch width {@link #newState(int)} allocates when the caller doesn't pick one: a prefill of
     * up to this many tokens ingests in a single batch; longer prompts are re-chunked by the
     * caller. Defaults to {@link RuntimeFlags#BATCH_CAPACITY}; override with {@code
     * -Djinfer.batchCapacity} (read at run time, JVM or native).
     */
    default S newState(int contextCapacity) {
        return newState(contextCapacity, RuntimeFlags.BATCH_CAPACITY);
    }

    /** As {@link #newState(int)} over a caller-owned (borrowed) arena. */
    default S newState(int contextCapacity, java.lang.foreign.Arena arena) {
        return newState(contextCapacity, RuntimeFlags.BATCH_CAPACITY, arena);
    }

    /**
     * Ingest one batch at the state's cursor ({@link RuntimeState#position()}), advancing it, and
     * retain the final hidden states selected by {@link Batch#outputs()}. The {@link Batch.Input}
     * union is the multi-modal seam.
     */
    default void ingest(S state, Batch batch) {
        BaseState base = (BaseState) state;
        base.enter();
        try {
            forward(state, batch);
        } finally {
            base.exit();
        }
        java.lang.ref.Reference.reachabilityFence(this);
    }

    /** The forward pass behind {@link #ingest} - the implementation seam, never called directly. */
    void forward(S state, Batch batch);
}
