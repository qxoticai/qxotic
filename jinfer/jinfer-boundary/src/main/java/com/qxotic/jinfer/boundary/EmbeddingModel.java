package com.qxotic.jinfer.boundary;

import com.qxotic.jota.memory.MemoryView;
import java.util.Arrays;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * A context model with a pooled semantic-embedding head.
 *
 * <p>Consumers run synchronously on the calling thread while the state is held exclusively. Each
 * embedding is a state-owned borrowed view, guaranteed live and unchanged only until its consumer
 * returns; copy it to retain it. Other-thread state operations fail fast, while a close from
 * another thread waits. Consumers must not invoke model operations on the same state because nested
 * operations may reuse its scratch memory. Consumer failures propagate and exclusive access is
 * still released.
 */
public interface EmbeddingModel<C extends ContextConfiguration, W, S extends ContextState>
        extends ContextModel<C, W, S> {

    /**
     * Projects one output retained by the most recent ingestion.
     *
     * <p>{@code outputIndex} is relative to the retained outputs and must be in {@code [0,
     * state.outputCount())}.
     *
     * <p><b>Implementation contract:</b> validate, project and invoke {@code consumer}
     * synchronously on the calling thread while holding {@link RuntimeState#exclusively(Runnable)
     * exclusive access}. The consumer's return ends the projected view's validity. Release access
     * if projection or the consumer fails, and keep the model strongly reachable until the call
     * completes, normally through {@link java.lang.ref.Reference#reachabilityFence(Object)}.
     */
    void projectEmbedding(S state, int outputIndex, Consumer<MemoryView<?>> consumer);

    /** Projects the final output retained by the most recent ingestion. */
    default void projectEmbedding(S state, Consumer<MemoryView<?>> consumer) {
        Objects.requireNonNull(consumer, "consumer");
        state.exclusively(
                () -> {
                    int outputs = state.outputCount();
                    if (outputs == 0)
                        throw new IllegalStateException("state has no retained outputs");
                    projectEmbedding(state, outputs - 1, consumer);
                });
    }

    /**
     * Embeds packed sequences, delivering one borrowed view per sequence in input order while
     * holding the state for the entire operation.
     */
    default void embedAll(S state, Batch.Input.Sequences sequences, Consumer<MemoryView<?>> sink) {
        Objects.requireNonNull(sequences, "sequences");
        Objects.requireNonNull(sink, "sink");
        requireComplete(sequences);
        state.exclusively(() -> embedAll0(state, sequences, sink));
    }

    private static void requireComplete(Batch.Input.Sequences sequences) {
        long total = 0;
        int[] lengths = sequences.seqLen();
        for (int i = 0; i < lengths.length; i++) {
            if (lengths[i] <= 0)
                throw new IllegalArgumentException(
                        "sequence " + i + " has invalid length " + lengths[i]);
            total += lengths[i];
        }
        int tokens = sequences.tokens().ids().length;
        if (total != tokens)
            throw new IllegalArgumentException(
                    "packed token count " + tokens + " != sequence lengths " + total);
    }

    private void embedAll0(S state, Batch.Input.Sequences sequences, Consumer<MemoryView<?>> sink) {
        int[] lengths = sequences.seqLen();
        int[] ids = sequences.tokens().ids();
        if (ids.length > state.contextCapacity()) {
            throw new IllegalArgumentException(
                    "state contextCapacity "
                            + state.contextCapacity()
                            + " < packed length "
                            + ids.length);
        }
        state.reset();
        int sequence = 0;
        int sequenceStart = 0;
        int batchCapacity = state.batchCapacity();
        for (int from = 0; from < ids.length; from += batchCapacity) {
            int to = Math.min(from + batchCapacity, ids.length);
            ingest(
                    state,
                    new Batch(
                            new Batch.Input.Sequences(
                                    new Batch.Input.Tokens(Arrays.copyOfRange(ids, from, to)),
                                    lengths),
                            Batch.Outputs.ALL));
            while (sequence < lengths.length && sequenceStart + lengths[sequence] - 1 < to) {
                projectEmbedding(state, sequenceStart + lengths[sequence] - 1 - from, sink);
                sequenceStart += lengths[sequence++];
            }
        }
    }
}
