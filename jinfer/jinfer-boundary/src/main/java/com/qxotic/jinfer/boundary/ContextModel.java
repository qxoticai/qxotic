package com.qxotic.jinfer.boundary;

import com.qxotic.jota.memory.MemoryArena;
import java.lang.foreign.MemorySegment;
import java.util.Optional;

/** A model that incrementally ingests batches into a bounded context. */
public interface ContextModel<C extends ContextConfiguration, W, S extends ContextState>
        extends Model<C, W, S> {

    /**
     * Creates a state that owns its memory.
     *
     * @param contextCapacity positive positions allocated for the state; configuration sentinels
     *     such as {@code 0} must be resolved before this call
     */
    S newState(int contextCapacity, int batchCapacity);

    /**
     * Creates a state that borrows caller-owned memory. The arena is the state's memory SOURCE, not
     * just a lifetime scope: a GPU-shared arena (Metal, unified memory) places the KV cache and
     * activations where device kernels can reach them. Pinned to {@code MemorySegment} -
     * host-addressable memory only; device-private memory is a different engine's job.
     *
     * @param contextCapacity positive positions allocated for the state; configuration sentinels
     *     such as {@code 0} must be resolved before this call
     */
    S newState(int contextCapacity, int batchCapacity, MemoryArena<MemorySegment> arena);

    default S newState(int contextCapacity) {
        return newState(contextCapacity, RuntimeFlags.BATCH_CAPACITY);
    }

    default S newState(int contextCapacity, MemoryArena<MemorySegment> arena) {
        return newState(contextCapacity, RuntimeFlags.BATCH_CAPACITY, arena);
    }

    /**
     * Safely ingests one batch and advances the context only after a successful forward.
     *
     * <p><b>Implementation contract:</b> validate, compute and mutate the state while holding
     * {@link RuntimeState#exclusively(Runnable) exclusive access}; advance position and output
     * metadata only after computation succeeds. Keep the model strongly reachable until the call
     * completes, normally through {@link java.lang.ref.Reference#reachabilityFence(Object)}.
     */
    void ingest(S state, Batch batch);

    /** Optional model-specific context checkpoint codec. */
    default Optional<CheckpointCodec<S>> checkpointCodec() {
        return Optional.empty();
    }
}
