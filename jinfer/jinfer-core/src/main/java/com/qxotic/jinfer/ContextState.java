package com.qxotic.jinfer;

import com.qxotic.jota.memory.MemoryArena;
import java.lang.foreign.MemorySegment;
import java.lang.ref.Cleaner;
import java.util.concurrent.atomic.AtomicBoolean;

/** Mutable state of one bounded, positioned and resumable model context. */
public abstract class ContextState extends RuntimeState {

    private static final Cleaner CLEANER = Cleaner.create();

    static final String FREED_MESSAGE =
            "the state's buffers have been freed - the arena passed to newState must outlive the"
                    + " state (close your arena last)";

    private final MemoryArena<MemorySegment> memoryArena;
    private final Cleaner.Cleanable cleanup;
    private final AtomicBoolean closeMarker;
    private final int contextCapacity;
    private final int batchCapacity;
    private int position;
    private int outputCount;
    private int lastBatchSize;

    protected ContextState(
            int contextCapacity,
            int batchCapacity,
            MemoryArena<MemorySegment> arena,
            boolean ownsArena) {
        if (contextCapacity <= 0) throw new IllegalArgumentException("contextCapacity must be > 0");
        if (batchCapacity <= 0) throw new IllegalArgumentException("batchCapacity must be > 0");
        if (arena == null) throw new IllegalArgumentException("null arena");
        this.contextCapacity = contextCapacity;
        this.batchCapacity = batchCapacity;
        this.memoryArena = arena;
        if (ownsArena) {
            AtomicBoolean closed = new AtomicBoolean();
            Throwable site = LeakWatch.site("owned state arena");
            cleanup =
                    CLEANER.register(
                            this,
                            () -> {
                                if (site != null && !closed.get() && arena.isAlive()) {
                                    LeakWatch.report(site);
                                }
                                Arenas.close(arena);
                            });
            closeMarker = closed;
        } else {
            cleanup = null;
            closeMarker = null;
        }
    }

    protected final MemoryArena<MemorySegment> memoryArena() {
        return memoryArena;
    }

    /** Positive number of context positions allocated for this state. */
    public final int contextCapacity() {
        return contextCapacity;
    }

    public final int batchCapacity() {
        return batchCapacity;
    }

    public final int position() {
        return position;
    }

    public final int outputCount() {
        return outputCount;
    }

    public final int lastBatchSize() {
        return lastBatchSize;
    }

    public final void resumeAt(int position) {
        if (position < 0 || position > contextCapacity) {
            throw new IllegalArgumentException(
                    "position " + position + " outside 0.." + contextCapacity);
        }
        exclusively(
                () -> {
                    this.position = position;
                    outputCount = 0;
                    lastBatchSize = 0;
                });
    }

    public final void reset() {
        exclusively(
                () -> {
                    clearHistory();
                    position = 0;
                    outputCount = 0;
                    lastBatchSize = 0;
                });
    }

    /** Advances context bookkeeping after a successful model forward. */
    protected final void advanceContext(int batchSize, Batch.Outputs outputs) {
        if (batchSize <= 0 || batchSize > batchCapacity) {
            throw new IllegalArgumentException(
                    "batch size " + batchSize + " outside 1.." + batchCapacity);
        }
        if (position + batchSize > contextCapacity) {
            throw new IllegalArgumentException(
                    "batch ends at "
                            + (position + batchSize)
                            + " beyond context capacity "
                            + contextCapacity);
        }
        lastBatchSize = batchSize;
        outputCount = outputs == Batch.Outputs.ALL ? batchSize : 1;
        position += batchSize;
    }

    /** Clears history that cannot be discarded by moving the logical cursor to zero. */
    protected abstract void clearHistory();

    @Override
    protected final void checkResourcesAlive() {
        if (!memoryArena.isAlive()) throw new IllegalStateException(FREED_MESSAGE);
    }

    @Override
    protected final void releaseResources() {
        if (cleanup == null) return;
        closeMarker.set(true);
        cleanup.clean();
    }
}
