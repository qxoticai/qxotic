package com.qxotic.jinfer;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * Copies resumable model context between a live state and opaque memory. Public operations own
 * exclusive state access and validate the span and exact memory size; implementations provide only
 * the representation transfer.
 *
 * <p>{@code byteSize(0)} is the fixed endpoint state duplicated by every checkpoint. The prompt
 * cache uses that cost to choose its write strategy.
 */
public abstract class CheckpointCodec<S extends ContextState> {

    /** Bytes needed to capture a checkpoint spanning {@code positions}. */
    public final long byteSize(int positions) {
        if (positions < 0) throw new IllegalArgumentException("negative positions: " + positions);
        return sizeOf(positions);
    }

    /** Captures {@code [from,to)} from a state positioned at {@code to}. */
    public final void capture(S state, int from, int to, MemorySegment destination) {
        Objects.requireNonNull(state, "state");
        Objects.requireNonNull(destination, "destination");
        state.exclusively(() -> transferChecked(state, from, to, destination, true));
    }

    /** Restores {@code [from,to)} without changing the state's logical position. */
    public final void restore(S state, int from, int to, MemorySegment source) {
        Objects.requireNonNull(state, "state");
        Objects.requireNonNull(source, "source");
        state.exclusively(() -> transferChecked(state, from, to, source, false));
    }

    /** Model-private size calculation after non-negative validation. */
    protected abstract long sizeOf(int positions);

    /** Model-private transfer after exclusive-access, span and memory validation. */
    protected abstract void transfer(
            S state, int from, int to, MemorySegment memory, boolean capture);

    private void transferChecked(S state, int from, int to, MemorySegment memory, boolean capture) {
        if (from < 0 || from > to || to > state.contextCapacity()) {
            throw new IllegalArgumentException(
                    "state span ["
                            + from
                            + ","
                            + to
                            + ") outside context capacity "
                            + state.contextCapacity());
        }
        long expectedBytes = byteSize(to - from);
        if (memory.byteSize() != expectedBytes) {
            throw new IllegalArgumentException(
                    "checkpoint bytes " + memory.byteSize() + " != " + expectedBytes);
        }
        if (capture && state.position() != to) {
            throw new IllegalStateException(
                    "state position " + state.position() + " != checkpoint endpoint " + to);
        }
        transfer(state, from, to, memory, capture);
    }
}
