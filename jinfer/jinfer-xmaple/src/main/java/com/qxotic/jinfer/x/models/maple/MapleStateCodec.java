package com.qxotic.jinfer.x.models.maple;

import com.qxotic.jinfer.x.boundary.StateCodec;
import com.qxotic.jinfer.x.kernels.KvTransfer;
import java.lang.foreign.MemorySegment;

/** Checkpoint codec for Maple's sliding-window and global F16 KV rows. */
final class MapleStateCodec implements StateCodec<Maple.State> {
    private final Maple.Configuration config;
    private final long bytesPerPosition;

    MapleStateCodec(Maple.Configuration config) {
        this.config = config;
        bytesPerPosition =
                Math.multiplyExact(
                        2L * Short.BYTES * config.numberOfLayers(), (long) config.kvDim());
    }

    @Override
    public long checkpointBytes(int positions) {
        if (positions < 0) throw new IllegalArgumentException("positions " + positions);
        return Math.multiplyExact((long) positions, bytesPerPosition);
    }

    @Override
    public void saveCheckpoint(Maple.State state, int from, int to, MemorySegment destination) {
        transfer(state, from, to, destination, true);
    }

    @Override
    public void restoreCheckpoint(Maple.State state, int from, int to, MemorySegment source) {
        transfer(state, from, to, source, false);
    }

    private void transfer(Maple.State state, int from, int to, MemorySegment blob, boolean save) {
        StateCodec.requireCheckpoint(this, state, from, to, blob, save);
        long offset = 0;
        int kvDim = config.kvDim();
        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            if (config.isSliding(layer)) {
                offset +=
                        KvTransfer.ringSpan(
                                state.keyCache[layer],
                                from,
                                to,
                                config.slidingWindow(),
                                kvDim,
                                blob,
                                offset,
                                save);
                offset +=
                        KvTransfer.ringSpan(
                                state.valueCache[layer],
                                from,
                                to,
                                config.slidingWindow(),
                                kvDim,
                                blob,
                                offset,
                                save);
            } else {
                long elements = Math.multiplyExact((long) (to - from), kvDim);
                long elementOffset = Math.multiplyExact((long) from, kvDim);
                offset +=
                        KvTransfer.transfer(
                                state.keyCache[layer], elementOffset, blob, offset, elements, save);
                offset +=
                        KvTransfer.transfer(
                                state.valueCache[layer],
                                elementOffset,
                                blob,
                                offset,
                                elements,
                                save);
            }
        }
    }
}
