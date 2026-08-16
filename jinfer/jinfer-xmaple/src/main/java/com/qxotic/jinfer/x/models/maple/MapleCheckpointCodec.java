package com.qxotic.jinfer.x.models.maple;

import com.qxotic.jinfer.x.boundary.CheckpointCodec;
import com.qxotic.jinfer.x.kernels.KvTransfer;
import java.lang.foreign.MemorySegment;

/** Checkpoint codec for Maple's sliding-window and global F16 KV rows. */
final class MapleCheckpointCodec extends CheckpointCodec<Maple.State> {
    private final Maple.Configuration config;
    private final long bytesPerPosition;

    MapleCheckpointCodec(Maple.Configuration config) {
        this.config = config;
        bytesPerPosition =
                Math.multiplyExact(
                        2L * Short.BYTES * config.numberOfLayers(), (long) config.kvDim());
    }

    @Override
    protected long sizeOf(int positions) {
        return Math.multiplyExact((long) positions, bytesPerPosition);
    }

    @Override
    protected void transfer(
            Maple.State state, int from, int to, MemorySegment blob, boolean capture) {
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
                                capture);
                offset +=
                        KvTransfer.ringSpan(
                                state.valueCache[layer],
                                from,
                                to,
                                config.slidingWindow(),
                                kvDim,
                                blob,
                                offset,
                                capture);
            } else {
                long elements = Math.multiplyExact((long) (to - from), kvDim);
                long elementOffset = Math.multiplyExact((long) from, kvDim);
                offset +=
                        KvTransfer.transfer(
                                state.keyCache[layer],
                                elementOffset,
                                blob,
                                offset,
                                elements,
                                capture);
                offset +=
                        KvTransfer.transfer(
                                state.valueCache[layer],
                                elementOffset,
                                blob,
                                offset,
                                elements,
                                capture);
            }
        }
    }
}
