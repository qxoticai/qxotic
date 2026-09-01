package com.qxotic.jinfer.models.laguna;

import com.qxotic.jinfer.CheckpointCodec;
import com.qxotic.jinfer.kernels.KvTransfer;
import java.lang.foreign.MemorySegment;

/** Checkpoint codec for Laguna's full-attention rows and sliding-window KV rings. */
final class LagunaCheckpointCodec extends CheckpointCodec<Laguna.State> {
    private final Laguna.Configuration config;
    private final long bytesPerPosition;

    LagunaCheckpointCodec(Laguna.Configuration config) {
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
            Laguna.State state, int from, int to, MemorySegment blob, boolean capture) {
        long offset = 0;
        int kvDim = config.kvDim();
        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            if (config.isSwa()[layer]) {
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
