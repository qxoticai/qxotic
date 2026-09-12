package com.qxotic.jinfer.models.mellum;

import com.qxotic.jinfer.CheckpointCodec;
import com.qxotic.jinfer.kernels.KvTransfer;
import java.lang.foreign.MemorySegment;

/**
 * Mellum's attention history: per-position FP16 K/V rows of uniform {@code kvDim}, no residue.
 * Full-attention layers store rows at absolute offsets; sliding-window layers store them THROUGH
 * their ring slots ({@code pos & (W-1)}), so the live window rebuilds from restored rows alone and
 * every block is a resume point. See {@link KvTransfer#ringSpan} for why spans longer than the
 * window are safe in both directions.
 */
final class MellumCheckpointCodec extends CheckpointCodec<Mellum.State> {
    private final Mellum.Configuration config;
    private final long bytesPerPosition;

    MellumCheckpointCodec(Mellum.Configuration config) {
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
            Mellum.State state, int from, int to, MemorySegment blob, boolean capture) {
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
