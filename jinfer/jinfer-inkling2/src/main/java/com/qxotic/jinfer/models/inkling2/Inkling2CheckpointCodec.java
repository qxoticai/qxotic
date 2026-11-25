package com.qxotic.jinfer.models.inkling2;

import com.qxotic.jinfer.CheckpointCodec;
import com.qxotic.jinfer.kernels.KvTransfer;
import com.qxotic.jota.DataType;
import java.lang.foreign.MemorySegment;

/**
 * Inkling 2's history, per cache slot: FP16 K/V rows per position, then the convolution inputs at
 * the block endpoint.
 */
final class Inkling2CheckpointCodec extends CheckpointCodec<Inkling2.State> {
    private final Inkling2.Configuration config;
    private final long bytesPerPosition;
    private final long convElements;

    Inkling2CheckpointCodec(Inkling2.Configuration config) {
        this.config = config;
        bytesPerPosition = Math.multiplyExact(2L * Short.BYTES * config.slots(), config.kvDim());
        convElements =
                (config.convLength() - 1) * (2L * config.kvDim() + 2L * config.embeddingLength());
    }

    @Override
    protected long sizeOf(int positions) {
        return Math.addExact(
                Math.multiplyExact((long) positions, bytesPerPosition),
                config.slots() * convElements * Float.BYTES);
    }

    @Override
    protected void transfer(
            Inkling2.State state, int from, int to, MemorySegment blob, boolean capture) {
        long offset = 0;
        long elements = Math.multiplyExact((long) (to - from), config.kvDim());
        long at = Math.multiplyExact((long) from, config.kvDim());
        for (int slot = 0; slot < config.slots(); slot++) {
            offset +=
                    KvTransfer.transfer(state.keyCache[slot], at, blob, offset, elements, capture);
            offset +=
                    KvTransfer.transfer(
                            state.valueCache[slot], at, blob, offset, elements, capture);
            offset +=
                    KvTransfer.transfer(
                            state.conv[slot],
                            DataType.FP32,
                            0,
                            blob,
                            offset,
                            convElements,
                            capture);
        }
    }
}
