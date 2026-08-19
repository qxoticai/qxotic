package com.qxotic.jinfer.models.qwen3;

import com.qxotic.jinfer.CheckpointCodec;
import com.qxotic.jinfer.kernels.KvTransfer;
import java.lang.foreign.MemorySegment;

/** Qwen3 attention history: per-position K/V rows - every layer is full attention, no residue. */
final class Qwen3CheckpointCodec extends CheckpointCodec<Qwen3.State> {

    private final int layers;
    private final int kvDim;
    private final long bytesPerPosition;

    Qwen3CheckpointCodec(Qwen3.Configuration config) {
        layers = config.numberOfLayers();
        kvDim = config.kvDim();
        bytesPerPosition = Math.multiplyExact(2L * Short.BYTES * layers, kvDim);
    }

    @Override
    protected long sizeOf(int positions) {
        return Math.multiplyExact((long) positions, bytesPerPosition);
    }

    @Override
    protected void transfer(
            Qwen3.State state, int from, int to, MemorySegment blob, boolean capture) {
        long offset = 0;
        long elements = Math.multiplyExact((long) (to - from), kvDim);
        long elementOffset = Math.multiplyExact((long) from, kvDim);
        for (int layer = 0; layer < layers; layer++) {
            offset +=
                    KvTransfer.transfer(
                            state.keyCache[layer], elementOffset, blob, offset, elements, capture);
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
