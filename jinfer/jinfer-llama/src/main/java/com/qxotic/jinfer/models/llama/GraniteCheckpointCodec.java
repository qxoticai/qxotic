package com.qxotic.jinfer.models.llama;

import com.qxotic.jinfer.boundary.CheckpointCodec;
import com.qxotic.jinfer.kernels.KvTransfer;
import java.lang.foreign.MemorySegment;

/**
 * Granite resume-state codec: uniform full attention, per-position K/V rows at absolute offsets
 * (linear indexing, no ring, no recurrent residue), so every block is a resume point. RoPE is baked
 * into K at absolute positions, so a restored row is valid at its true slot regardless of when it
 * was saved.
 */
final class GraniteCheckpointCodec extends CheckpointCodec<Granite.State> {

    private final int layers;
    private final int kvDim;

    GraniteCheckpointCodec(Granite.Configuration config) {
        layers = config.numberOfLayers();
        kvDim = config.kvDim();
    }

    @Override
    protected long sizeOf(int positions) {
        return Math.multiplyExact(
                positions, Math.multiplyExact(4L, Math.multiplyExact(layers, kvDim)));
    }

    @Override
    protected void transfer(
            Granite.State state, int from, int to, MemorySegment blob, boolean capture) {
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
