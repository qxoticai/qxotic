package com.qxotic.jinfer.x.models.llama;

import com.qxotic.jinfer.x.boundary.CheckpointCodec;
import com.qxotic.jinfer.x.kernels.KvTransfer;
import java.lang.foreign.MemorySegment;

final class LlamaCheckpointCodec extends CheckpointCodec<Llama.State> {

    private final int layers;
    private final int kvDim;

    LlamaCheckpointCodec(Llama.Configuration config) {
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
            Llama.State state, int from, int to, MemorySegment blob, boolean capture) {
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
