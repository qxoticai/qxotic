package com.qxotic.jinfer.x.models.llama;

import com.qxotic.jinfer.x.boundary.StateCodec;
import com.qxotic.jinfer.x.kernels.KvTransfer;
import java.lang.foreign.MemorySegment;

final class LlamaStateCodec implements StateCodec<Llama.State> {

    private final int layers;
    private final int kvDim;

    LlamaStateCodec(Llama.Configuration config) {
        layers = config.numberOfLayers();
        kvDim = config.kvDim();
    }

    @Override
    public long checkpointBytes(int positions) {
        if (positions < 0) throw new IllegalArgumentException("positions " + positions);
        return Math.multiplyExact(
                positions, Math.multiplyExact(4L, Math.multiplyExact(layers, kvDim)));
    }

    @Override
    public void saveCheckpoint(Llama.State state, int from, int to, MemorySegment destination) {
        transfer(state, from, to, destination, true);
    }

    @Override
    public void restoreCheckpoint(Llama.State state, int from, int to, MemorySegment source) {
        transfer(state, from, to, source, false);
    }

    private void transfer(Llama.State state, int from, int to, MemorySegment blob, boolean save) {
        StateCodec.requireCheckpoint(this, state, from, to, blob, save);
        long offset = 0;
        long elements = Math.multiplyExact((long) (to - from), kvDim);
        long elementOffset = Math.multiplyExact((long) from, kvDim);
        for (int layer = 0; layer < layers; layer++) {
            offset +=
                    KvTransfer.transfer(
                            state.keyCache[layer], elementOffset, blob, offset, elements, save);
            offset +=
                    KvTransfer.transfer(
                            state.valueCache[layer], elementOffset, blob, offset, elements, save);
        }
    }
}
