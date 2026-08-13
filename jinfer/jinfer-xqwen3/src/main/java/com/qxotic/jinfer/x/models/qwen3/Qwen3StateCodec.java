package com.qxotic.jinfer.x.models.qwen3;

import com.qxotic.jinfer.x.boundary.StateCodec;
import com.qxotic.jinfer.x.kernels.KvTransfer;
import java.lang.foreign.MemorySegment;

/** Qwen3 attention history: per-position K/V rows - every layer is full attention, no residue. */
final class Qwen3StateCodec implements StateCodec<Qwen3.State> {

    private final int layers;
    private final int kvDim;
    private final long bytesPerPosition;

    Qwen3StateCodec(Qwen3.Configuration config) {
        layers = config.numberOfLayers();
        kvDim = config.kvDim();
        bytesPerPosition = Math.multiplyExact(2L * Short.BYTES * layers, kvDim);
    }

    @Override
    public long checkpointBytes(int positions) {
        if (positions < 0) throw new IllegalArgumentException("positions " + positions);
        return Math.multiplyExact((long) positions, bytesPerPosition);
    }

    @Override
    public void saveCheckpoint(Qwen3.State state, int from, int to, MemorySegment destination) {
        transfer(state, from, to, destination, true);
    }

    @Override
    public void restoreCheckpoint(Qwen3.State state, int from, int to, MemorySegment source) {
        transfer(state, from, to, source, false);
    }

    private void transfer(Qwen3.State state, int from, int to, MemorySegment blob, boolean save) {
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
