package com.qxotic.jinfer.x.models.llama;

import com.qxotic.jinfer.x.boundary.StateCodec;
import com.qxotic.jinfer.x.kernels.KvTransfer;
import java.lang.foreign.MemorySegment;

/**
 * Granite resume-state codec: uniform full attention, per-position K/V rows at absolute offsets
 * (linear indexing, no ring, no recurrent residue), so every block is a resume point. RoPE is baked
 * into K at absolute positions, so a restored row is valid at its true slot regardless of when it
 * was saved.
 */
final class GraniteStateCodec implements StateCodec<Granite.State> {

    private final int layers;
    private final int kvDim;

    GraniteStateCodec(Granite.Configuration config) {
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
    public void saveCheckpoint(Granite.State state, int from, int to, MemorySegment destination) {
        transfer(state, from, to, destination, true);
    }

    @Override
    public void restoreCheckpoint(Granite.State state, int from, int to, MemorySegment source) {
        transfer(state, from, to, source, false);
    }

    private void transfer(Granite.State state, int from, int to, MemorySegment blob, boolean save) {
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
