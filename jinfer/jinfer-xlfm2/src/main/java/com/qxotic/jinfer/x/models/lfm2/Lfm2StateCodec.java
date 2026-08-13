package com.qxotic.jinfer.x.models.lfm2;

import com.qxotic.jinfer.x.boundary.StateCodec;
import com.qxotic.jinfer.x.kernels.KvTransfer;
import com.qxotic.jota.DataType;
import java.lang.foreign.MemorySegment;

/** LFM2 attention history plus the short-convolution state at each block endpoint. */
final class Lfm2StateCodec implements StateCodec<Lfm2.State> {

    private final Lfm2.Configuration config;
    private final int convElements;
    private final long bytesPerPosition;
    private final long residueBytes;

    Lfm2StateCodec(Lfm2.Configuration config) {
        this.config = config;
        convElements =
                Math.multiplyExact(
                        Math.max(config.shortConvLCache() - 1, 0), config.embeddingLength());
        long rowBytes = 0;
        long recurrentLayers = 0;
        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            if (config.isRecurrentLayer(layer)) {
                recurrentLayers++;
            } else {
                rowBytes =
                        Math.addExact(
                                rowBytes,
                                Math.multiplyExact(2L * Short.BYTES, config.kvDim(layer)));
            }
        }
        bytesPerPosition = rowBytes;
        residueBytes =
                Math.multiplyExact(
                        recurrentLayers, Math.multiplyExact((long) convElements, Float.BYTES));
    }

    @Override
    public long checkpointBytes(int positions) {
        if (positions < 0) throw new IllegalArgumentException("positions " + positions);
        return Math.addExact(Math.multiplyExact((long) positions, bytesPerPosition), residueBytes);
    }

    @Override
    public void saveCheckpoint(Lfm2.State state, int from, int to, MemorySegment destination) {
        transfer(state, from, to, destination, true);
    }

    @Override
    public void restoreCheckpoint(Lfm2.State state, int from, int to, MemorySegment source) {
        transfer(state, from, to, source, false);
    }

    private void transfer(Lfm2.State state, int from, int to, MemorySegment blob, boolean save) {
        StateCodec.requireCheckpoint(this, state, from, to, blob, save);

        long offset = 0;
        int positions = to - from;
        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            if (config.isRecurrentLayer(layer)) continue;
            int kvDim = config.kvDim(layer);
            long elements = Math.multiplyExact((long) positions, kvDim);
            long elementOffset = Math.multiplyExact((long) from, kvDim);
            offset +=
                    KvTransfer.transfer(
                            state.keyCache[layer], elementOffset, blob, offset, elements, save);
            offset +=
                    KvTransfer.transfer(
                            state.valueCache[layer], elementOffset, blob, offset, elements, save);
        }
        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            if (!config.isRecurrentLayer(layer)) continue;
            offset +=
                    KvTransfer.transfer(
                            state.shortConvState[layer],
                            DataType.FP32,
                            0,
                            blob,
                            offset,
                            convElements,
                            save);
        }
    }
}
