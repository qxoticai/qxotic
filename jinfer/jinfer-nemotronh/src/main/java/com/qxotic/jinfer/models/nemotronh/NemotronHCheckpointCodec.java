package com.qxotic.jinfer.models.nemotronh;

import com.qxotic.jinfer.CheckpointCodec;
import com.qxotic.jinfer.kernels.KvTransfer;
import com.qxotic.jota.DataType;
import java.lang.foreign.MemorySegment;

/**
 * Nemotron-H resume-state codec: full-attention layers store per-position K/V rows (full context,
 * linear position indexing - no ring); Mamba2 layers contribute their conv ring plus the SSM state
 * ({@code ssmInnerSize x ssmStateSize} F32) as a large fixed checkpoint overhead. MoE routing is
 * per-token and carries no cross-token state.
 */
final class NemotronHCheckpointCodec extends CheckpointCodec<NemotronH.State> {

    private final NemotronH.Configuration config;
    private final int recurrentFloats; // ssmInnerSize * ssmStateSize per SSM layer
    private final int convFloats; // (ssmConvKernel - 1) * ssmConvChannels per SSM layer
    private final long bytesPerPosition;
    private final long residueBytes;

    NemotronHCheckpointCodec(NemotronH.Configuration config) {
        this.config = config;
        recurrentFloats = Math.multiplyExact(config.ssmInnerSize(), config.ssmStateSize());
        convFloats =
                Math.multiplyExact(
                        Math.max(config.ssmConvKernel() - 1, 0), config.ssmConvChannels());
        long rowBytes = 0;
        long ssmLayers = 0;
        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            switch (config.layerTypes()[layer]) {
                case ATTENTION ->
                        rowBytes =
                                Math.addExact(
                                        rowBytes,
                                        Math.multiplyExact(2L * Short.BYTES, config.kvDim()));
                case SSM -> ssmLayers++;
                case MOE -> {}
            }
        }
        bytesPerPosition = rowBytes;
        residueBytes =
                Math.multiplyExact(
                        ssmLayers,
                        Math.multiplyExact((long) recurrentFloats + convFloats, Float.BYTES));
    }

    @Override
    protected long sizeOf(int positions) {
        return Math.addExact(Math.multiplyExact((long) positions, bytesPerPosition), residueBytes);
    }

    @Override
    protected void transfer(
            NemotronH.State state, int from, int to, MemorySegment blob, boolean capture) {
        long offset = 0;
        long elements = Math.multiplyExact((long) (to - from), config.kvDim());
        long elementOffset = Math.multiplyExact((long) from, config.kvDim());
        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            if (config.layerTypes()[layer] != NemotronH.LayerType.ATTENTION) continue;
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
        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            if (config.layerTypes()[layer] != NemotronH.LayerType.SSM) continue;
            offset +=
                    KvTransfer.transfer(
                            state.recurrentState[layer],
                            DataType.FP32,
                            0,
                            blob,
                            offset,
                            recurrentFloats,
                            capture);
            offset +=
                    KvTransfer.transfer(
                            state.convState[layer],
                            DataType.FP32,
                            0,
                            blob,
                            offset,
                            convFloats,
                            capture);
        }
    }
}
