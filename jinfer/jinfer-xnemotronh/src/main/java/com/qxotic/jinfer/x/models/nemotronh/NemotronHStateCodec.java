package com.qxotic.jinfer.x.models.nemotronh;

import com.qxotic.jinfer.x.boundary.StateCodec;
import com.qxotic.jinfer.x.kernels.KvTransfer;
import com.qxotic.jota.DataType;
import java.lang.foreign.MemorySegment;

/**
 * Nemotron-H resume-state codec: full-attention layers store per-position K/V rows (full context,
 * linear position indexing - no ring); Mamba2 layers contribute their conv ring plus the SSM state
 * ({@code ssmInnerSize x ssmStateSize} F32) as the block RESIDUE - MBs per SSM layer, so this model
 * is {@link #coarseCheckpoints()}: cached prompts commit as ONE block per prompt (one residue per
 * prompt, not one per turn). MoE routing is per-token and carries no cross-token state.
 */
final class NemotronHStateCodec implements StateCodec<NemotronH.State> {

    private final NemotronH.Configuration config;
    private final int recurrentFloats; // ssmInnerSize * ssmStateSize per SSM layer
    private final int convFloats; // (ssmConvKernel - 1) * ssmConvChannels per SSM layer
    private final long bytesPerPosition;
    private final long residueBytes;

    NemotronHStateCodec(NemotronH.Configuration config) {
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
    public long checkpointBytes(int positions) {
        if (positions < 0) throw new IllegalArgumentException("positions " + positions);
        return Math.addExact(Math.multiplyExact((long) positions, bytesPerPosition), residueBytes);
    }

    @Override
    public boolean coarseCheckpoints() {
        return true;
    }

    @Override
    public void saveCheckpoint(NemotronH.State state, int from, int to, MemorySegment destination) {
        transfer(state, from, to, destination, true);
    }

    @Override
    public void restoreCheckpoint(NemotronH.State state, int from, int to, MemorySegment source) {
        transfer(state, from, to, source, false);
    }

    private void transfer(
            NemotronH.State state, int from, int to, MemorySegment blob, boolean save) {
        StateCodec.requireCheckpoint(this, state, from, to, blob, save);
        long offset = 0;
        long elements = Math.multiplyExact((long) (to - from), config.kvDim());
        long elementOffset = Math.multiplyExact((long) from, config.kvDim());
        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            if (config.layerTypes()[layer] != NemotronH.LayerType.ATTENTION) continue;
            offset +=
                    KvTransfer.transfer(
                            state.keyCache[layer], elementOffset, blob, offset, elements, save);
            offset +=
                    KvTransfer.transfer(
                            state.valueCache[layer], elementOffset, blob, offset, elements, save);
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
                            save);
            offset +=
                    KvTransfer.transfer(
                            state.convState[layer],
                            DataType.FP32,
                            0,
                            blob,
                            offset,
                            convFloats,
                            save);
        }
    }
}
