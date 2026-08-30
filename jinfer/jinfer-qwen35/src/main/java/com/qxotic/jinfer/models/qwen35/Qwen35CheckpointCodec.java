package com.qxotic.jinfer.models.qwen35;

import com.qxotic.jinfer.CheckpointCodec;
import com.qxotic.jinfer.kernels.KvTransfer;
import com.qxotic.jota.DataType;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Qwen3.5 resume-state codec: the full-attention layers store per-position K/V rows (full context,
 * linear position indexing - no ring); the gated-delta-net layers contribute their conv history
 * plus the S matrices ({@code heads x headVDim x headVDim} F32) as ~66MB of fixed checkpoint
 * overhead at 35B-A3B dims (30 linear layers plus conv). When present, the embedded MTP block
 * contributes ordinary per-position K/V rows plus one normalized target-hidden row to that fixed
 * state; no separate cache format or mode exists. MoE routing is per-token and carries no
 * cross-token state; everything else is per-batch scratch.
 */
final class Qwen35CheckpointCodec extends CheckpointCodec<Qwen35.State> {

    private final Qwen35.Configuration config;
    private final int recurrentFloats; // heads * headVDim * headVDim per linear layer
    private final int convFloats; // (ssmConvKernel - 1) * convChannels per linear layer
    private final long bytesPerPosition;
    private final long residueBytes;

    Qwen35CheckpointCodec(Qwen35.Configuration config) {
        this.config = config;
        recurrentFloats =
                Math.multiplyExact(
                        config.ssmTimeStepRank(),
                        Math.multiplyExact(config.headVDim(), config.headVDim()));
        convFloats =
                Math.multiplyExact(Math.max(config.ssmConvKernel() - 1, 0), config.convChannels());
        long rowBytes = 0;
        long linearLayers = 0;
        for (int layer = 0; layer < config.storedLayers(); layer++) {
            if (config.isFullAttention()[layer]) {
                rowBytes =
                        Math.addExact(
                                rowBytes, Math.multiplyExact(2L * Short.BYTES, config.kvDim()));
            } else {
                linearLayers++;
            }
        }
        bytesPerPosition = rowBytes;
        residueBytes =
                Math.addExact(
                        Math.addExact(
                                Math.multiplyExact(
                                        linearLayers,
                                        Math.multiplyExact(
                                                (long) recurrentFloats + convFloats, Float.BYTES)),
                                config.hasMtp()
                                        ? Math.multiplyExact(
                                                (long) config.embeddingLength(), Float.BYTES)
                                        : 0),
                        Integer.BYTES);
    }

    @Override
    protected long sizeOf(int positions) {
        return Math.addExact(Math.multiplyExact((long) positions, bytesPerPosition), residueBytes);
    }

    @Override
    protected void transfer(
            Qwen35.State state, int from, int to, MemorySegment blob, boolean capture) {
        long offset = 0;
        long elements = Math.multiplyExact((long) (to - from), config.kvDim());
        long elementOffset = Math.multiplyExact((long) from, config.kvDim());
        for (int layer = 0; layer < config.storedLayers(); layer++) {
            if (!config.isFullAttention()[layer]) continue;
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
        for (int layer = 0; layer < config.storedLayers(); layer++) {
            if (config.isFullAttention()[layer]) continue;
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
        if (config.hasMtp())
            offset +=
                    KvTransfer.transfer(
                            state.pendingHidden,
                            DataType.FP32,
                            0,
                            blob,
                            offset,
                            config.embeddingLength(),
                            capture);
        if (capture) blob.set(ValueLayout.JAVA_INT_UNALIGNED, offset, state.ropeDelta);
        else state.ropeDelta = blob.get(ValueLayout.JAVA_INT_UNALIGNED, offset);
    }
}
