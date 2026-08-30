package com.qxotic.jinfer.models.bailingmoe3;

import com.qxotic.jinfer.CheckpointCodec;
import com.qxotic.jinfer.kernels.KvTransfer;
import com.qxotic.jota.DataType;
import java.lang.foreign.MemorySegment;

/** Checkpoint layout for MLA cache rows plus the fixed KDA and optional MTP endpoint state. */
final class BailingMoe3CheckpointCodec extends CheckpointCodec<BailingMoe3.State> {
    private final BailingMoe3.Configuration config;
    private final int convFloats;
    private final int recurrentFloats;
    private final long bytesPerPosition;
    private final long residueBytes;

    BailingMoe3CheckpointCodec(BailingMoe3.Configuration config) {
        this.config = config;
        convFloats = Math.multiplyExact(3 * (config.convKernel() - 1), config.kdaInnerSize());
        recurrentFloats =
                Math.multiplyExact(
                        config.numberOfHeads(),
                        Math.multiplyExact(config.kdaHeadDim(), config.kdaHeadDim()));
        int attentionLayers = 0;
        int recurrentLayers = 0;
        for (int layer = 0; layer < config.storedLayers(); layer++) {
            if (config.isAttention()[layer]) attentionLayers++;
            else recurrentLayers++;
        }
        bytesPerPosition =
                Math.multiplyExact((long) attentionLayers * Short.BYTES, config.mlaCacheDim());
        residueBytes =
                Math.addExact(
                        Math.multiplyExact(
                                (long) recurrentLayers * (recurrentFloats + convFloats),
                                Float.BYTES),
                        config.hasMtp()
                                ? Math.multiplyExact((long) config.embeddingLength(), Float.BYTES)
                                : 0);
    }

    @Override
    protected long sizeOf(int positions) {
        return Math.addExact(Math.multiplyExact((long) positions, bytesPerPosition), residueBytes);
    }

    @Override
    protected void transfer(
            BailingMoe3.State state, int from, int to, MemorySegment blob, boolean capture) {
        long offset = 0;
        long elements = Math.multiplyExact((long) (to - from), config.mlaCacheDim());
        long elementOffset = Math.multiplyExact((long) from, config.mlaCacheDim());
        for (int layer = 0; layer < config.storedLayers(); layer++) {
            if (!config.isAttention()[layer]) continue;
            offset +=
                    KvTransfer.transfer(
                            state.attentionCache[layer],
                            elementOffset,
                            blob,
                            offset,
                            elements,
                            capture);
        }
        int convElements = (config.convKernel() - 1) * config.kdaInnerSize();
        for (int layer = 0; layer < config.storedLayers(); layer++) {
            if (config.isAttention()[layer]) continue;
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
                            state.qConvState[layer],
                            DataType.FP32,
                            0,
                            blob,
                            offset,
                            convElements,
                            capture);
            offset +=
                    KvTransfer.transfer(
                            state.kConvState[layer],
                            DataType.FP32,
                            0,
                            blob,
                            offset,
                            convElements,
                            capture);
            offset +=
                    KvTransfer.transfer(
                            state.vConvState[layer],
                            DataType.FP32,
                            0,
                            blob,
                            offset,
                            convElements,
                            capture);
        }
        if (config.hasMtp())
            KvTransfer.transfer(
                    state.pendingHidden,
                    DataType.FP32,
                    0,
                    blob,
                    offset,
                    config.embeddingLength(),
                    capture);
    }
}
