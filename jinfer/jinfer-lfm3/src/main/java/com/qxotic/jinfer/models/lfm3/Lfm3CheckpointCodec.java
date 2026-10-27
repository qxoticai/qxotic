package com.qxotic.jinfer.models.lfm3;

import com.qxotic.jinfer.CheckpointCodec;
import com.qxotic.jinfer.kernels.KvTransfer;
import com.qxotic.jota.DataType;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;

/**
 * LFM3 history: full-attention KV rows, the sliding-window rings walked by position, and the
 * short-conv taps at the block's end.
 */
final class Lfm3CheckpointCodec extends CheckpointCodec<Lfm3.State> {

    private final Lfm3.Configuration config;
    private final int convElements;
    private final long bytesPerPosition, convBytes;

    Lfm3CheckpointCodec(Lfm3.Configuration config) {
        this.config = config;
        convElements = Math.multiplyExact(config.shortConvLCache() - 1, config.embeddingLength());
        long rowBytes = 0, conv = 0;
        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            if (config.isShortConv(layer)) conv += (long) convElements * Float.BYTES;
            else rowBytes += 2L * Short.BYTES * config.kvDim(layer);
        }
        bytesPerPosition = rowBytes;
        convBytes = conv;
    }

    @Override
    protected long sizeOf(int positions) {
        return Math.addExact(Math.multiplyExact((long) positions, bytesPerPosition), convBytes);
    }

    @Override
    protected void transfer(
            Lfm3.State state, int from, int to, MemorySegment blob, boolean capture) {
        long offset = 0;
        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            if (config.isShortConv(layer)) continue;
            offset += rows(state.keyCache[layer], layer, from, to, blob, offset, capture);
            offset += rows(state.valueCache[layer], layer, from, to, blob, offset, capture);
        }
        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            if (!config.isShortConv(layer)) continue;
            offset +=
                    KvTransfer.transfer(
                            state.shortConvState[layer],
                            DataType.FP32,
                            0,
                            blob,
                            offset,
                            convElements,
                            capture);
        }
    }

    /** Positions {@code [from, to)} of one cache, through the ring on a sliding-window layer. */
    private long rows(
            MemoryView<MemorySegment> cache,
            int layer,
            int from,
            int to,
            MemorySegment blob,
            long offset,
            boolean capture) {
        int kvDim = config.kvDim(layer);
        if (config.isSwa(layer))
            return KvTransfer.ringSpan(
                    cache, from, to, config.slidingWindow(), kvDim, blob, offset, capture);
        return KvTransfer.transfer(
                cache, (long) from * kvDim, blob, offset, (long) (to - from) * kvDim, capture);
    }
}
