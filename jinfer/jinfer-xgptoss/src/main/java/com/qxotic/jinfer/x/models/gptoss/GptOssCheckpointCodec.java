package com.qxotic.jinfer.x.models.gptoss;

import com.qxotic.jinfer.x.boundary.CheckpointCodec;
import com.qxotic.jinfer.x.kernels.KvTransfer;
import java.lang.foreign.MemorySegment;

/**
 * gpt-oss attention history: per-position K/V rows, alternating sliding-window/full attention,
 * uniform kvDim, no residue. Full-attention layers store rows at absolute offsets; sliding-window
 * layers store them THROUGH their ring slots ({@code pos & (W-1)}), so the live window rebuilds
 * from restored rows alone and every block is a resume point. Spans longer than W alias ring slots;
 * see {@link KvTransfer#ringSpan} for why that is safe in both directions.
 */
final class GptOssCheckpointCodec extends CheckpointCodec<GptOss.State> {

    private final GptOss.Configuration config;
    private final long bytesPerPosition;

    GptOssCheckpointCodec(GptOss.Configuration config) {
        this.config = config;
        bytesPerPosition =
                Math.multiplyExact(
                        2L * Short.BYTES * config.numberOfLayers(), (long) config.kvDim());
    }

    @Override
    protected long sizeOf(int positions) {
        return Math.multiplyExact((long) positions, bytesPerPosition);
    }

    @Override
    protected void transfer(
            GptOss.State state, int from, int to, MemorySegment blob, boolean capture) {
        long offset = 0;
        int kvDim = config.kvDim();
        for (int l = 0; l < config.numberOfLayers(); l++) {
            if (config.isSWA(l)) {
                offset +=
                        KvTransfer.ringSpan(
                                state.keyCache[l],
                                from,
                                to,
                                config.slidingWindow(),
                                kvDim,
                                blob,
                                offset,
                                capture);
                offset +=
                        KvTransfer.ringSpan(
                                state.valueCache[l],
                                from,
                                to,
                                config.slidingWindow(),
                                kvDim,
                                blob,
                                offset,
                                capture);
            } else {
                long elements = Math.multiplyExact((long) (to - from), kvDim);
                long elementOffset = Math.multiplyExact((long) from, kvDim);
                offset +=
                        KvTransfer.transfer(
                                state.keyCache[l], elementOffset, blob, offset, elements, capture);
                offset +=
                        KvTransfer.transfer(
                                state.valueCache[l],
                                elementOffset,
                                blob,
                                offset,
                                elements,
                                capture);
            }
        }
    }
}
