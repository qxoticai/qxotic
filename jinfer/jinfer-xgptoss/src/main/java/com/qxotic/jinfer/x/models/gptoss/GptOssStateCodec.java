package com.qxotic.jinfer.x.models.gptoss;

import com.qxotic.jinfer.x.boundary.StateCodec;
import com.qxotic.jinfer.x.kernels.KvTransfer;
import java.lang.foreign.MemorySegment;

/**
 * gpt-oss attention history: per-position K/V rows, alternating sliding-window/full attention,
 * uniform kvDim, no residue. Full-attention layers store rows at absolute offsets; sliding-window
 * layers store them THROUGH their ring slots ({@code pos & (W-1)}), so the live window rebuilds
 * from restored rows alone and every block is a resume point. Spans longer than W alias ring slots;
 * see {@link KvTransfer#ringSpan} for why that is safe in both directions.
 */
final class GptOssStateCodec implements StateCodec<GptOss.State> {

    private final GptOss.Configuration config;
    private final long bytesPerPosition;

    GptOssStateCodec(GptOss.Configuration config) {
        this.config = config;
        bytesPerPosition =
                Math.multiplyExact(
                        2L * Short.BYTES * config.numberOfLayers(), (long) config.kvDim());
    }

    @Override
    public long checkpointBytes(int positions) {
        if (positions < 0) throw new IllegalArgumentException("positions " + positions);
        return Math.multiplyExact((long) positions, bytesPerPosition);
    }

    @Override
    public void saveCheckpoint(GptOss.State state, int from, int to, MemorySegment destination) {
        transfer(state, from, to, destination, true);
    }

    @Override
    public void restoreCheckpoint(GptOss.State state, int from, int to, MemorySegment source) {
        transfer(state, from, to, source, false);
    }

    private void transfer(GptOss.State state, int from, int to, MemorySegment blob, boolean save) {
        StateCodec.requireCheckpoint(this, state, from, to, blob, save);
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
                                save);
                offset +=
                        KvTransfer.ringSpan(
                                state.valueCache[l],
                                from,
                                to,
                                config.slidingWindow(),
                                kvDim,
                                blob,
                                offset,
                                save);
            } else {
                long elements = Math.multiplyExact((long) (to - from), kvDim);
                long elementOffset = Math.multiplyExact((long) from, kvDim);
                offset +=
                        KvTransfer.transfer(
                                state.keyCache[l], elementOffset, blob, offset, elements, save);
                offset +=
                        KvTransfer.transfer(
                                state.valueCache[l], elementOffset, blob, offset, elements, save);
            }
        }
    }
}
