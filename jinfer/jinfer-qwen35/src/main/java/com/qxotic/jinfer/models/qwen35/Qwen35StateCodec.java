package com.qxotic.jinfer.models.qwen35;

import com.qxotic.jinfer.cache.AbstractStateCodec;
import com.qxotic.jinfer.cache.KvTransfer;
import java.lang.foreign.MemorySegment;

/**
 * Qwen3.5 (dense and MoE) resume-state codec: full-attention layers store per-position K/V rows
 * (shared base); gated-delta-net (SSM) layers checkpoint a fixed-size F32 snapshot — the conv ring
 * plus the delta-net S matrix — that only exists at the state's current position, which is exactly
 * why blocks match completely or not at all. MoE routing is per-row and stateless. The S matrix is
 * a heap array, copied via a heap segment view.
 */
public final class Qwen35StateCodec extends AbstractStateCodec<Qwen35.State> {

    private final Qwen35.Configuration config;
    private final int convFloats; // (convKernel-1) * convChannels, per SSM layer

    public Qwen35StateCodec(Qwen35.Configuration config) {
        super(
                config.numberOfLayers,
                l -> config.isFullAttention[l],
                l -> config.kvDim(),
                s -> s.keyCache,
                s -> s.valueCache,
                l ->
                        config.isFullAttention[l]
                                ? 0L
                                : (convFloats(config) + ssmFloats(config)) * 4L);
        this.config = config;
        this.convFloats = convFloats(config);
    }

    private static int convFloats(Qwen35.Configuration c) {
        return (c.ssmConvKernel - 1) * c.convChannels();
    }

    private static int ssmFloats(Qwen35.Configuration c) {
        return c.headVDim() * c.headVDim() * c.ssmTimeStepRank;
    }

    @Override
    protected void checkpoint(Qwen35.State state, int to, MemorySegment blob, boolean out) {
        long off = 0;
        for (int l = 0; l < config.numberOfLayers; l++) {
            if (config.isFullAttention[l]) continue;
            off += KvTransfer.transfer(state.ssmConvState[l], 0, blob, off, convFloats, out);
            off += KvTransfer.transfer(state.ssmState[l], blob, off, out);
        }
    }
}
