package com.qxotic.jinfer.models.lfm2;

import com.qxotic.jinfer.cache.AbstractStateCodec;
import com.qxotic.jinfer.cache.KvTransfer;
import java.lang.foreign.MemorySegment;

/**
 * LFM2/LFM2-MoE resume-state codec: attention layers store per-position K/V rows (shared base);
 * short-conv (recurrent) layers checkpoint the rolling FIR history — a fixed-size F32 snapshot that
 * only exists at the state's current position, which is exactly why blocks match completely or not
 * at all. MoE routing is per-token and carries no cross-token state — nothing to checkpoint.
 */
public final class Lfm2StateCodec extends AbstractStateCodec<Lfm2.State> {

    private final Lfm2.Configuration config;
    private final int convFloats; // hist * dim, per recurrent layer

    public Lfm2StateCodec(Lfm2.Configuration config) {
        super(
                config.numberOfLayers(),
                l -> !config.isRecurrentLayer(l),
                l -> config.kvDim(l),
                s -> s.keyCache,
                s -> s.valueCache,
                l -> config.isRecurrentLayer(l) ? convFloats(config) * 4L : 0L);
        this.config = config;
        this.convFloats = convFloats(config);
    }

    private static int convFloats(Lfm2.Configuration c) {
        return Math.max(c.shortConvLCache() - 1, 0) * c.embeddingLength();
    }

    @Override
    protected void checkpoint(Lfm2.State state, int to, MemorySegment blob, boolean out) {
        long off = 0;
        for (int l = 0; l < config.numberOfLayers(); l++)
            if (config.isRecurrentLayer(l))
                off += KvTransfer.transfer(state.shortConvState[l], 0, blob, off, convFloats, out);
    }
}
