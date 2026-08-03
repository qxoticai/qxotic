package com.qxotic.jinfer.models.qwen35;

import com.qxotic.jinfer.cache.AbstractStateCodec;
import com.qxotic.jinfer.cache.KvTransfer;
import java.lang.foreign.MemorySegment;

/**
 * Qwen3.5 resume-state codec: the full-attention layers store per-position K/V rows (full context,
 * linear position indexing - no ring); the gated-delta-net layers contribute their conv history
 * plus the S matrices as the block RESIDUE - {@code headVDim^2 x dtRank} F32 per linear layer
 * (~2.1MB each, ~66MB total at 35B-A3B dims: 30 linear layers of 128^2 x 32), so this model
 * overrides {@link #coarseBlocks()}: cached prompts commit as ONE block per prompt, one residue per
 * prompt rather than one per turn. MoE routing is per-token and carries no cross-token state;
 * everything else in the state is per-batch scratch (exactly what {@code State.reset} clears).
 */
public final class Qwen35StateCodec extends AbstractStateCodec<Qwen35.State> {

    private final Qwen35.Configuration config;

    public Qwen35StateCodec(Qwen35.Configuration config) {
        super(
                config.numberOfLayers,
                l -> config.isFullAttention[l],
                l -> config.kvDim(),
                l -> 0, // full attention: linear position indexing, no ring
                s -> s.keyCache,
                s -> s.valueCache,
                residueBytes(config));
        this.config = config;
    }

    private static int sFloats(Qwen35.Configuration c) {
        return c.headVDim() * c.headVDim() * c.ssmTimeStepRank;
    }

    private static int convFloats(Qwen35.Configuration c) {
        return Math.max(c.ssmConvKernel - 1, 0) * c.convChannels();
    }

    private static long residueBytes(Qwen35.Configuration c) {
        long perLayer = (long) sFloats(c) * 4L + (long) convFloats(c) * 4L;
        long linear = 0;
        for (boolean full : c.isFullAttention) {
            if (!full) linear++;
        }
        return linear * perLayer;
    }

    @Override
    public boolean coarseBlocks() {
        return true;
    }

    @Override
    protected void residue(Qwen35.State state, int to, MemorySegment blob, boolean out) {
        long off = 0;
        for (int l = 0; l < config.numberOfLayers; l++) {
            if (config.isFullAttention[l]) continue;
            off += KvTransfer.transfer(state.ssmState[l], blob, off, out);
            off +=
                    KvTransfer.transfer(
                            state.ssmConvState[l], 0, blob, off, convFloats(config), out);
        }
    }
}
