package com.qxotic.jinfer.models.nemotronh;

import com.qxotic.jinfer.cache.AbstractStateCodec;
import com.qxotic.jinfer.cache.KvTransfer;
import java.lang.foreign.MemorySegment;

/**
 * Nemotron-H resume-state codec: full-attention layers store per-position K/V rows (shared base);
 * Mamba2 layers contribute the conv ring plus the SSM state as the block RESIDUE - MBs per SSM
 * layer (~2MB at Cascade-2-30B dims: dInner 4096 x state 128, F32), so this model overrides {@link
 * #coarseBlocks()}: cached prompts commit as ONE block per prompt (one residue per prompt, not one
 * per turn). MoE routing is per-token and carries no cross-token state.
 */
public final class NemotronHStateCodec extends AbstractStateCodec<NemotronH.State> {

    private final NemotronH.Configuration config;

    public NemotronHStateCodec(NemotronH.Configuration config) {
        super(
                config.numberOfLayers(),
                l -> config.layerTypes()[l] == NemotronH.LayerType.ATTENTION,
                l -> config.kvDim(),
                l -> 0, // full attention: linear position indexing, no ring
                s -> s.keyCache,
                s -> s.valueCache,
                residueBytes(config));
        this.config = config;
    }

    private static int ssmFloats(NemotronH.Configuration c) {
        return c.ssmInnerSize() * c.ssmStateSize();
    }

    private static int convFloats(NemotronH.Configuration c) {
        return Math.max(c.ssmConvKernel() - 1, 0) * c.ssmConvChannels();
    }

    private static long residueBytes(NemotronH.Configuration c) {
        long perLayer = (long) ssmFloats(c) * 4L + (long) convFloats(c) * 4L;
        long layers = 0;
        for (int l = 0; l < c.numberOfLayers(); l++) {
            if (c.layerTypes()[l] == NemotronH.LayerType.SSM) layers++;
        }
        return layers * perLayer;
    }

    @Override
    public boolean coarseBlocks() {
        return true;
    }

    @Override
    protected void residue(NemotronH.State state, int to, MemorySegment blob, boolean out) {
        long off = 0;
        for (int l = 0; l < config.numberOfLayers(); l++) {
            if (config.layerTypes()[l] != NemotronH.LayerType.SSM) continue;
            off += KvTransfer.transfer(state.ssmState[l], blob, off, out);
            off +=
                    KvTransfer.transfer(
                            state.ssmConvState[l], 0, blob, off, convFloats(config), out);
        }
    }
}
