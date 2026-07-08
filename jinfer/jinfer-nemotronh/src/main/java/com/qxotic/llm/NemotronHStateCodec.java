package com.qxotic.llm;

import com.qxotic.jinfer.cache.AbstractStateCodec;
import com.qxotic.jinfer.cache.KvTransfer;

import java.lang.foreign.MemorySegment;

/** Nemotron-H resume-state codec: ATTENTION layers store per-position K/V rows (shared base); SSM
 *  (Mamba2) layers checkpoint a fixed-size F32 snapshot — the conv ring plus the recurrent S matrix —
 *  which only exists at the state's current position, exactly why blocks match completely or not at all.
 *  MOE layers route per token and carry no cross-token state. The S matrix is a heap array copied via a
 *  heap segment view.
 *
 *  <p>Size note: the fixed checkpoint dominates — 23 SSM layers × (2 MiB S + 72 KiB ring) is ~47.6 MiB
 *  per block regardless of span, vs ~6 KiB/position of attention K/V. Turn-aligned blocks amortize it;
 *  single-token decode blocks are expensive, so harnesses keep decode budgets small and cache budgets
 *  large (checkpoint compression is a noted follow-up). */
public final class NemotronHStateCodec extends AbstractStateCodec<NemotronH.State> {

    private final NemotronH.Configuration config;
    private final int convFloats;                 // (dConv-1) * convChannels, per SSM layer

    public NemotronHStateCodec(NemotronH.Configuration config) {
        super(config.numberOfLayers(), l -> config.layerTypes()[l] == NemotronH.LayerType.ATTENTION,
              l -> config.kvDim(), s -> s.keyCache, s -> s.valueCache,
              l -> config.layerTypes()[l] == NemotronH.LayerType.SSM ? (convFloats(config) + ssmFloats(config)) * 4L : 0L);
        this.config = config;
        this.convFloats = convFloats(config);
    }

    private static int convFloats(NemotronH.Configuration c) { return (c.ssmConvKernel() - 1) * c.ssmConvChannels(); }
    private static int ssmFloats(NemotronH.Configuration c) { return c.ssmInnerSize() * c.ssmStateSize(); }

    @Override
    protected void checkpoint(NemotronH.State state, int to, MemorySegment blob, boolean out) {
        long off = 0;
        for (int l = 0; l < config.numberOfLayers(); l++) {
            if (config.layerTypes()[l] != NemotronH.LayerType.SSM) continue;
            off += KvTransfer.transfer(state.ssmConvState[l], 0, blob, off, convFloats, out);
            off += KvTransfer.transfer(state.ssmState[l], blob, off, out);
        }
    }
}
