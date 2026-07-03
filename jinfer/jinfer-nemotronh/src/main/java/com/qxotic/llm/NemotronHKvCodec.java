package com.qxotic.llm;

import com.qxotic.jinfer.cache.KvCodec;
import com.qxotic.jinfer.cache.KvTransfer;

import java.lang.foreign.MemorySegment;

/** Nemotron-H resume-state codec: ATTENTION layers serialize their per-position K/V rows for the
 *  span (raw F16, linear absolute positions); SSM (Mamba2) layers serialize their fixed-size F32
 *  checkpoint - the conv ring plus the recurrent S matrix - which only exists at the position the
 *  state is at, exactly why blocks match completely or not at all. MOE layers route per token and
 *  carry no cross-token state - nothing to checkpoint.
 *
 *  <p>Blob layout, layer-major: ATTENTION layer l -> K rows {@code [from,to)} then V rows (native
 *  F16); SSM layer l -> conv ring ({@code (dConv-1) x convChannels} F32) then the S matrix
 *  ({@code dInner x dState} F32, a heap array copied via {@code MemorySegment.ofArray}). Restore
 *  is a pure copy; the cache chain-applies blocks in order (deepest SSM checkpoint wins) and then
 *  resumes the state at the chain end.
 *
 *  <p>Size note: the fixed checkpoint dominates - 23 SSM layers x (2 MiB S + 72 KiB ring) is
 *  ~47.6 MiB per block regardless of span, vs ~6 KiB per position of attention K/V. Turn-aligned
 *  blocks amortize it well; single-token decode blocks are expensive for this model, so harnesses
 *  keep decode budgets small and cache budgets large (compression of unchanged checkpoints is a
 *  noted follow-up). */
public final class NemotronHKvCodec implements KvCodec<NemotronH.State> {

    private final NemotronH.Configuration config;
    private final int convFloats;                 // (dConv-1) * convChannels, per SSM layer
    private final int ssmFloats;                  // dInner * dState, per SSM layer
    private final long bytesPerPosition;          // attention K+V rows, native F16
    private final long checkpointBytes;           // all SSM layers' conv + S, F32

    public NemotronHKvCodec(NemotronH.Configuration config) {
        this.config = config;
        this.convFloats = (config.ssmConvKernel() - 1) * config.ssmConvChannels();
        this.ssmFloats = config.ssmInnerSize() * config.ssmStateSize();
        long perPos = 0, fixed = 0;
        for (NemotronH.LayerType type : config.layerTypes()) {
            switch (type) {
                case ATTENTION -> perPos += 2L * config.kvDim() * 2L;        // K+V rows, F16
                case SSM -> fixed += (convFloats + ssmFloats) * 4L;          // F32 checkpoint
                case MOE -> { }
            }
        }
        this.bytesPerPosition = perPos;
        this.checkpointBytes = fixed;
    }

    @Override
    public long bytes(int positions) {
        return positions * bytesPerPosition + checkpointBytes;
    }

    @Override
    public void save(NemotronH.State state, int from, int to, MemorySegment dst) {
        copy(state, from, to, dst, true);
    }

    @Override
    public void restore(NemotronH.State state, int from, int to, MemorySegment src) {
        copy(state, from, to, src, false);
    }

    /** One walk drives both directions so the blob layout is single-sourced. */
    private void copy(NemotronH.State state, int from, int to, MemorySegment blob, boolean out) {
        long off = 0;
        int n = to - from;
        long kvDim = config.kvDim();
        for (int l = 0; l < config.numberOfLayers(); l++) {
            switch (config.layerTypes()[l]) {
                case ATTENTION -> {
                    off += KvTransfer.transfer(state.keyCache[l], from * kvDim, blob, off, n * kvDim, out);
                    off += KvTransfer.transfer(state.valueCache[l], from * kvDim, blob, off, n * kvDim, out);
                }
                case SSM -> {
                    off += KvTransfer.transfer(state.ssmConvState[l], 0, blob, off, convFloats, out);
                    off += KvTransfer.transfer(state.ssmState[l], blob, off, out);
                }
                case MOE -> { }
            }
        }
    }
}
