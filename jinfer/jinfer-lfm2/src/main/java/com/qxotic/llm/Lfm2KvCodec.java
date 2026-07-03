package com.qxotic.llm;

import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.cache.KvCodec;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/** LFM2/LFM2-MoE resume-state codec: attention layers serialize their per-position K/V rows for
 *  the span; short-conv (recurrent) layers serialize the rolling FIR history — a fixed-size
 *  checkpoint that only exists at the position the state is at, which is exactly why
 *  {@link #save} requires {@code state.position() == to} and why cache blocks match completely or
 *  not at all. MoE routing is per-token and carries no cross-token state — nothing to checkpoint.
 *
 *  <p>Blob layout, layer-major, F32: attention layer l → K rows {@code [from,to)} then V rows;
 *  recurrent layer l → the {@code hist × dim} conv history (as of {@code to}). On restore the
 *  chain is applied in order, so the deepest block's conv history wins — the state resumes at its
 *  {@code to}. */
public final class Lfm2KvCodec implements KvCodec<Lfm2.State> {

    private static final ValueLayout.OfFloat F32 = ValueLayout.JAVA_FLOAT;

    private final Lfm2.Configuration config;
    private final int convFloats;                 // hist * dim, per recurrent layer

    public Lfm2KvCodec(Lfm2.Configuration config) {
        this.config = config;
        this.convFloats = Math.max(config.shortConvLCache() - 1, 0) * config.embeddingLength();
    }

    @Override
    public long bytes(int from, int to) {
        int n = to - from;
        long floats = 0;
        for (int l = 0; l < config.numberOfLayers(); l++) {
            floats += config.isRecurrentLayer(l) ? convFloats : 2L * n * config.kvDim(l);
        }
        return floats * Float.BYTES;
    }

    @Override
    public void save(Lfm2.State state, int from, int to, MemorySegment dst) {
        if (state.position != to) {
            throw new IllegalStateException("conv checkpoint requires position==" + to + ", state is at " + state.position);
        }
        long idx = 0;
        for (int l = 0; l < config.numberOfLayers(); l++) {
            if (config.isRecurrentLayer(l)) {
                idx = copyOut(state.shortConvState[l], 0, convFloats, dst, idx);
            } else {
                long kvDim = config.kvDim(l);
                idx = copyOut(state.keyCache[l], from * kvDim, (to - from) * kvDim, dst, idx);
                idx = copyOut(state.valueCache[l], from * kvDim, (to - from) * kvDim, dst, idx);
            }
        }
    }

    @Override
    public void restore(Lfm2.State state, int from, int to, MemorySegment src) {
        long idx = 0;
        for (int l = 0; l < config.numberOfLayers(); l++) {
            if (config.isRecurrentLayer(l)) {
                idx = copyIn(src, idx, state.shortConvState[l], 0, convFloats);
            } else {
                long kvDim = config.kvDim(l);
                idx = copyIn(src, idx, state.keyCache[l], from * kvDim, (to - from) * kvDim);
                idx = copyIn(src, idx, state.valueCache[l], from * kvDim, (to - from) * kvDim);
            }
        }
        state.position = to;                       // resumable exactly at the block boundary
        state.lastChunkLen = 0;                    // logits/output scratch is NOT restored:
        state.outputCount = 0;                     // the caller ingests before reading logits
    }

    private static long copyOut(FloatTensor src, long srcOff, long len, MemorySegment dst, long dstIdx) {
        for (long i = 0; i < len; i++) dst.setAtIndex(F32, dstIdx + i, src.getFloat(srcOff + i));
        return dstIdx + len;
    }

    private static long copyIn(MemorySegment src, long srcIdx, FloatTensor dst, long dstOff, long len) {
        for (long i = 0; i < len; i++) dst.setFloat(dstOff + i, src.getAtIndex(F32, srcIdx + i));
        return srcIdx + len;
    }
}
