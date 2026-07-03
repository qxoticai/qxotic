package com.qxotic.llm;

import com.qxotic.jinfer.cache.KvCodec;

import java.lang.foreign.MemorySegment;

/** LFM2/LFM2-MoE resume-state codec: attention layers serialize their per-position K/V rows for
 *  the span (raw F16, bulk segment copies); short-conv (recurrent) layers serialize the rolling
 *  FIR history — a fixed-size F32 checkpoint that only exists at the position the state is at,
 *  which is exactly why blocks match completely or not at all. MoE routing is per-token and
 *  carries no cross-token state — nothing to checkpoint.
 *
 *  <p>Blob layout, layer-major: attention layer l → K rows {@code [from,to)} then V rows (native
 *  F16); recurrent layer l → the {@code hist × dim} conv history (F32, as of {@code to}). Restore
 *  is a pure copy; the cache chain-applies blocks in order (deepest conv checkpoint wins) and then
 *  resumes the state at the chain end. */
public final class Lfm2KvCodec implements KvCodec<Lfm2.State> {

    private final Lfm2.Configuration config;
    private final int convFloats;                 // hist * dim, per recurrent layer
    private final long bytesPerPosition;          // attention K+V, native F16
    private final long checkpointBytes;           // all recurrent layers' conv history, F32

    public Lfm2KvCodec(Lfm2.Configuration config) {
        this.config = config;
        this.convFloats = Math.max(config.shortConvLCache() - 1, 0) * config.embeddingLength();
        long perPos = 0, fixed = 0;
        for (int l = 0; l < config.numberOfLayers(); l++) {
            if (config.isRecurrentLayer(l)) {
                fixed += convFloats * 4L;                          // F32
            } else {
                perPos += 2L * config.kvDim(l) * 2L;               // K+V rows, F16
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
    public void save(Lfm2.State state, int from, int to, MemorySegment dst) {
        long off = 0;
        int n = to - from;
        for (int l = 0; l < config.numberOfLayers(); l++) {
            if (config.isRecurrentLayer(l)) {
                off += state.shortConvState[l].copyRawTo(0, dst, off, convFloats);
            } else {
                long kvDim = config.kvDim(l);
                off += state.keyCache[l].copyRawTo(from * kvDim, dst, off, n * kvDim);
                off += state.valueCache[l].copyRawTo(from * kvDim, dst, off, n * kvDim);
            }
        }
    }

    @Override
    public void restore(Lfm2.State state, int from, int to, MemorySegment src) {
        long off = 0;
        int n = to - from;
        for (int l = 0; l < config.numberOfLayers(); l++) {
            if (config.isRecurrentLayer(l)) {
                off += state.shortConvState[l].copyRawFrom(src, off, 0, convFloats);
            } else {
                long kvDim = config.kvDim(l);
                off += state.keyCache[l].copyRawFrom(src, off, from * kvDim, n * kvDim);
                off += state.valueCache[l].copyRawFrom(src, off, from * kvDim, n * kvDim);
            }
        }
    }
}
