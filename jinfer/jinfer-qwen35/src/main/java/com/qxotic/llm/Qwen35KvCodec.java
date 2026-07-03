package com.qxotic.llm;

import com.qxotic.jinfer.cache.KvCodec;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/** Qwen3.5 (dense and MoE) resume-state codec: full-attention layers serialize their per-position
 *  K/V rows for the span (raw F16, bulk segment copies); gated-delta-net (SSM) layers serialize a
 *  fixed-size F32 checkpoint - the conv ring plus the delta-net S matrix - that only exists at the
 *  position the state is at, which is exactly why blocks match completely or not at all. MoE
 *  routing is per-row and carries no cross-token state - nothing to checkpoint.
 *
 *  <p>Blob layout, layer-major: attention layer l → K rows {@code [from,to)} then V rows (native
 *  F16); SSM layer l → the {@code (convKernel-1) × convChannels} conv ring (F32) then the
 *  {@code headVDim × headVDim × dtRank} delta-net state (F32, a heap array - copied via a heap
 *  segment view). Restore is a pure copy; the cache chain-applies blocks in order (deepest
 *  checkpoint wins) and then resumes the state at the chain end. */
public final class Qwen35KvCodec implements KvCodec<Qwen35.State> {

    private final Qwen35.Configuration config;
    private final int convFloats;                 // (convKernel-1) * convChannels, per SSM layer
    private final int ssmFloats;                  // headVDim * headVDim * dtRank, per SSM layer
    private final long bytesPerPosition;          // attention K+V, native F16
    private final long checkpointBytes;           // all SSM layers' conv ring + delta-net state, F32

    public Qwen35KvCodec(Qwen35.Configuration config) {
        this.config = config;
        this.convFloats = (config.ssmConvKernel - 1) * config.convChannels();
        this.ssmFloats = config.headVDim() * config.headVDim() * config.ssmTimeStepRank;
        long perPos = 0, fixed = 0;
        for (int l = 0; l < config.numberOfLayers; l++) {
            if (config.isFullAttention[l]) {
                perPos += 2L * config.kvDim() * 2L;                // K+V rows, F16
            } else {
                fixed += (convFloats + ssmFloats) * 4L;            // F32
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
    public void save(Qwen35.State state, int from, int to, MemorySegment dst) {
        long off = 0;
        int n = to - from;
        long kvDim = config.kvDim();
        for (int l = 0; l < config.numberOfLayers; l++) {
            if (config.isFullAttention[l]) {
                off += state.keyCache[l].copyRawTo(from * kvDim, dst, off, n * kvDim);
                off += state.valueCache[l].copyRawTo(from * kvDim, dst, off, n * kvDim);
            } else {
                off += state.ssmConvState[l].copyRawTo(0, dst, off, convFloats);
                MemorySegment.copy(MemorySegment.ofArray(state.ssmState[l]), ValueLayout.JAVA_FLOAT, 0,
                        dst, ValueLayout.JAVA_FLOAT_UNALIGNED, off, ssmFloats);
                off += ssmFloats * 4L;
            }
        }
    }

    @Override
    public void restore(Qwen35.State state, int from, int to, MemorySegment src) {
        long off = 0;
        int n = to - from;
        long kvDim = config.kvDim();
        for (int l = 0; l < config.numberOfLayers; l++) {
            if (config.isFullAttention[l]) {
                off += state.keyCache[l].copyRawFrom(src, off, from * kvDim, n * kvDim);
                off += state.valueCache[l].copyRawFrom(src, off, from * kvDim, n * kvDim);
            } else {
                off += state.ssmConvState[l].copyRawFrom(src, off, 0, convFloats);
                MemorySegment.copy(src, ValueLayout.JAVA_FLOAT_UNALIGNED, off,
                        MemorySegment.ofArray(state.ssmState[l]), ValueLayout.JAVA_FLOAT, 0, ssmFloats);
                off += ssmFloats * 4L;
            }
        }
    }
}
