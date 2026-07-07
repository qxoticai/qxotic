package com.qxotic.llm;

import com.qxotic.jinfer.cache.StateCodec;
import com.qxotic.jinfer.cache.KvTransfer;

import java.lang.foreign.MemorySegment;

/** LFM2/LFM2-MoE resume-state codec: attention layers serialize their per-position K/V rows for
 *  the span (raw F16, bulk segment copies); short-conv (recurrent) layers serialize the rolling
 *  FIR history — a fixed-size F32 checkpoint that only exists at the position the state is at,
 *  which is exactly why blocks match completely or not at all. MoE routing is per-token and
 *  carries no cross-token state — nothing to checkpoint.
 *
 *  <p>Rows section, layer-major: attention layer l → K rows {@code [from,to)} then V rows (native
 *  F16). Checkpoint section: recurrent layer l → the {@code hist × dim} conv history (F32, as of
 *  the block end). Checkpoints are sparse (cache policy); a resume restores rows along the chain
 *  and the deepest checkpoint. */
public final class Lfm2StateCodec implements StateCodec<Lfm2.State> {

    private final Lfm2.Configuration config;
    private final int convFloats;                 // hist * dim, per recurrent layer
    private final long bytesPerPosition;          // attention K+V, native F16
    private final long checkpointBytes;           // all recurrent layers' conv history, F32

    public Lfm2StateCodec(Lfm2.Configuration config) {
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
    public long rowBytes(int positions) {
        return positions * bytesPerPosition;
    }

    @Override
    public long checkpointBytes() {
        return checkpointBytes;
    }

    @Override
    public void saveRows(Lfm2.State state, int from, int to, MemorySegment dst) {
        rows(state, from, to, dst, true);
    }

    @Override
    public void restoreRows(Lfm2.State state, int from, int to, MemorySegment src) {
        rows(state, from, to, src, false);
    }

    @Override
    public void saveCheckpoint(Lfm2.State state, int to, MemorySegment dst) {
        checkpoint(state, dst, true);
    }

    @Override
    public void restoreCheckpoint(Lfm2.State state, int to, MemorySegment src) {
        checkpoint(state, src, false);
    }

    /** One walk per section drives both directions so each layout is single-sourced. */
    private void rows(Lfm2.State state, int from, int to, MemorySegment blob, boolean out) {
        long off = 0;
        int n = to - from;
        for (int l = 0; l < config.numberOfLayers(); l++) {
            if (config.isRecurrentLayer(l)) continue;
            long kvDim = config.kvDim(l);
            off += KvTransfer.transfer(state.keyCache[l], from * kvDim, blob, off, n * kvDim, out);
            off += KvTransfer.transfer(state.valueCache[l], from * kvDim, blob, off, n * kvDim, out);
        }
    }

    private void checkpoint(Lfm2.State state, MemorySegment blob, boolean out) {
        long off = 0;
        for (int l = 0; l < config.numberOfLayers(); l++) {
            if (!config.isRecurrentLayer(l)) continue;
            off += KvTransfer.transfer(state.shortConvState[l], 0, blob, off, convFloats, out);
        }
    }
}
