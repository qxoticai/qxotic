package com.qxotic.llm;

import com.qxotic.jinfer.cache.KvCodec;

import java.lang.foreign.MemorySegment;

/** Gemma 4 resume-state codec (E2B/A4B shapes): only the own-KV layers are serialized — the shared
 *  tail layers reuse layer {@code ownKvLayers-2}/{@code -1} rings and carry no state of their own.
 *  Full-attention own layers serialize per-position K/V rows for the span at absolute slots.
 *  Sliding-window own layers serialize a FIXED-SIZE checkpoint: the live window rows
 *  {@code [max(0,to-W), to)}, read from their ring slots ({@code pos & (W-1)}), valid only at
 *  {@code position == to} — which is exactly why blocks match completely or not at all. RoPE is
 *  baked into K at absolute positions, so restore writes every row back at its true slot.
 *
 *  <p>Blob layout, own-layer-major, native F16: full layer l → K rows {@code [from,to)} then V
 *  rows; SWA layer l → K window then V window, each padded to a fixed {@code W} rows (blob row r
 *  = position {@code max(0,to-W)+r}; the ring span wraps at most once, so each direction is at
 *  most two contiguous segment copies). Restore is a pure copy; the cache chain-applies blocks in
 *  order — later blocks overwrite ring slots with newer positions, so the deepest window wins —
 *  and then resumes the state at the chain end. */
public final class Gemma4KvCodec implements KvCodec<Gemma4.State> {

    private final Gemma4.Configuration config;
    private final long bytesPerPosition;          // full-attention K+V rows, native F16
    private final long checkpointBytes;           // all SWA layers' fixed W-row windows, native F16

    public Gemma4KvCodec(Gemma4.Configuration config) {
        this.config = config;
        long perPos = 0, fixed = 0;
        for (int l = 0; l < config.ownKvLayers(); l++) {
            if (config.isSWA()[l]) {
                fixed += 2L * config.slidingWindow() * config.kvDim(l) * 2L;
            } else {
                perPos += 2L * config.kvDim(l) * 2L;
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
    public void save(Gemma4.State state, int from, int to, MemorySegment dst) {
        copy(state, from, to, dst, true);
    }

    @Override
    public void restore(Gemma4.State state, int from, int to, MemorySegment src) {
        copy(state, from, to, src, false);
    }

    /** One walk drives both directions so the blob layout is single-sourced. */
    private void copy(Gemma4.State state, int from, int to, MemorySegment blob, boolean out) {
        int w = config.slidingWindow();
        long off = 0;
        for (int l = 0; l < config.ownKvLayers(); l++) {
            long kvDim = config.kvDim(l);
            if (config.isSWA()[l]) {
                // fixed-size window checkpoint: rows at ring slots, padded to W rows (shared
                // wrap-aware copy - the ring-boundary math lives once, in KvTransfer.window)
                off += com.qxotic.jinfer.cache.KvTransfer.window(state.keyCache[l], to, w, kvDim, 2, blob, off, out);
                off += com.qxotic.jinfer.cache.KvTransfer.window(state.valueCache[l], to, w, kvDim, 2, blob, off, out);
            } else {
                long n = to - from;
                off += com.qxotic.jinfer.cache.KvTransfer.transfer(state.keyCache[l], from * kvDim, blob, off, n * kvDim, out);
                off += com.qxotic.jinfer.cache.KvTransfer.transfer(state.valueCache[l], from * kvDim, blob, off, n * kvDim, out);
            }
        }
    }


}
