package com.qxotic.llm;

import com.qxotic.jinfer.cache.StateCodec;
import com.qxotic.jinfer.cache.KvTransfer;

import java.lang.foreign.MemorySegment;

/** gpt-oss resume-state codec: alternating sliding-window/full attention, all layers own their KV
 *  (no shared tail), uniform kvDim. Full-attention layers serialize per-position K/V rows for the
 *  span at absolute slots. Sliding-window layers (W=128) serialize a FIXED-SIZE checkpoint: the
 *  live window rows {@code [max(0,to-W), to)}, read from their ring slots ({@code pos & (W-1)}),
 *  valid only at {@code position == to}. RoPE (YaRN/NeoX) is baked into K at absolute positions,
 *  so restore writes every row back at its true slot. Attention sinks are per-head learned
 *  weights, not state - nothing to checkpoint; MoE routing is per-token and stateless.
 *
 *  <p>Rows section (every block), layer-major, native F16: full layer l → K rows {@code [from,to)} then V rows;
 *  Checkpoint section (sparse, cache policy): SWA layer l → K window then V window, each padded to a fixed {@code W} rows. The ring span
 *  wraps at most once, so each direction is at most two contiguous segment copies. Restore is a
 *  pure copy; the cache chain-applies blocks in order (the deepest window wins) and then resumes
 *  the state at the chain end. */
public final class GptOssStateCodec implements StateCodec<GptOss.State> {

    private final GptOss.Configuration config;
    private final long bytesPerPosition;          // full-attention K+V rows, native F16
    private final long checkpointBytes;           // all SWA layers' fixed W-row windows, native F16

    public GptOssStateCodec(GptOss.Configuration config) {
        this.config = config;
        long perPos = 0, fixed = 0;
        for (int l = 0; l < config.numberOfLayers(); l++) {
            if (config.isSWA(l)) {
                fixed += 2L * config.slidingWindow() * config.kvDim() * 2L;
            } else {
                perPos += 2L * config.kvDim() * 2L;
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
    public void saveRows(GptOss.State state, int from, int to, MemorySegment dst) {
        rows(state, from, to, dst, true);
    }

    @Override
    public void restoreRows(GptOss.State state, int from, int to, MemorySegment src) {
        rows(state, from, to, src, false);
    }

    @Override
    public void saveCheckpoint(GptOss.State state, int to, MemorySegment dst) {
        checkpoint(state, to, dst, true);
    }

    @Override
    public void restoreCheckpoint(GptOss.State state, int to, MemorySegment src) {
        checkpoint(state, to, src, false);
    }

    /** One walk per section drives both directions so each layout is single-sourced. */
    private void rows(GptOss.State state, int from, int to, MemorySegment blob, boolean out) {
        long off = 0;
        long n = to - from;
        long kvDim = config.kvDim();
        for (int l = 0; l < config.numberOfLayers(); l++) {
            if (config.isSWA(l)) continue;
            off += KvTransfer.transfer(state.keyCache[l], from * kvDim, blob, off, n * kvDim, out);
            off += KvTransfer.transfer(state.valueCache[l], from * kvDim, blob, off, n * kvDim, out);
        }
    }

    private void checkpoint(GptOss.State state, int to, MemorySegment blob, boolean out) {
        int w = config.slidingWindow();
        long kvDim = config.kvDim();
        long off = 0;
        for (int l = 0; l < config.numberOfLayers(); l++) {
            if (!config.isSWA(l)) continue;
            // fixed-size window checkpoint: rows at ring slots, padded to W rows (shared
            // wrap-aware copy - the ring-boundary math lives once, in KvTransfer.window)
            off += KvTransfer.window(state.keyCache[l], to, w, kvDim, 2, blob, off, out);
            off += KvTransfer.window(state.valueCache[l], to, w, kvDim, 2, blob, off, out);
        }
    }
}
