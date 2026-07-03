package com.qxotic.llm;

import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.cache.KvCodec;

import java.lang.foreign.MemorySegment;

/** gpt-oss resume-state codec: alternating sliding-window/full attention, all layers own their KV
 *  (no shared tail), uniform kvDim. Full-attention layers serialize per-position K/V rows for the
 *  span at absolute slots. Sliding-window layers (W=128) serialize a FIXED-SIZE checkpoint: the
 *  live window rows {@code [max(0,to-W), to)}, read from their ring slots ({@code pos & (W-1)}),
 *  valid only at {@code position == to}. RoPE (YaRN/NeoX) is baked into K at absolute positions,
 *  so restore writes every row back at its true slot. Attention sinks are per-head learned
 *  weights, not state - nothing to checkpoint; MoE routing is per-token and stateless.
 *
 *  <p>Blob layout, layer-major, native F16: full layer l → K rows {@code [from,to)} then V rows;
 *  SWA layer l → K window then V window, each padded to a fixed {@code W} rows. The ring span
 *  wraps at most once, so each direction is at most two contiguous segment copies. Restore is a
 *  pure copy; the cache chain-applies blocks in order (the deepest window wins) and then resumes
 *  the state at the chain end. */
public final class GptOssKvCodec implements KvCodec<GptOss.State> {

    private final GptOss.Configuration config;
    private final long bytesPerPosition;          // full-attention K+V rows, native F16
    private final long checkpointBytes;           // all SWA layers' fixed W-row windows, native F16

    public GptOssKvCodec(GptOss.Configuration config) {
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
    public long bytes(int positions) {
        return positions * bytesPerPosition + checkpointBytes;
    }

    @Override
    public void save(GptOss.State state, int from, int to, MemorySegment dst) {
        copy(state, from, to, dst, true);
    }

    @Override
    public void restore(GptOss.State state, int from, int to, MemorySegment src) {
        copy(state, from, to, src, false);
    }

    /** One walk drives both directions so the blob layout is single-sourced. */
    private void copy(GptOss.State state, int from, int to, MemorySegment blob, boolean out) {
        int w = config.slidingWindow();
        long kvDim = config.kvDim();
        long off = 0;
        for (int l = 0; l < config.numberOfLayers(); l++) {
            if (config.isSWA(l)) {
                // fixed-size window checkpoint: rows [lo,to) at ring slots, padded to W rows
                int lo = Math.max(0, to - w);
                int n = to - lo;
                off = ringCopy(state.keyCache[l], lo, n, w, kvDim, blob, off, out);
                off += (long) (w - n) * kvDim * 2;                       // pad K to W rows
                off = ringCopy(state.valueCache[l], lo, n, w, kvDim, blob, off, out);
                off += (long) (w - n) * kvDim * 2;                       // pad V to W rows
            } else {
                long n = to - from;
                off += com.qxotic.jinfer.cache.KvTransfer.transfer(state.keyCache[l], from * kvDim, blob, off, n * kvDim, out);
                off += com.qxotic.jinfer.cache.KvTransfer.transfer(state.valueCache[l], from * kvDim, blob, off, n * kvDim, out);
            }
        }
    }

    /** Window rows [lo, lo+n) live at ring slots {@code pos & (w-1)}; the span wraps at most once,
     *  so it is at most two contiguous runs. */
    private static long ringCopy(FloatTensor ring, int lo, int n, int w,
                                 long kvDim, MemorySegment blob, long off, boolean out) {
        int done = 0;
        while (done < n) {
            int slot = (lo + done) & (w - 1);
            int run = Math.min(n - done, w - slot);                      // stop at the ring edge
            off += com.qxotic.jinfer.cache.KvTransfer.transfer(ring, (long) slot * kvDim, blob, off, (long) run * kvDim, out);
            done += run;
        }
        return off;
    }

    private static long transfer(FloatTensor t, long elemOff,
                                 MemorySegment blob, long byteOff, long elems, boolean out) {
        return out ? t.copyRawTo(elemOff, blob, byteOff, elems)
                   : t.copyRawFrom(blob, byteOff, elemOff, elems);
    }
}
