package com.qxotic.llm;

import com.qxotic.jinfer.cache.AbstractStateCodec;
import com.qxotic.jinfer.cache.KvTransfer;
import java.lang.foreign.MemorySegment;

/**
 * Gemma 4 resume-state codec (E2B/A4B shapes): only own-KV layers are serialized — the shared tail
 * layers reuse earlier rings and carry no state. Full-attention own layers store per-position K/V
 * rows (shared base); sliding-window own layers checkpoint a FIXED-SIZE snapshot: the live window
 * rows {@code [max(0,to-W), to)}, read from their ring slots ({@code pos & (W-1)}), valid only at
 * {@code position == to} — which is exactly why blocks match completely or not at all. RoPE is
 * baked into K at absolute positions, so restore writes each row back at its true slot; the ring
 * span wraps at most once, so each direction is at most two contiguous copies (see {@link
 * KvTransfer#window}).
 */
public final class Gemma4StateCodec extends AbstractStateCodec<Gemma4.State> {

    private final Gemma4.Configuration config;

    public Gemma4StateCodec(Gemma4.Configuration config) {
        super(
                config.ownKvLayers(),
                l -> !config.isSWA()[l],
                l -> config.kvDim(l),
                s -> s.keyCache,
                s -> s.valueCache,
                l -> config.isSWA()[l] ? 2L * config.slidingWindow() * config.kvDim(l) * 2L : 0L);
        this.config = config;
    }

    @Override
    protected void checkpoint(Gemma4.State state, int to, MemorySegment blob, boolean out) {
        int w = config.slidingWindow();
        long off = 0;
        for (int l = 0; l < config.ownKvLayers(); l++) {
            if (!config.isSWA()[l]) continue;
            long kvDim = config.kvDim(l);
            off += KvTransfer.window(state.keyCache[l], to, w, kvDim, 2, blob, off, out);
            off += KvTransfer.window(state.valueCache[l], to, w, kvDim, 2, blob, off, out);
        }
    }
}
