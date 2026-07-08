package com.qxotic.llm;

import com.qxotic.jinfer.cache.AbstractStateCodec;
import com.qxotic.jinfer.cache.KvTransfer;

import java.lang.foreign.MemorySegment;

/** gpt-oss resume-state codec: alternating sliding-window/full attention, all layers own their KV,
 *  uniform kvDim. Full-attention layers store per-position K/V rows (shared base); sliding-window layers
 *  (W=128) checkpoint a FIXED-SIZE snapshot: the live window rows {@code [max(0,to-W), to)}, read from
 *  their ring slots ({@code pos & (W-1)}), valid only at {@code position == to} — which is exactly why
 *  blocks match completely or not at all. RoPE (YaRN/NeoX) is baked into K at absolute positions, so
 *  restore writes each row back at its true slot. Attention sinks are learned weights, not state; MoE
 *  routing is per-token and stateless. */
public final class GptOssStateCodec extends AbstractStateCodec<GptOss.State> {

    private final GptOss.Configuration config;

    public GptOssStateCodec(GptOss.Configuration config) {
        super(config.numberOfLayers(), l -> !config.isSWA(l), l -> config.kvDim(),
              s -> s.keyCache, s -> s.valueCache,
              l -> config.isSWA(l) ? 2L * config.slidingWindow() * config.kvDim() * 2L : 0L);
        this.config = config;
    }

    @Override
    protected void checkpoint(GptOss.State state, int to, MemorySegment blob, boolean out) {
        int w = config.slidingWindow();
        long kvDim = config.kvDim();
        long off = 0;
        for (int l = 0; l < config.numberOfLayers(); l++) {
            if (!config.isSWA(l)) continue;
            off += KvTransfer.window(state.keyCache[l], to, w, kvDim, 2, blob, off, out);
            off += KvTransfer.window(state.valueCache[l], to, w, kvDim, 2, blob, off, out);
        }
    }
}
