package com.qxotic.jinfer.models.gptoss;

import com.qxotic.jinfer.cache.AbstractStateCodec;

/**
 * gpt-oss resume-state codec: alternating sliding-window/full attention, all layers own their KV,
 * uniform kvDim. Full-attention layers store per-position K/V rows at absolute offsets;
 * sliding-window layers (W=128) store per-position rows THROUGH their ring slots ({@code pos &
 * (W-1)}), so the live window rebuilds from restored rows alone and every block is a resume point -
 * no checkpoint, no residue. Spans longer than W alias ring slots; see {@code KvTransfer.ringSpan}
 * for why that is safe in both directions.
 */
public final class GptOssStateCodec extends AbstractStateCodec<GptOss.State> {

    public GptOssStateCodec(GptOss.Configuration config) {
        super(
                config.numberOfLayers(),
                l -> true,
                l -> config.kvDim(),
                l -> config.isSWA(l) ? config.slidingWindow() : 0,
                s -> s.keyCache,
                s -> s.valueCache,
                0L);
    }
}
