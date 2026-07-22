package com.qxotic.jinfer.models.gemma4;

import com.qxotic.jinfer.cache.AbstractStateCodec;

/**
 * Gemma 4 resume-state codec (E2B/A4B shapes): only own-KV layers are serialized - the shared tail
 * layers reuse earlier rings and carry no state. Full-attention own layers store per-position K/V
 * rows at absolute offsets; sliding-window own layers store per-position rows THROUGH their ring
 * slots ({@code pos & (W-1)}), so the live window rebuilds from restored rows alone and every block
 * is a resume point - no checkpoint, no residue. RoPE is baked into K at absolute positions, so a
 * restored row is valid at its true slot regardless of when it was saved.
 */
public final class Gemma4StateCodec extends AbstractStateCodec<Gemma4.State> {

    public Gemma4StateCodec(Gemma4.Configuration config) {
        super(
                config.ownKvLayers(),
                l -> true,
                l -> config.kvDim(l),
                l -> config.isSWA()[l] ? config.slidingWindow() : 0,
                s -> s.keyCache,
                s -> s.valueCache,
                0L);
    }
}
