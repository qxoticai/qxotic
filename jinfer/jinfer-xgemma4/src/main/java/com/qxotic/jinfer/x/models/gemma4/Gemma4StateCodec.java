package com.qxotic.jinfer.x.models.gemma4;

import com.qxotic.jinfer.x.boundary.StateCodec;
import com.qxotic.jinfer.x.kernels.KvTransfer;
import java.lang.foreign.MemorySegment;

/**
 * Gemma 4 resume-state codec (E2B/A4B shapes): only own-KV layers are serialized - the shared tail
 * layers reuse earlier rings and carry no state. Full-attention own layers store per-position K/V
 * rows at absolute offsets; sliding-window own layers store per-position rows THROUGH their ring
 * slots ({@code pos & (W-1)}), so the live window rebuilds from restored rows alone and every block
 * is a resume point - no checkpoint, no residue. RoPE is baked into K at absolute positions, so a
 * restored row is valid at its true slot regardless of when it was saved.
 */
final class Gemma4StateCodec implements StateCodec<Gemma4.State> {

    private final Gemma4.Configuration config;
    private final long bytesPerPosition;

    Gemma4StateCodec(Gemma4.Configuration config) {
        this.config = config;
        long rowBytes = 0;
        for (int layer = 0; layer < config.ownKvLayers(); layer++) {
            rowBytes =
                    Math.addExact(
                            rowBytes, Math.multiplyExact(2L * Short.BYTES, config.kvDim(layer)));
        }
        bytesPerPosition = rowBytes;
    }

    @Override
    public long checkpointBytes(int positions) {
        if (positions < 0) throw new IllegalArgumentException("positions " + positions);
        return Math.multiplyExact((long) positions, bytesPerPosition);
    }

    @Override
    public void saveCheckpoint(Gemma4.State state, int from, int to, MemorySegment destination) {
        transfer(state, from, to, destination, true);
    }

    @Override
    public void restoreCheckpoint(Gemma4.State state, int from, int to, MemorySegment source) {
        transfer(state, from, to, source, false);
    }

    private void transfer(Gemma4.State state, int from, int to, MemorySegment blob, boolean save) {
        StateCodec.requireCheckpoint(this, state, from, to, blob, save);
        long offset = 0;
        for (int layer = 0; layer < config.ownKvLayers(); layer++) {
            int kvDim = config.kvDim(layer);
            if (config.isSwa()[layer]) {
                offset +=
                        KvTransfer.ringSpan(
                                state.keyCache[layer],
                                from,
                                to,
                                config.slidingWindow(),
                                kvDim,
                                blob,
                                offset,
                                save);
                offset +=
                        KvTransfer.ringSpan(
                                state.valueCache[layer],
                                from,
                                to,
                                config.slidingWindow(),
                                kvDim,
                                blob,
                                offset,
                                save);
            } else {
                long elements = Math.multiplyExact((long) (to - from), kvDim);
                long elementOffset = Math.multiplyExact((long) from, kvDim);
                offset +=
                        KvTransfer.transfer(
                                state.keyCache[layer], elementOffset, blob, offset, elements, save);
                offset +=
                        KvTransfer.transfer(
                                state.valueCache[layer],
                                elementOffset,
                                blob,
                                offset,
                                elements,
                                save);
            }
        }
    }
}
