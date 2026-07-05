package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.RuntimeState;

import java.lang.foreign.MemorySegment;
import java.util.function.Function;

/** The degenerate resume-state codec for uniform full-attention models: every layer stores
 *  absolute-position KV rows, so a block is just the span's K/V rows per layer (raw F16,
 *  layer-major: layer l → K rows {@code [from,to)} then V rows) and there is no fixed
 *  checkpoint ({@code checkpointBytes()==0}: every block is a resume point). Dense models
 *  (Llama, Granite) plug in with their two cache accessors; hybrid
 *  models (windows, recurrent checkpoints) write their own codec — their per-layer shapes
 *  genuinely differ and are clearer hand-written. */
public final class DenseKvCodec<S extends RuntimeState> implements KvCodec<S> {

    private final int layers;
    private final long kvDim;
    private final long bytesPerPosition;          // K+V rows across all layers, native F16
    private final Function<S, FloatTensor[]> keys;
    private final Function<S, FloatTensor[]> values;

    public DenseKvCodec(int layers, long kvDim, Function<S, FloatTensor[]> keys, Function<S, FloatTensor[]> values) {
        this.layers = layers;
        this.kvDim = kvDim;
        this.bytesPerPosition = layers * 2L * kvDim * 2L;
        this.keys = keys;
        this.values = values;
    }

    @Override
    public long rowBytes(int positions) {
        return positions * bytesPerPosition;
    }

    @Override
    public void saveRows(S state, int from, int to, MemorySegment dst) {
        long off = 0;
        long n = to - from;
        FloatTensor[] k = keys.apply(state), v = values.apply(state);
        for (int l = 0; l < layers; l++) {
            off += k[l].copyRawTo(from * kvDim, dst, off, n * kvDim);
            off += v[l].copyRawTo(from * kvDim, dst, off, n * kvDim);
        }
    }

    @Override
    public void restoreRows(S state, int from, int to, MemorySegment src) {
        long off = 0;
        long n = to - from;
        FloatTensor[] k = keys.apply(state), v = values.apply(state);
        for (int l = 0; l < layers; l++) {
            off += k[l].copyRawFrom(src, off, from * kvDim, n * kvDim);
            off += v[l].copyRawFrom(src, off, from * kvDim, n * kvDim);
        }
    }
}
