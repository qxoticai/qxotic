package com.qxotic.llm;

import com.qxotic.jinfer.cache.KvCodec;

import java.lang.foreign.MemorySegment;

/** Granite resume-state codec: the degenerate case, same as {@link LlamaKvCodec} — every layer is
 *  full attention with absolute-position KV rows, so a block is just the span's K/V rows per
 *  layer, raw F16, and there is no fixed checkpoint. Blob layout, layer-major: layer l → K rows
 *  {@code [from,to)} then V rows. */
public final class GraniteKvCodec implements KvCodec<Granite.State> {

    private final int numberOfLayers;
    private final long kvDim;
    private final long bytesPerPosition;          // K+V rows across all layers, native F16

    public GraniteKvCodec(Granite.Configuration config) {
        this.numberOfLayers = config.numberOfLayers();
        this.kvDim = config.kvDim();
        this.bytesPerPosition = numberOfLayers * 2L * kvDim * 2L;
    }

    @Override
    public long bytes(int positions) {
        return positions * bytesPerPosition;
    }

    @Override
    public void save(Granite.State state, int from, int to, MemorySegment dst) {
        long off = 0;
        long n = to - from;
        for (int l = 0; l < numberOfLayers; l++) {
            off += state.keyCache[l].copyRawTo(from * kvDim, dst, off, n * kvDim);
            off += state.valueCache[l].copyRawTo(from * kvDim, dst, off, n * kvDim);
        }
    }

    @Override
    public void restore(Granite.State state, int from, int to, MemorySegment src) {
        long off = 0;
        long n = to - from;
        for (int l = 0; l < numberOfLayers; l++) {
            off += state.keyCache[l].copyRawFrom(src, off, from * kvDim, n * kvDim);
            off += state.valueCache[l].copyRawFrom(src, off, from * kvDim, n * kvDim);
        }
    }
}
