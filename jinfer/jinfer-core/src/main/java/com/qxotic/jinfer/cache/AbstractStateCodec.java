package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.RuntimeState;

import java.lang.foreign.MemorySegment;
import java.util.function.Function;
import java.util.function.IntPredicate;
import java.util.function.IntToLongFunction;

/** Shared scaffolding for {@link StateCodec}s. The rows section — per-position K/V for the attention
 *  layers — is mechanically identical across models, so it is driven here from a row-layer predicate,
 *  the per-layer {@code kvDim}, and key/value accessors; the block sizes are accumulated once from those.
 *  A concrete codec supplies that wiring plus, for hybrid models, the genuinely model-specific
 *  {@link #checkpoint} (recurrent/windowed state). Pure-attention models leave the checkpoint at its
 *  no-op default ({@code checkpointBytes == 0}, every block a resume point). Save and restore share one
 *  walk (a direction flag), so the two blob layouts can't drift apart. */
public abstract class AbstractStateCodec<S extends RuntimeState> implements StateCodec<S> {

    private final int layers;
    private final IntPredicate isRowLayer;
    private final IntToLongFunction kvDim;
    private final Function<S, FloatTensor[]> keys;
    private final Function<S, FloatTensor[]> values;
    private final long bytesPerPosition;   // rows: sum over row layers of 2 (K+V) * kvDim * 2 (F16)
    private final long checkpointBytes;

    /** @param checkpointLayerBytes checkpoint bytes contributed by layer {@code l} (0 for a row layer). */
    protected AbstractStateCodec(int layers, IntPredicate isRowLayer, IntToLongFunction kvDim,
                                 Function<S, FloatTensor[]> keys, Function<S, FloatTensor[]> values,
                                 IntToLongFunction checkpointLayerBytes) {
        this.layers = layers;
        this.isRowLayer = isRowLayer;
        this.kvDim = kvDim;
        this.keys = keys;
        this.values = values;
        long perPos = 0, checkpoint = 0;
        for (int l = 0; l < layers; l++) {
            if (isRowLayer.test(l)) perPos += 2L * kvDim.applyAsLong(l) * 2L;
            checkpoint += checkpointLayerBytes.applyAsLong(l);
        }
        this.bytesPerPosition = perPos;
        this.checkpointBytes = checkpoint;
    }

    @Override public final long rowBytes(int positions) { return positions * bytesPerPosition; }
    @Override public final long checkpointBytes() { return checkpointBytes; }

    @Override public final void saveRows(S s, int from, int to, MemorySegment dst) { rows(s, from, to, dst, true); }
    @Override public final void restoreRows(S s, int from, int to, MemorySegment src) { rows(s, from, to, src, false); }
    @Override public final void saveCheckpoint(S s, int to, MemorySegment dst) { checkpoint(s, to, dst, true); }
    @Override public final void restoreCheckpoint(S s, int to, MemorySegment src) { checkpoint(s, to, src, false); }

    /** The shared rows walk (one direction per {@code out}): row layer l → K rows {@code [from,to)} then
     *  V rows, native F16, layer-major. */
    private void rows(S s, int from, int to, MemorySegment blob, boolean out) {
        long off = 0, n = to - from;
        FloatTensor[] k = keys.apply(s), v = values.apply(s);
        for (int l = 0; l < layers; l++) {
            if (!isRowLayer.test(l)) continue;
            long kd = kvDim.applyAsLong(l);
            off += KvTransfer.transfer(k[l], from * kd, blob, off, n * kd, out);
            off += KvTransfer.transfer(v[l], from * kd, blob, off, n * kd, out);
        }
    }

    /** Serialize ({@code out=true}) or restore the model's recurrent/windowed checkpoint as of position
     *  {@code to}. Default no-op — pure-attention models ({@code checkpointBytes == 0}) never reach it. */
    protected void checkpoint(S s, int to, MemorySegment blob, boolean out) {}
}
