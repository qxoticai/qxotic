package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.RuntimeState;
import java.lang.foreign.MemorySegment;
import java.util.function.Function;
import java.util.function.IntPredicate;
import java.util.function.IntToLongFunction;
import java.util.function.IntUnaryOperator;

/**
 * Shared scaffolding for {@link StateCodec}s. The rows section - per-position K/V for the attention
 * layers - is mechanically identical across models, so it is driven here from a row-layer
 * predicate, the per-layer {@code kvDim}, key/value accessors, and a per-layer RING WIDTH (0 for
 * full-attention layers with linear position indexing; W for sliding-window layers whose rows live
 * at ring slots {@code pos & (W-1)}). A concrete codec supplies that wiring plus, for models with
 * genuinely recurrent state, the small fixed {@link #residue} trailer. Save and restore share one
 * walk (a direction flag), so the two blob layouts can't drift apart.
 */
public abstract class AbstractStateCodec<S extends RuntimeState> implements StateCodec<S> {

    private final int layers;
    private final IntPredicate isRowLayer;
    private final IntToLongFunction kvDim;
    private final IntUnaryOperator ringWidth; // 0 = linear indexing; else slot = pos & (W-1)
    private final Function<S, FloatTensor[]> keys;
    private final Function<S, FloatTensor[]> values;
    private final long bytesPerPosition; // rows: sum over row layers of 2 (K+V) * kvDim * 2 (F16)
    private final long residueBytes;

    protected AbstractStateCodec(
            int layers,
            IntPredicate isRowLayer,
            IntToLongFunction kvDim,
            IntUnaryOperator ringWidth,
            Function<S, FloatTensor[]> keys,
            Function<S, FloatTensor[]> values,
            long residueBytes) {
        this.layers = layers;
        this.isRowLayer = isRowLayer;
        this.kvDim = kvDim;
        this.ringWidth = ringWidth;
        this.keys = keys;
        this.values = values;
        long perPos = 0;
        for (int l = 0; l < layers; l++) {
            if (isRowLayer.test(l)) perPos += 2L * kvDim.applyAsLong(l) * 2L;
        }
        this.bytesPerPosition = perPos;
        this.residueBytes = residueBytes;
    }

    @Override
    public final long blockBytes(int positions) {
        return positions * bytesPerPosition + residueBytes;
    }

    @Override
    public final void save(S s, int from, int to, MemorySegment dst) {
        walk(s, from, to, dst, true);
    }

    @Override
    public final void restore(S s, int from, int to, MemorySegment src) {
        walk(s, from, to, src, false);
    }

    /**
     * The shared block walk (one direction per {@code out}): row layer l → K rows {@code [from,to)}
     * then V rows (through ring slots where the layer is windowed), native F16, layer-major; then
     * the residue trailer.
     */
    private void walk(S s, int from, int to, MemorySegment blob, boolean out) {
        long off = 0, n = to - from;
        FloatTensor[] k = keys.apply(s), v = values.apply(s);
        for (int l = 0; l < layers; l++) {
            if (!isRowLayer.test(l)) continue;
            long kd = kvDim.applyAsLong(l);
            int w = ringWidth.applyAsInt(l);
            if (w == 0) {
                off += KvTransfer.transfer(k[l], from * kd, blob, off, n * kd, out);
                off += KvTransfer.transfer(v[l], from * kd, blob, off, n * kd, out);
            } else {
                off += KvTransfer.ringSpan(k[l], from, to, w, kd, blob, off, out);
                off += KvTransfer.ringSpan(v[l], from, to, w, kd, blob, off, out);
            }
        }
        residue(s, to, blob.asSlice(off), out);
    }

    /**
     * Serialize ({@code out=true}) or restore the model's small recurrent residue as of position
     * {@code to}. Default no-op - models whose state is entirely rows never reach it.
     */
    protected void residue(S s, int to, MemorySegment blob, boolean out) {}
}
