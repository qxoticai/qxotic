package com.qxotic.jinfer.kernels;

import static com.qxotic.jinfer.Segments.F_SPECIES;
import static com.qxotic.jinfer.Segments.USE_VECTOR_API;
import static com.qxotic.jinfer.Segments.readFloat;
import static com.qxotic.jinfer.Segments.writeFloat;

import com.qxotic.jinfer.Parallel;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

/**
 * Normalization kernels over views. All operands are dense FP32 (checked at entry via {@link
 * Raw#f32}); {@code USE_VECTOR_API=false} selects the scalar path with the same math.
 */
public final class Norms {
    private Norms() {}

    /**
     * RMS normalization: {@code out = weight * x / sqrt(mean(x^2) + eps)} over {@code size}
     * contiguous lanes.
     */
    public static void rmsnorm(
            MemoryView<MemorySegment> out,
            long outOffset,
            MemoryView<MemorySegment> x,
            long xOffset,
            MemoryView<MemorySegment> weight,
            int size,
            float rmsNormEps) {
        Raw o = Raw.f32(out, "out");
        Raw xv = Raw.f32(x, "x");
        Raw w = Raw.f32(weight, "weight");
        if (USE_VECTOR_API) {
            var species = F_SPECIES;
            int upperBound = species.loopBound(size);
            FloatVector acc = FloatVector.zero(species);
            int i = 0;
            for (; i < upperBound; i += species.length()) {
                var xvv =
                        FloatVector.fromMemorySegment(
                                species,
                                xv.vseg(),
                                xv.vbase() + (xOffset + i) * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
                acc = xvv.fma(xvv, acc);
            }
            float ss = acc.reduceLanes(VectorOperators.ADD);
            for (; i < size; i++) {
                float xi = readFloat(xv.vseg(), xv.vbase() + (xOffset + i) * Float.BYTES);
                ss += xi * xi;
            }
            ss /= size;
            ss += rmsNormEps;
            ss = (float) (1.0 / Math.sqrt(ss));
            FloatVector scale = FloatVector.broadcast(species, ss);
            for (i = 0; i < upperBound; i += species.length()) {
                var xvv =
                        FloatVector.fromMemorySegment(
                                species,
                                xv.vseg(),
                                xv.vbase() + (xOffset + i) * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
                var wv =
                        FloatVector.fromMemorySegment(
                                species,
                                w.vseg(),
                                w.vbase() + (long) i * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
                wv.mul(scale)
                        .mul(xvv)
                        .intoMemorySegment(
                                o.vseg(),
                                o.vbase() + (outOffset + i) * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < size; i++) {
                writeFloat(
                        o.vseg(),
                        o.vbase() + (outOffset + i) * Float.BYTES,
                        readFloat(w.vseg(), w.vbase() + (long) i * Float.BYTES)
                                * ss
                                * readFloat(xv.vseg(), xv.vbase() + (xOffset + i) * Float.BYTES));
            }
            return;
        }
        float ss = 0f;
        for (int i = 0; i < size; i++) {
            float xi = readFloat(xv.vseg(), xv.vbase() + (xOffset + i) * Float.BYTES);
            ss += xi * xi;
        }
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        for (int i = 0; i < size; i++) {
            writeFloat(
                    o.vseg(),
                    o.vbase() + (outOffset + i) * Float.BYTES,
                    readFloat(w.vseg(), w.vbase() + (long) i * Float.BYTES)
                            * ss
                            * readFloat(xv.vseg(), xv.vbase() + (xOffset + i) * Float.BYTES));
        }
    }

    /**
     * Per-row {@link #rmsnorm} over {@code rows} rows of {@code rowDim} lanes ({@code out == x} for
     * in-place post-norms) - the pre/post-norm idiom of a transformer block, shared by every model
     * port so none of them re-rolls the row loop.
     */
    public static void rmsnormRows(
            MemoryView<MemorySegment> out,
            MemoryView<MemorySegment> x,
            MemoryView<MemorySegment> weight,
            int rows,
            int rowDim,
            float rmsNormEps) {
        Parallel.forLoop(
                rows,
                r ->
                        rmsnorm(
                                out,
                                (long) r * rowDim,
                                x,
                                (long) r * rowDim,
                                weight,
                                rowDim,
                                rmsNormEps));
    }

    /**
     * Sum of squares of {@code size} contiguous lanes from {@code xOffset} (shared rms-scale
     * derivation).
     */
    public static float sumOfSquares(MemoryView<MemorySegment> x, long xOffset, int size) {
        Raw xv = Raw.f32(x, "x");
        if (USE_VECTOR_API) {
            var species = F_SPECIES;
            int upperBound = species.loopBound(size);
            FloatVector acc = FloatVector.zero(species);
            int i = 0;
            for (; i < upperBound; i += species.length()) {
                var xvv =
                        FloatVector.fromMemorySegment(
                                species,
                                xv.vseg(),
                                xv.vbase() + (xOffset + i) * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
                acc = xvv.fma(xvv, acc);
            }
            float ss = acc.reduceLanes(VectorOperators.ADD);
            for (; i < size; i++) {
                float xi = readFloat(xv.vseg(), xv.vbase() + (xOffset + i) * Float.BYTES);
                ss += xi * xi;
            }
            return ss;
        }
        float ss = 0f;
        for (int i = 0; i < size; i++) {
            float xi = readFloat(xv.vseg(), xv.vbase() + (xOffset + i) * Float.BYTES);
            ss += xi * xi;
        }
        return ss;
    }

    /**
     * {@code out = weight * scale * x} over {@code size} lanes - the apply half of {@link #rmsnorm}
     * with a caller-supplied {@code scale}.
     */
    public static void scaleByWeight(
            MemoryView<MemorySegment> out,
            long outOffset,
            MemoryView<MemorySegment> x,
            long xOffset,
            MemoryView<MemorySegment> weight,
            int size,
            float scale) {
        Raw o = Raw.f32(out, "out");
        Raw xv = Raw.f32(x, "x");
        Raw w = Raw.f32(weight, "weight");
        if (USE_VECTOR_API) {
            var species = F_SPECIES;
            int upperBound = species.loopBound(size);
            FloatVector sv = FloatVector.broadcast(species, scale);
            int i = 0;
            for (; i < upperBound; i += species.length()) {
                var xvv =
                        FloatVector.fromMemorySegment(
                                species,
                                xv.vseg(),
                                xv.vbase() + (xOffset + i) * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
                var wv =
                        FloatVector.fromMemorySegment(
                                species,
                                w.vseg(),
                                w.vbase() + (long) i * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
                wv.mul(sv)
                        .mul(xvv)
                        .intoMemorySegment(
                                o.vseg(),
                                o.vbase() + (outOffset + i) * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < size; i++) {
                writeFloat(
                        o.vseg(),
                        o.vbase() + (outOffset + i) * Float.BYTES,
                        readFloat(w.vseg(), w.vbase() + (long) i * Float.BYTES)
                                * scale
                                * readFloat(xv.vseg(), xv.vbase() + (xOffset + i) * Float.BYTES));
            }
            return;
        }
        for (int i = 0; i < size; i++) {
            writeFloat(
                    o.vseg(),
                    o.vbase() + (outOffset + i) * Float.BYTES,
                    readFloat(w.vseg(), w.vbase() + (long) i * Float.BYTES)
                            * scale
                            * readFloat(xv.vseg(), xv.vbase() + (xOffset + i) * Float.BYTES));
        }
    }

    /**
     * Bare RMS norm (normalize to unit RMS, no learned weights). {@code out} may be {@code x} (the
     * common in-place case never copies).
     */
    public static void rmsnormNoWeight(
            MemoryView<MemorySegment> out,
            long outOffset,
            MemoryView<MemorySegment> x,
            long xOffset,
            int size,
            float eps) {
        float rms = (float) Math.sqrt(sumOfSquares(x, xOffset, size) / size + eps);
        if (out == x && outOffset == xOffset) {
            Ops.divideInPlace(x, xOffset, size, rms);
            return;
        }
        Raw o = Raw.f32(out, "out");
        Raw xv = Raw.f32(x, "x");
        for (int i = 0; i < size; i++) {
            writeFloat(
                    o.vseg(),
                    o.vbase() + (outOffset + i) * Float.BYTES,
                    readFloat(xv.vseg(), xv.vbase() + (xOffset + i) * Float.BYTES) / rms);
        }
    }

    /**
     * LayerNorm over FP32 activations: {@code out = gamma·(x-mean)/σ + beta}, time-major ({@code
     * data[t*C + c]}, channels contiguous). {@code out} and {@code x} may be the same view. Gamma
     * and beta must be FP32 (the old F16 scalar fallback is gone by contract).
     */
    public static void layerNorm(
            MemoryView<MemorySegment> out,
            MemoryView<MemorySegment> x,
            MemoryView<MemorySegment> gamma,
            MemoryView<MemorySegment> beta,
            int C,
            int T,
            float eps) {
        Raw o = Raw.f32(out, "out");
        Raw xv = Raw.f32(x, "x");
        Raw g = Raw.f32(gamma, "gamma");
        Raw b = Raw.f32(beta, "beta");
        var sp = F_SPECIES;
        int bound = USE_VECTOR_API ? sp.loopBound(C) : 0;
        for (int t = 0; t < T; t++) {
            long row = (long) t * C;
            float mean = 0;
            for (int c = 0; c < C; c++) {
                mean += readFloat(xv.vseg(), xv.vbase() + (row + c) * Float.BYTES);
            }
            mean /= C;
            float variance = 0;
            for (int c = 0; c < C; c++) {
                float d = readFloat(xv.vseg(), xv.vbase() + (row + c) * Float.BYTES) - mean;
                variance += d * d;
            }
            float inv = (float) (1.0 / Math.sqrt(variance / C + eps));
            int c = 0;
            if (USE_VECTOR_API) {
                var means = FloatVector.broadcast(sp, mean);
                var invs = FloatVector.broadcast(sp, inv);
                for (; c < bound; c += sp.length()) {
                    long byteOff = xv.vbase() + (row + c) * Float.BYTES;
                    var v =
                            FloatVector.fromMemorySegment(
                                            sp, xv.vseg(), byteOff, ByteOrder.LITTLE_ENDIAN)
                                    .sub(means)
                                    .mul(invs)
                                    .mul(
                                            FloatVector.fromMemorySegment(
                                                    sp,
                                                    g.vseg(),
                                                    g.vbase() + (long) c * Float.BYTES,
                                                    ByteOrder.LITTLE_ENDIAN))
                                    .add(
                                            FloatVector.fromMemorySegment(
                                                    sp,
                                                    b.vseg(),
                                                    b.vbase() + (long) c * Float.BYTES,
                                                    ByteOrder.LITTLE_ENDIAN));
                    v.intoMemorySegment(
                            o.vseg(), o.vbase() + (row + c) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                }
            }
            for (; c < C; c++) {
                writeFloat(
                        o.vseg(),
                        o.vbase() + (row + c) * Float.BYTES,
                        (readFloat(xv.vseg(), xv.vbase() + (row + c) * Float.BYTES) - mean)
                                        * inv
                                        * readFloat(g.vseg(), g.vbase() + (long) c * Float.BYTES)
                                + readFloat(b.vseg(), b.vbase() + (long) c * Float.BYTES));
            }
        }
    }

    /**
     * In-place LayerNorm and ReLU across channels of channel-major {@code [channel][position]}
     * data. There is a learned weight and no bias.
     */
    public static void layerNormChannelsReluInPlace(
            MemoryView<MemorySegment> x,
            MemoryView<MemorySegment> weight,
            int channels,
            int positions,
            float eps) {
        Raw xv = Raw.f32(x, "x");
        Raw w = Raw.f32(weight, "weight");
        Parallel.forLoop(
                0,
                positions,
                p -> {
                    double mean = 0;
                    for (int c = 0; c < channels; c++) {
                        mean +=
                                readFloat(
                                        xv.vseg(),
                                        xv.vbase() + ((long) c * positions + p) * Float.BYTES);
                    }
                    mean /= channels;
                    double variance = 0;
                    for (int c = 0; c < channels; c++) {
                        double d =
                                readFloat(
                                                xv.vseg(),
                                                xv.vbase()
                                                        + ((long) c * positions + p) * Float.BYTES)
                                        - mean;
                        variance += d * d;
                    }
                    float inv = (float) (1.0 / Math.sqrt(variance / channels + eps));
                    for (int c = 0; c < channels; c++) {
                        long at = xv.vbase() + ((long) c * positions + p) * Float.BYTES;
                        float normalized =
                                (float) ((readFloat(xv.vseg(), at) - mean) * inv)
                                        * readFloat(w.vseg(), w.vbase() + (long) c * Float.BYTES);
                        writeFloat(xv.vseg(), at, Math.max(0f, normalized));
                    }
                });
    }
}
