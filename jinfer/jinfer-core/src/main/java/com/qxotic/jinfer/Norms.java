// Normalization kernels shared across architectures. Like RoPE/Activations, this is shared
// infrastructure: every model normalizes rows the same way, only the learned weights and eps
// differ. Each kernel keeps a vectorized fast path AND a pure-Java scalar fallback — the fallback
// runs when the Vector API is unavailable and is the correctness oracle the parity tests check
// the vector path against (their summation orders differ at the ulp level; see FIXES.md).
package com.qxotic.jinfer;

import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

public final class Norms {
    private Norms() {}

    /**
     * RMS normalization: {@code out = weight * x / sqrt(mean(x^2) + eps)} over {@code size}
     * contiguous lanes. F32 tensors take an explicit Vector API path (scalar segment loops do not
     * auto-vectorize); everything else falls through to the scalar loop.
     */
    public static void rmsnorm(
            FloatTensor out,
            long outOffset,
            FloatTensor x,
            long xOffset,
            F32FloatTensor weight,
            int size,
            float rmsNormEps) {
        if (out instanceof F32FloatTensor outF32
                && x instanceof F32FloatTensor xF32
                && FloatTensor.USE_VECTOR_API) {
            // All lanes load via (vseg, vbase): with GLOBAL_SEGMENT every fromMemorySegment call
            // site sees a single segment implementation type, which native-image AOT requires.
            var species = FloatTensor.F_SPECIES;
            int upperBound = species.loopBound(size);
            FloatVector acc = FloatVector.zero(species);
            int i = 0;
            for (; i < upperBound; i += species.length()) {
                var xv =
                        FloatVector.fromMemorySegment(
                                species,
                                xF32.vseg,
                                xF32.vbase + (long) (xOffset + i) * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
                acc = xv.fma(xv, acc);
            }
            float ss = acc.reduceLanes(VectorOperators.ADD);
            for (; i < size; i++) {
                float xi = x.getFloat(xOffset + i);
                ss += xi * xi;
            }
            ss /= size;
            ss += rmsNormEps;
            ss = (float) (1.0 / Math.sqrt(ss));
            FloatVector scale = FloatVector.broadcast(species, ss);
            for (i = 0; i < upperBound; i += species.length()) {
                var xv =
                        FloatVector.fromMemorySegment(
                                species,
                                xF32.vseg,
                                xF32.vbase + (long) (xOffset + i) * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
                var wv =
                        FloatVector.fromMemorySegment(
                                species,
                                weight.vseg,
                                weight.vbase + (long) i * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
                wv.mul(scale)
                        .mul(xv)
                        .intoMemorySegment(
                                outF32.vseg,
                                outF32.vbase + (long) (outOffset + i) * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < size; i++) {
                out.setFloat(outOffset + i, weight.getFloat(i) * ss * x.getFloat(xOffset + i));
            }
            return;
        }
        float ss = 0f;
        for (int i = 0; i < size; i++) {
            float xi = x.getFloat(xOffset + i);
            ss += xi * xi;
        }
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        for (int i = 0; i < size; i++) {
            out.setFloat(outOffset + i, weight.getFloat(i) * ss * x.getFloat(xOffset + i));
        }
    }

    /** Convenience overload normalizing from offset 0. */
    public static void rmsnorm(
            FloatTensor out, FloatTensor x, F32FloatTensor weight, int size, float eps) {
        rmsnorm(out, 0, x, 0, weight, size, eps);
    }

    /**
     * Sum of squares of {@code size} contiguous lanes from {@code xOffset}. Vectorized for F32
     * (same fma accumulation + scalar tail as {@link #rmsnorm}, so a caller can derive the rms
     * scale once and share it across several normalizations of the same row instead of recomputing
     * the reduction).
     */
    public static float sumOfSquares(FloatTensor x, long xOffset, int size) {
        if (x instanceof F32FloatTensor xF32 && FloatTensor.USE_VECTOR_API) {
            var species = FloatTensor.F_SPECIES;
            int upperBound = species.loopBound(size);
            FloatVector acc = FloatVector.zero(species);
            int i = 0;
            for (; i < upperBound; i += species.length()) {
                var xv =
                        FloatVector.fromMemorySegment(
                                species,
                                xF32.vseg,
                                xF32.vbase + (long) (xOffset + i) * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
                acc = xv.fma(xv, acc);
            }
            float ss = acc.reduceLanes(VectorOperators.ADD);
            for (; i < size; i++) {
                float xi = x.getFloat(xOffset + i);
                ss += xi * xi;
            }
            return ss;
        }
        float ss = 0f;
        for (int i = 0; i < size; i++) {
            float xi = x.getFloat(xOffset + i);
            ss += xi * xi;
        }
        return ss;
    }

    /**
     * {@code out = weight * scale * x} over {@code size} lanes — the apply half of {@link #rmsnorm}
     * with a caller-supplied {@code scale} (e.g. a shared {@code 1/rms}). Vectorized for F32.
     */
    public static void scaleByWeight(
            FloatTensor out,
            long outOffset,
            FloatTensor x,
            long xOffset,
            F32FloatTensor weight,
            int size,
            float scale) {
        if (out instanceof F32FloatTensor outF32
                && x instanceof F32FloatTensor xF32
                && FloatTensor.USE_VECTOR_API) {
            var species = FloatTensor.F_SPECIES;
            int upperBound = species.loopBound(size);
            FloatVector sv = FloatVector.broadcast(species, scale);
            int i = 0;
            for (; i < upperBound; i += species.length()) {
                var xv =
                        FloatVector.fromMemorySegment(
                                species,
                                xF32.vseg,
                                xF32.vbase + (long) (xOffset + i) * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
                var wv =
                        FloatVector.fromMemorySegment(
                                species,
                                weight.vseg,
                                weight.vbase + (long) i * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
                wv.mul(sv)
                        .mul(xv)
                        .intoMemorySegment(
                                outF32.vseg,
                                outF32.vbase + (long) (outOffset + i) * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
            }
            for (; i < size; i++)
                out.setFloat(outOffset + i, weight.getFloat(i) * scale * x.getFloat(xOffset + i));
            return;
        }
        for (int i = 0; i < size; i++)
            out.setFloat(outOffset + i, weight.getFloat(i) * scale * x.getFloat(xOffset + i));
    }

    /** Bare RMS norm (normalize to unit RMS, no learned weights) — e.g. Gemma's V norm. */
    public static void rmsnormNoWeight(
            FloatTensor out, long outOffset, FloatTensor x, long xOffset, int size, float eps) {
        float ss = 0f;
        for (int i = 0; i < size; i++) {
            float xi = x.getFloat(xOffset + i);
            ss += xi * xi;
        }
        ss = (float) (1.0 / Math.sqrt(ss / size + eps));
        for (int i = 0; i < size; i++) {
            out.setFloat(outOffset + i, ss * x.getFloat(xOffset + i));
        }
    }
}
