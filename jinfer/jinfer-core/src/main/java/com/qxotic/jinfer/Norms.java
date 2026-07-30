// Normalization kernels shared across architectures. Like RoPE/Activations, this is shared
// infrastructure: every model normalizes rows the same way, only the learned weights and eps
// differ. Each kernel keeps a vectorized fast path AND a pure-Java scalar fallback — the fallback
// runs when the Vector API is unavailable and is the correctness oracle the parity tests check
// the vector path against (their summation orders differ at the ulp level; see FIXES.md).
package com.qxotic.jinfer;

import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

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

    // ═════════════════════════════════════════════════════════════════════
    // LayerNorm — channel-wise normalization for time-major float[] data.
    // Mirrors PyTorch F.layer_norm(x, (C,), gamma, beta, eps) across T steps.
    // Per step: mean, var over C contiguous channels, then gamma·(x-mean)/σ + beta.
    // Vectorized for F32 lanes when channels are contiguous.
    // ═════════════════════════════════════════════════════════════════════

    /**
     * Apply LayerNorm to time-major float[] data in-place. {@code data[t*C + c]}; channels are
     * contiguous per step. {@code gamma} and {@code beta} are length {@code C}.
     */
    public static void layerNorm(
            float[] data, float[] gamma, float[] beta, int C, int T, float eps) {
        layerNorm(data, data, gamma, beta, C, T, eps);
    }

    /** Apply LayerNorm: {@code out = gamma·(x-mean)/σ + beta}. Input and output may alias. */
    public static void layerNorm(
            float[] out, float[] x, float[] gamma, float[] beta, int C, int T, float eps) {
        if (FloatTensor.USE_VECTOR_API && FloatTensor.F_SPECIES != null) {
            layerNormVector(out, x, gamma, beta, C, T, eps, FloatTensor.F_SPECIES);
        } else {
            layerNormScalar(out, x, gamma, beta, C, T, eps);
        }
    }

    /** LayerNorm with FloatTensor weights — transparently handles F16/F32/quantized. */
    public static void layerNorm(
            float[] out, float[] x, FloatTensor gamma, FloatTensor beta, int C, int T, float eps) {
        if (gamma instanceof F32FloatTensor g && beta instanceof F32FloatTensor b) {
            layerNormF32(out, x, g, b, C, T, eps);
        } else {
            float[] g = new float[C], bt = new float[C];
            for (int c = 0; c < C; c++) {
                g[c] = gamma.getFloat(c);
                bt[c] = beta.getFloat(c);
            }
            layerNorm(out, x, g, bt, C, T, eps);
        }
    }

    /**
     * LayerNorm over native F32 activations: {@code out = gamma·(x-mean)/σ + beta}, time-major
     * ({@code data[t*C + c]}, channels contiguous). {@code out} and {@code x} may be the same
     * tensor. The float[] overloads above are the same math on heap data.
     */
    public static void layerNorm(
            F32FloatTensor out,
            F32FloatTensor x,
            FloatTensor gamma,
            FloatTensor beta,
            int C,
            int T,
            float eps) {
        var sp = FloatTensor.F_SPECIES;
        // Only F32 weights can be read as vectors; F16/quantized gamma go down the scalar path.
        boolean vector =
                FloatTensor.USE_VECTOR_API
                        && sp != null
                        && gamma instanceof F32FloatTensor
                        && beta instanceof F32FloatTensor;
        int bound = vector ? sp.loopBound(C) : 0;
        for (int t = 0; t < T; t++) {
            long row = (long) t * C;
            float mean = 0;
            for (int c = 0; c < C; c++) mean += x.getFloat(row + c);
            mean /= C;
            float variance = 0;
            for (int c = 0; c < C; c++) {
                float d = x.getFloat(row + c) - mean;
                variance += d * d;
            }
            float inv = (float) (1.0 / Math.sqrt(variance / C + eps));
            int c = 0;
            if (vector) {
                var means = FloatVector.broadcast(sp, mean);
                var invs = FloatVector.broadcast(sp, inv);
                for (; c < bound; c += sp.length()) {
                    long byteOff = x.vbase + (row + c) * Float.BYTES;
                    var v =
                            FloatVector.fromMemorySegment(
                                            sp, x.vseg, byteOff, ByteOrder.LITTLE_ENDIAN)
                                    .sub(means)
                                    .mul(invs)
                                    .mul(gamma.getFloatVector(sp, c))
                                    .add(beta.getFloatVector(sp, c));
                    v.intoMemorySegment(
                            out.vseg, out.vbase + (row + c) * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                }
            }
            for (; c < C; c++)
                out.setFloat(
                        row + c,
                        (x.getFloat(row + c) - mean) * inv * gamma.getFloat(c) + beta.getFloat(c));
        }
    }

    private static void layerNormF32(
            float[] out,
            float[] x,
            F32FloatTensor gamma,
            F32FloatTensor beta,
            int C,
            int T,
            float eps) {
        if (FloatTensor.USE_VECTOR_API && FloatTensor.F_SPECIES != null) {
            var sp = FloatTensor.F_SPECIES;
            long gb = gamma.vbase, bb = beta.vbase;
            var gs = gamma.vseg;
            var bs = beta.vseg;
            int bound = sp.loopBound(C);
            for (int t = 0; t < T; t++) {
                int bi = t * C;
                FloatVector acc = FloatVector.zero(sp);
                int c = 0;
                for (; c < bound; c += sp.length())
                    acc = acc.add(FloatVector.fromArray(sp, x, bi + c));
                float mean = acc.reduceLanes(VectorOperators.ADD);
                for (; c < C; c++) mean += x[bi + c];
                mean /= C;
                FloatVector accSq = FloatVector.zero(sp);
                c = 0;
                for (; c < bound; c += sp.length()) {
                    var xv = FloatVector.fromArray(sp, x, bi + c);
                    var diff = xv.sub(mean);
                    accSq = diff.fma(diff, accSq);
                }
                float var = accSq.reduceLanes(VectorOperators.ADD);
                for (; c < C; c++) {
                    float d = x[bi + c] - mean;
                    var += d * d;
                }
                float invStd = 1f / (float) Math.sqrt(var / C + eps);
                c = 0;
                for (; c < bound; c += sp.length()) {
                    long off = (long) c * Float.BYTES;
                    var xv = FloatVector.fromArray(sp, x, bi + c);
                    var gv =
                            FloatVector.fromMemorySegment(
                                    sp, gs, gb + off, ByteOrder.LITTLE_ENDIAN);
                    var bv =
                            FloatVector.fromMemorySegment(
                                    sp, bs, bb + off, ByteOrder.LITTLE_ENDIAN);
                    xv.sub(mean).mul(invStd).mul(gv).add(bv).intoArray(out, bi + c);
                }
                for (; c < C; c++)
                    out[bi + c] =
                            (x[bi + c] - mean) * invStd * gamma.getFloat(c) + beta.getFloat(c);
            }
        } else {
            float[] g = new float[C], bt = new float[C];
            for (int c = 0; c < C; c++) {
                g[c] = gamma.getFloat(c);
                bt[c] = beta.getFloat(c);
            }
            layerNormScalar(out, x, g, bt, C, T, eps);
        }
    }

    private static void layerNormScalar(
            float[] out, float[] x, float[] gamma, float[] beta, int C, int T, float eps) {
        for (int t = 0; t < T; t++) {
            int base = t * C;
            float sum = 0f;
            for (int c = 0; c < C; c++) sum += x[base + c];
            float mean = sum / C;
            float var = 0f;
            for (int c = 0; c < C; c++) {
                float d = x[base + c] - mean;
                var += d * d;
            }
            float invStd = 1f / (float) Math.sqrt(var / C + eps);
            for (int c = 0; c < C; c++)
                out[base + c] = (x[base + c] - mean) * invStd * gamma[c] + beta[c];
        }
    }

    private static void layerNormVector(
            float[] out,
            float[] x,
            float[] gamma,
            float[] beta,
            int C,
            int T,
            float eps,
            VectorSpecies<Float> sp) {
        int bound = sp.loopBound(C);
        for (int t = 0; t < T; t++) {
            int bi = t * C;

            // Pass 1: mean
            FloatVector acc = FloatVector.zero(sp);
            int c = 0;
            for (; c < bound; c += sp.length()) acc = acc.add(FloatVector.fromArray(sp, x, bi + c));
            float mean = acc.reduceLanes(VectorOperators.ADD);
            for (; c < C; c++) mean += x[bi + c];
            mean /= C;

            // Pass 2: variance
            FloatVector accSq = FloatVector.zero(sp);
            c = 0;
            for (; c < bound; c += sp.length()) {
                var xv = FloatVector.fromArray(sp, x, bi + c);
                var diff = xv.sub(mean);
                accSq = diff.fma(diff, accSq);
            }
            float var = accSq.reduceLanes(VectorOperators.ADD);
            for (; c < C; c++) {
                float d = x[bi + c] - mean;
                var += d * d;
            }
            float invStd = 1f / (float) Math.sqrt(var / C + eps);

            // Pass 3: apply
            c = 0;
            for (; c < bound; c += sp.length()) {
                var xv = FloatVector.fromArray(sp, x, bi + c);
                var gv = FloatVector.fromArray(sp, gamma, c);
                var bv = FloatVector.fromArray(sp, beta, c);
                xv.sub(mean).mul(invStd).mul(gv).add(bv).intoArray(out, bi + c);
            }
            for (; c < C; c++) out[bi + c] = (x[bi + c] - mean) * invStd * gamma[c] + beta[c];
        }
    }
}
