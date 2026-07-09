package com.qxotic.jinfer;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Public SIMD primitives over contiguous {@code float[]} ranges, for model ports that live outside
 * this package (e.g. the gated-delta-net recurrence in {@code
 * com.qxotic.jinfer.models.qwen35.Qwen35}) and so cannot reach {@link FloatTensor}'s
 * package-private {@code F_SPECIES}/{@code USE_VECTOR_API}. Same vector path {@link FloatTensor}
 * uses, with a scalar fallback.
 */
public final class VectorMath {
    private VectorMath() {}

    /** Dot product {@code A[ao..ao+n) · B[bo..bo+n)} (SIMD-tree reduction). */
    public static float dot(float[] A, int ao, float[] B, int bo, int n) {
        if (FloatTensor.USE_VECTOR_API) {
            VectorSpecies<Float> sp = FloatTensor.F_SPECIES;
            int u = sp.length(), i = 0;
            var acc = FloatVector.zero(sp);
            for (; i + u <= n; i += u)
                acc =
                        FloatVector.fromArray(sp, A, ao + i)
                                .fma(FloatVector.fromArray(sp, B, bo + i), acc);
            float s = acc.reduceLanes(VectorOperators.ADD);
            for (; i < n; i++) s += A[ao + i] * B[bo + i];
            return s;
        }
        float s = 0;
        for (int i = 0; i < n; i++) s += A[ao + i] * B[bo + i];
        return s;
    }

    /** {@code Y[yo..yo+n) += s * X[xo..xo+n)}. */
    public static void axpy(float[] Y, int yo, float s, float[] X, int xo, int n) {
        if (FloatTensor.USE_VECTOR_API) {
            VectorSpecies<Float> sp = FloatTensor.F_SPECIES;
            int u = sp.length(), i = 0;
            var sv = FloatVector.broadcast(sp, s);
            for (; i + u <= n; i += u)
                FloatVector.fromArray(sp, X, xo + i)
                        .fma(sv, FloatVector.fromArray(sp, Y, yo + i))
                        .intoArray(Y, yo + i);
            for (; i < n; i++) Y[yo + i] += s * X[xo + i];
            return;
        }
        for (int i = 0; i < n; i++) Y[yo + i] += s * X[xo + i];
    }

    /** {@code Y[yo..yo+n) *= s}. */
    public static void scale(float[] Y, int yo, float s, int n) {
        if (FloatTensor.USE_VECTOR_API) {
            VectorSpecies<Float> sp = FloatTensor.F_SPECIES;
            int u = sp.length(), i = 0;
            var sv = FloatVector.broadcast(sp, s);
            for (; i + u <= n; i += u)
                FloatVector.fromArray(sp, Y, yo + i).mul(sv).intoArray(Y, yo + i);
            for (; i < n; i++) Y[yo + i] *= s;
            return;
        }
        for (int i = 0; i < n; i++) Y[yo + i] *= s;
    }
}
