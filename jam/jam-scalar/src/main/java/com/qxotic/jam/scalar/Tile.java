package com.qxotic.jam.scalar;

/**
 * The register tile both prefill kernels run, and the loop shape both JITs vectorize:
 *
 * <pre>
 *   c_j[i] = fma(q[l+3][i], x_j[l+3], fma(q[l+2][i], x_j[l+2], fma(q[l+1][i], x_j[l+1], fma(q[l][i], x_j[l], c_j[i]))))
 * </pre>
 *
 * for three accumulator rows {@code j} and four {@code l} per pass: twelve vector FMAs per seven
 * vector loads and three stores, every array indexed by the bare lane {@code i}. The panel rows
 * {@code q} are the SIMD-axis operand (tokens in {@link Gemm}, weight rows in {@link RowGemm}),
 * {@code x} holds the twelve broadcast scalars in three rows at stride {@link #XS}. Four k-steps
 * per c load/store keep the store port idle, three rows share each panel load, 12 broadcasts plus
 * the unrolled temporaries fit the 32 vector registers, and nothing else is live: one more general
 * register and C2 spills the accumulator pointers out of the hot loop (-20%). Measured on Zen 5:
 * 60-70 GMAC/s per core on either JIT at 512 lanes, 55 at 256, 40 at 128 (256-bit).
 *
 * <p>Every other form measured loses on C2 (JDK 25): an offset index ({@code c[at + i]}), a {@code
 * long} induction variable or a manually unrolled body are left scalar (C2 only orders the loads
 * and stores of distinct arrays when every index is the bare lane), a mul-add tree or a flat panel
 * with per-row offsets likewise, an exact constant bound gains nothing, and several lane loops in
 * one method blow its loop budget. Graal compiles most of them; this one it compiles best.
 *
 * <p>One C2 limitation no loop shape evades: it sizes the vectors from the trip count its tier-3
 * profile recorded, capped at about a quarter of it, and that profile is racy. Gathered on 16
 * threads at once the counters lose most increments (a 512-lane loop was recorded as 72) and C2
 * then settles on 256-bit vectors, or none, for the life of the process - at random between runs,
 * -35% on the kernel and up to -15% on prefill. Four threads, or a single-threaded first call,
 * record it right every time. Callers keep the lane count at {@link #LANES} or more so the recorded
 * count stays as high as it can.
 */
final class Tile {

    private Tile() {}

    /** k-steps folded into one c load/store. */
    static final int KU = 4;

    /** Accumulator rows per pass. */
    static final int TR = 3;

    /** The fewest lanes a sweep is run over; a shorter lane axis is padded up to it. */
    static final int LANES = 256;

    /** Stride of the three scalar rows in the tile's {@code x} operand: room for any k-block. */
    static final int XS = Gemm.KC_MAX + KU;

    /**
     * {@code c_j[i] += sum_{u<4} q[l+u][i] x[j*XS + l+u]} over {@code kc} k-steps (a multiple of
     * {@link #KU}) and {@code len} lanes.
     */
    static void sweep(float[][] q, int kc, float[] x, float[] c0, float[] c1, float[] c2, int len) {
        for (int l = 0; l < kc; l += 4) {
            float[] q0 = q[l], q1 = q[l + 1], q2 = q[l + 2], q3 = q[l + 3];
            float w00 = x[l], w01 = x[l + 1], w02 = x[l + 2], w03 = x[l + 3];
            float w10 = x[XS + l], w11 = x[XS + l + 1], w12 = x[XS + l + 2], w13 = x[XS + l + 3];
            float w20 = x[2 * XS + l],
                    w21 = x[2 * XS + l + 1],
                    w22 = x[2 * XS + l + 2],
                    w23 = x[2 * XS + l + 3];
            for (int i = 0; i < len; i++) {
                float v0 = q0[i], v1 = q1[i], v2 = q2[i], v3 = q3[i];
                c0[i] =
                        Math.fma(
                                v3,
                                w03,
                                Math.fma(v2, w02, Math.fma(v1, w01, Math.fma(v0, w00, c0[i]))));
                c1[i] =
                        Math.fma(
                                v3,
                                w13,
                                Math.fma(v2, w12, Math.fma(v1, w11, Math.fma(v0, w10, c1[i]))));
                c2[i] =
                        Math.fma(
                                v3,
                                w23,
                                Math.fma(v2, w22, Math.fma(v1, w21, Math.fma(v0, w20, c2[i]))));
            }
        }
    }

    /** The lanes to sweep for {@code n} used ones: {@code n}, or {@link #LANES} if that is more. */
    static int lanes(int n) {
        return Math.max(n, LANES);
    }

    /** {@code ceil(a / b)}. */
    static int ceilDiv(int a, int b) {
        return (a + b - 1) / b;
    }

    /** {@code n} rounded up to a multiple of {@link #KU}. */
    static int padK(int n) {
        return ceilDiv(n, KU) * KU;
    }
}
