package com.qxotic.jam;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

import static com.qxotic.jam.VectorSupport.F_SPECIES;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;

/**
 * Shared decode-free F32 band gemm — the back half of every dequant-to-scratch kernel (MXFP4, NVFP4, and the
 * k-quants Q4_K/Q5_K/Q6_K). The dtype kernel dequantizes a group of {@link #MR} weight rows into an F32
 * scratch ({@code dequantizeRow} per row), then the columns sweep this band with no per-element decode — so
 * the expensive part of a quant (its scale/nibble unpack) is amortized once per row, not per column tile.
 */
final class BandGemm {

    private BandGemm() {}

    /** Band register-tile shape: 4x4 (16 F32 accumulators, ~+22% prefill) on JITs that keep it spill-free,
     *  else 3x3 (9 accumulators) — the same wide-tile gate as the Q8_0 4x4 tile. Override with
     *  {@code -Djam.vector.band=3x3|4x4}. The dtype kernel dequantizes MR rows per band; the column sweep
     *  then holds MR*NR F32 accumulators with no per-element decode. */
    static final String BAND = VectorSupport.jamProp("jam.vector.band",
            VectorSupport.WIDE_TILE && VectorSupport.IS_512 ? "4x4" : "3x3");   // 4x4's 24 live vectors need 32 zmm
    static final int MR = BAND.equals("3x3") ? 3 : 4;
    static final int NR = BAND.equals("3x3") ? 3 : 4;

    /** Decode one weight row ({@code dim1} elements at element offset {@code rowElemOffset}) into {@code dst}
     *  at {@code dstOffset} — the ONLY part that differs between the dequant-to-scratch dtypes. */
    @FunctionalInterface
    interface RowDequant {
        void dequantize(MemorySegment w, long rowElemOffset, int dim1, float[] dst, int dstOffset);
    }

    /** The dequant-to-scratch band gemm, shared by every such dtype (k-quants + FP4). Tiles {@code m} into
     *  {@link #MR}-row groups (parallel), dequantizes each group's rows ONCE via {@code deq}, sweeps the
     *  columns with {@link #band}, and finishes the n/m remainders with per-column {@link #dotDeq} dots. A
     *  dtype kernel is exactly this driver bound to its own {@code dequantizeRow}. The {@code deq} call is
     *  per-row (m total) so its dispatch is free against the row's decode work. */
    static void gemm(MemorySegment w, MemorySegment a, long aBase, MemorySegment o, long oBase,
                     int aStride, int oStride, int n, int m, int k, long wOff, Scratch scratch, RowDequant deq) {
        int groups = m / MR;
        VectorSupport.parallelChunks(groups, (gLo, gHi) -> {
            float[] band = scratch.acquire(MR * k);   // one per worker; reused across mm, freed with the context
            try {
                for (int g = gLo; g < gHi; g++) {
                    int row0 = g * MR;
                    for (int i = 0; i < MR; i++) deq.dequantize(w, wOff + (long) (row0 + i) * k, k, band, i * k);
                    int s = 0;
                    for (; s + NR <= n; s += NR) band(band, k, a, aBase, o, oBase, aStride, oStride, row0, s);
                    for (; s < n; s++) for (int i = 0; i < MR; i++)
                        store(o, oBase, (long) s * oStride + row0 + i, dotDeq(band, i * k, k, a, aBase, (long) s * aStride));
                }
            } finally {
                scratch.release(band);
            }
        });
        int rem0 = groups * MR;
        if (rem0 < m) {                                // trailing rows (m % MR): cheap per-column dots
            float[] band = scratch.acquire(k);
            for (int row = rem0; row < m; row++) {
                deq.dequantize(w, wOff + (long) row * k, k, band, 0);
                int rr = row;
                VectorSupport.parallelFor(0, n, s -> store(o, oBase, (long) s * oStride + rr, dotDeq(band, 0, k, a, aBase, (long) s * aStride)));
            }
            scratch.release(band);
        }
    }

    /** Sweep one MR x NR band; constant-folds to the selected shape ({@link #BAND} is static final). */
    static void band(float[] w, int dim1, MemorySegment a, long aBase, MemorySegment o, long oBase,
                     int aStride, int oStride, int row0, int s0) {
        if (BAND.equals("3x3")) gemm512Band3x3(w, dim1, a, aBase, o, oBase, aStride, oStride, row0, s0);
        else gemm512Band4x4(w, dim1, a, aBase, o, oBase, aStride, oStride, row0, s0);
    }

    static void gemm512Band4x4(float[] w, int dim1, MemorySegment a, long aBase, MemorySegment o, long oBase,
                               int aStride, int oStride, int row0, int s0) {
        long b0 = (long) s0 * aStride, b1 = b0 + aStride, b2 = b1 + aStride, b3 = b2 + aStride;
        FloatVector c00 = FloatVector.zero(F_SPECIES), c01 = FloatVector.zero(F_SPECIES), c02 = FloatVector.zero(F_SPECIES), c03 = FloatVector.zero(F_SPECIES);
        FloatVector c10 = FloatVector.zero(F_SPECIES), c11 = FloatVector.zero(F_SPECIES), c12 = FloatVector.zero(F_SPECIES), c13 = FloatVector.zero(F_SPECIES);
        FloatVector c20 = FloatVector.zero(F_SPECIES), c21 = FloatVector.zero(F_SPECIES), c22 = FloatVector.zero(F_SPECIES), c23 = FloatVector.zero(F_SPECIES);
        FloatVector c30 = FloatVector.zero(F_SPECIES), c31 = FloatVector.zero(F_SPECIES), c32 = FloatVector.zero(F_SPECIES), c33 = FloatVector.zero(F_SPECIES);
        int len = F_SPECIES.length();
        for (int kk = 0; kk < dim1; kk += len) {
            FloatVector w0 = FloatVector.fromArray(F_SPECIES, w, kk), w1 = FloatVector.fromArray(F_SPECIES, w, dim1 + kk);
            FloatVector w2 = FloatVector.fromArray(F_SPECIES, w, 2 * dim1 + kk), w3 = FloatVector.fromArray(F_SPECIES, w, 3 * dim1 + kk);
            FloatVector x0 = av(a, aBase, b0 + kk), x1 = av(a, aBase, b1 + kk), x2 = av(a, aBase, b2 + kk), x3 = av(a, aBase, b3 + kk);
            c00 = w0.fma(x0, c00); c01 = w0.fma(x1, c01); c02 = w0.fma(x2, c02); c03 = w0.fma(x3, c03);
            c10 = w1.fma(x0, c10); c11 = w1.fma(x1, c11); c12 = w1.fma(x2, c12); c13 = w1.fma(x3, c13);
            c20 = w2.fma(x0, c20); c21 = w2.fma(x1, c21); c22 = w2.fma(x2, c22); c23 = w2.fma(x3, c23);
            c30 = w3.fma(x0, c30); c31 = w3.fma(x1, c31); c32 = w3.fma(x2, c32); c33 = w3.fma(x3, c33);
        }
        long o0 = (long) s0 * oStride + row0;
        store(o, oBase, o0, c00.reduceLanes(VectorOperators.ADD)); store(o, oBase, o0 + oStride, c01.reduceLanes(VectorOperators.ADD)); store(o, oBase, o0 + 2L * oStride, c02.reduceLanes(VectorOperators.ADD)); store(o, oBase, o0 + 3L * oStride, c03.reduceLanes(VectorOperators.ADD));
        store(o, oBase, o0 + 1, c10.reduceLanes(VectorOperators.ADD)); store(o, oBase, o0 + oStride + 1, c11.reduceLanes(VectorOperators.ADD)); store(o, oBase, o0 + 2L * oStride + 1, c12.reduceLanes(VectorOperators.ADD)); store(o, oBase, o0 + 3L * oStride + 1, c13.reduceLanes(VectorOperators.ADD));
        store(o, oBase, o0 + 2, c20.reduceLanes(VectorOperators.ADD)); store(o, oBase, o0 + oStride + 2, c21.reduceLanes(VectorOperators.ADD)); store(o, oBase, o0 + 2L * oStride + 2, c22.reduceLanes(VectorOperators.ADD)); store(o, oBase, o0 + 3L * oStride + 2, c23.reduceLanes(VectorOperators.ADD));
        store(o, oBase, o0 + 3, c30.reduceLanes(VectorOperators.ADD)); store(o, oBase, o0 + oStride + 3, c31.reduceLanes(VectorOperators.ADD)); store(o, oBase, o0 + 2L * oStride + 3, c32.reduceLanes(VectorOperators.ADD)); store(o, oBase, o0 + 3L * oStride + 3, c33.reduceLanes(VectorOperators.ADD));
    }

    /** MR=3 rows x NR=3 cols decode-free F32 band: 9 accumulators + 3 weight + 3 activation vectors. */
    static void gemm512Band3x3(float[] w, int dim1, MemorySegment a, long aBase, MemorySegment o, long oBase,
                               int aStride, int oStride, int row0, int s0) {
        int row1 = row0 + 1, row2 = row0 + 2;
        long b0 = (long) s0 * aStride, b1 = b0 + aStride, b2 = b1 + aStride;
        FloatVector c00 = FloatVector.zero(F_SPECIES), c01 = FloatVector.zero(F_SPECIES), c02 = FloatVector.zero(F_SPECIES);
        FloatVector c10 = FloatVector.zero(F_SPECIES), c11 = FloatVector.zero(F_SPECIES), c12 = FloatVector.zero(F_SPECIES);
        FloatVector c20 = FloatVector.zero(F_SPECIES), c21 = FloatVector.zero(F_SPECIES), c22 = FloatVector.zero(F_SPECIES);
        int len = F_SPECIES.length();
        for (int kk = 0; kk < dim1; kk += len) {
            FloatVector w0 = FloatVector.fromArray(F_SPECIES, w, kk);
            FloatVector w1 = FloatVector.fromArray(F_SPECIES, w, dim1 + kk);
            FloatVector w2 = FloatVector.fromArray(F_SPECIES, w, 2 * dim1 + kk);
            FloatVector x0 = av(a, aBase, b0 + kk);
            FloatVector x1 = av(a, aBase, b1 + kk);
            FloatVector x2 = av(a, aBase, b2 + kk);
            c00 = w0.fma(x0, c00); c01 = w0.fma(x1, c01); c02 = w0.fma(x2, c02);
            c10 = w1.fma(x0, c10); c11 = w1.fma(x1, c11); c12 = w1.fma(x2, c12);
            c20 = w2.fma(x0, c20); c21 = w2.fma(x1, c21); c22 = w2.fma(x2, c22);
        }
        store(o, oBase, (long) s0 * oStride + row0, c00.reduceLanes(VectorOperators.ADD));
        store(o, oBase, (long) (s0 + 1) * oStride + row0, c01.reduceLanes(VectorOperators.ADD));
        store(o, oBase, (long) (s0 + 2) * oStride + row0, c02.reduceLanes(VectorOperators.ADD));
        store(o, oBase, (long) s0 * oStride + row1, c10.reduceLanes(VectorOperators.ADD));
        store(o, oBase, (long) (s0 + 1) * oStride + row1, c11.reduceLanes(VectorOperators.ADD));
        store(o, oBase, (long) (s0 + 2) * oStride + row1, c12.reduceLanes(VectorOperators.ADD));
        store(o, oBase, (long) s0 * oStride + row2, c20.reduceLanes(VectorOperators.ADD));
        store(o, oBase, (long) (s0 + 1) * oStride + row2, c21.reduceLanes(VectorOperators.ADD));
        store(o, oBase, (long) (s0 + 2) * oStride + row2, c22.reduceLanes(VectorOperators.ADD));
    }

    /** Flat F32 dot of a dequantized weight row (at {@code w[wOffset..]}) against one activation column. */
    static float dotDeq(float[] w, int wOffset, int dim1, MemorySegment a, long aBase, long xbase) {
        FloatVector acc = FloatVector.zero(F_SPECIES);
        int len = F_SPECIES.length();
        for (int kk = 0; kk < dim1; kk += len) {
            acc = FloatVector.fromArray(F_SPECIES, w, wOffset + kk).fma(av(a, aBase, xbase + kk), acc);
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    private static FloatVector av(MemorySegment a, long aBase, long elem) {
        return FloatVector.fromMemorySegment(F_SPECIES, a, aBase + elem * 4, ByteOrder.LITTLE_ENDIAN);
    }

    /** Scalar F32 store at element index {@code elem} of the output (token-major). */
    static void store(MemorySegment o, long oBase, long elem, float v) {
        o.set(JAVA_FLOAT_UNALIGNED, oBase + elem * 4, v);
    }
}
