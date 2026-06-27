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

    /** Register-tile of the band: MR weight rows x NR activation columns (9 F32 accumulators on 512-bit). */
    static final int MR = 3, NR = 3;

    /** Per-worker F32 scratch holding the row group's dequantized weights; grown on demand, reused. */
    private static final ThreadLocal<float[]> DEQUANT_BAND = new ThreadLocal<>();

    static float[] bandScratch(int need) {
        float[] w = DEQUANT_BAND.get();
        if (w == null || w.length < need) {
            w = new float[need];
            DEQUANT_BAND.set(w);
        }
        return w;
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
