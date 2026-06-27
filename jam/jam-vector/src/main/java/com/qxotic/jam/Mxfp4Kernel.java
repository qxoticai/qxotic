package com.qxotic.jam;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

import static com.qxotic.jam.VectorSupport.F_SPECIES;
import static com.qxotic.jam.VectorSupport.readByte;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;

/**
 * MXFP4 register-tiled gemm, relocated from jinfer (segment-based). MXFP4 block: 32 elements / 17 bytes
 * ({@code e8m0 scale; 16 nibble bytes}), value {@code lut[nibble]·e8m0Half(scale)}. A group of 3 output
 * rows is dequantized once into an F32 scratch, then a decode-free 3x3 F32 band sweeps the columns.
 * Identical to jinfer's {@code MXFP4FloatTensor.vectorGemmMxfp4}. The shared band machinery
 * ({@link #bandScratch}, {@link #gemm512Band3x3}, {@link #dotDeq}) is reused by {@link Nvfp4Kernel}.
 */
public final class Mxfp4Kernel {

    private Mxfp4Kernel() {}

    static final int QK = 32, BYTES = 17, MR = 3, NR = 3;

    private static final byte[] MXFP4_LUT = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

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

    public static void gemm(MemorySegment w, MemorySegment a, long aBase, MemorySegment o, long oBase,
                            int aStride, int oStride, int n, int m, int k, long wOff) {
        int groups = m / MR;
        VectorSupport.parallelFor(0, groups, g -> {
            int row0 = g * MR;
            float[] band = bandScratch(MR * k);
            for (int i = 0; i < MR; i++) {
                dequantizeRow(w, wOff + (long) (row0 + i) * k, k, band, i * k);
            }
            int s = 0;
            for (; s + NR <= n; s += NR) {
                gemm512Band3x3(band, k, a, aBase, o, oBase, aStride, oStride, row0, s);
            }
            for (; s < n; s++) {
                for (int i = 0; i < MR; i++) {
                    store(o, oBase, (long) s * oStride + row0 + i, dotDeq(band, i * k, k, a, aBase, (long) s * aStride));
                }
            }
        });
        for (int row = groups * MR; row < m; row++) {  // trailing rows: cheap per-column dots
            float[] band = bandScratch(k);
            dequantizeRow(w, wOff + (long) row * k, k, band, 0);
            float[] bf = band;
            int rr = row;
            VectorSupport.parallelFor(0, n, s -> store(o, oBase, (long) s * oStride + rr, dotDeq(bf, 0, k, a, aBase, (long) s * aStride)));
        }
    }

    /** Dequantize one MXFP4 weight row (dim1 % 32 == 0) into {@code dst} at {@code dstOffset}. 512-bit. */
    private static void dequantizeRow(MemorySegment w, long rowElemOffset, int dim1, float[] dst, int dstOffset) {
        int kblocks = dim1 / QK;
        long firstBlock = rowElemOffset / QK;
        for (int blk = 0; blk < kblocks; blk++) {
            long blockOffset = (firstBlock + blk) * BYTES;
            float d = e8m0ToFp32Half(Byte.toUnsignedInt(readByte(w, blockOffset)));
            ByteVector packed = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, blockOffset + 1, ByteOrder.LITTLE_ENDIAN);
            FloatVector loC = ((FloatVector) mxfp4Decode(packed.and((byte) 0x0F)).castShape(F_SPECIES, 0)).mul(d);
            FloatVector hiC = ((FloatVector) mxfp4Decode(packed.lanewise(VectorOperators.LSHR, 4)).castShape(F_SPECIES, 0)).mul(d);
            int base = dstOffset + blk * QK;
            loC.intoArray(dst, base);                  // block elems 0..15 (low nibbles)
            hiC.intoArray(dst, base + QK / 2);         // block elems 16..31 (high nibbles)
        }
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

    /** Decode 16 nibble codes to signed MXFP4 values via one in-register table permute (vpshufb). */
    private static ByteVector mxfp4Decode(ByteVector nibbles) {
        return ByteVector.fromArray(ByteVector.SPECIES_128, MXFP4_LUT, 0).rearrange(nibbles.toShuffle());
    }

    private static float e8m0ToFp32Half(int x) {
        int bits;
        if (x < 2) {
            bits = 0x00200000 << x;
        } else {
            bits = (x - 1) << 23;
        }
        return Float.intBitsToFloat(bits);
    }

    private static FloatVector av(MemorySegment a, long aBase, long elem) {
        return FloatVector.fromMemorySegment(F_SPECIES, a, aBase + elem * 4, ByteOrder.LITTLE_ENDIAN);
    }

    private static void store(MemorySegment o, long oBase, long elem, float v) {
        o.set(JAVA_FLOAT_UNALIGNED, oBase + elem * 4, v);
    }
}
