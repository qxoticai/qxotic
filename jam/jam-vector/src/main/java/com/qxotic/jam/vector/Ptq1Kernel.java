package com.qxotic.jam.vector;

import static com.qxotic.jam.vector.VectorSupport.readFloat16;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * PTQ1_0 gemm. PTQ1_0 block: 128 elements / 28 bytes ({@code qs[24]; qh[2]; f16 d}), TQ1_0's base-3
 * packing at a 128-element block: element {@code n * c + m} of a stage of {@code c} bytes is trit
 * {@code n} of byte {@code m}, the trit {@code ((b * 3^n) & 255) * 3 >> 8}, value {@code (trit - 1)
 * * d}. Stages: 16 and 8 bytes of 5 trits (qs), then 2 bytes of 4 (qh). A {@link BandGemm#MR}-row
 * band is dequantized into an F32 scratch, then {@link BandGemm}'s F32 band sweeps the columns.
 */
final class Ptq1Kernel {

    private Ptq1Kernel() {}

    static final int QK = 128, BYTES = 28, QH = 24, D = 26; // qh and the f16 scale follow qs

    private static final ValueLayout.OfFloat FLOAT = ValueLayout.JAVA_FLOAT_UNALIGNED;

    public static void gemm(
            MemorySegment w,
            MemorySegment a,
            long aBase,
            MemorySegment o,
            long oBase,
            int aStride,
            int oStride,
            int n,
            int m,
            int k,
            long wOff,
            Scratch scratch) {
        BandGemm.gemm(
                w,
                a,
                aBase,
                o,
                oBase,
                aStride,
                oStride,
                n,
                m,
                k,
                wOff,
                scratch,
                Ptq1Kernel::dequantizeRow);
    }

    // ponytail: scalar trit unpack, amortized over every column of the band; vectorize if a
    // short-prompt profile shows it.
    private static void dequantizeRow(
            MemorySegment w, long rowElemOffset, int dim1, MemorySegment dst, long dstBase) {
        long firstBlock = rowElemOffset / QK;
        for (int blk = 0; blk < dim1 / QK; blk++) {
            long bo = (firstBlock + blk) * BYTES;
            float d = readFloat16(w, bo + D);
            long base = dstBase + (long) blk * QK * 4;
            trits(w, bo, 16, 5, d, dst, base);
            trits(w, bo + 16, 8, 5, d, dst, base + 80 * 4);
            trits(w, bo + QH, 2, 4, d, dst, base + 120 * 4);
        }
    }

    private static void trits(
            MemorySegment w, long p, int c, int digits, float d, MemorySegment dst, long base) {
        for (int m = 0; m < c; m++) {
            int b = Byte.toUnsignedInt(w.get(ValueLayout.JAVA_BYTE, p + m));
            for (int n = 0, pow = 1; n < digits; n++, pow *= 3)
                dst.set(FLOAT, base + 4L * (n * c + m), (((b * pow & 0xFF) * 3 >> 8) - 1) * d);
        }
    }
}
