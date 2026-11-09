package com.qxotic.jam.vector;

import static com.qxotic.jam.vector.VectorSupport.F_SPECIES;
import static com.qxotic.jam.vector.VectorSupport.readFloat16;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Q2_0 and PQ2_0 gemm. A block is {@code f16 d} then 2-bit codes, LSB-first, value {@code (code -
 * 1) * d}: 64 elements / 18 bytes for Q2_0, 128 / 34 for PQ2_0. Same shape as {@link Q1Kernel}: a
 * {@link BandGemm#MR}-row band is dequantized into an F32 scratch with the codes unpacked in vector
 * registers, then {@link BandGemm}'s F32 band sweeps the columns.
 */
final class Q2Kernel {

    private Q2Kernel() {}

    private static final ValueLayout.OfLong LONG_LE =
            ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private static final VectorSpecies<Integer> I_SPECIES =
            VectorSpecies.of(int.class, F_SPECIES.vectorShape());
    private static final IntVector LANE_SHIFT = IntVector.zero(I_SPECIES).addIndex(2);

    /** {@code qk}: 64 (Q2_0) or 128 (PQ2_0). */
    public static void gemm(
            int qk,
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
                qk == 64 ? Q2Kernel::dequantizeQ2_0 : Q2Kernel::dequantizePQ2_0);
    }

    private static void dequantizeQ2_0(
            MemorySegment w, long rowElemOffset, int dim1, MemorySegment dst, long dstBase) {
        dequantizeRow(64, w, rowElemOffset, dim1, dst, dstBase);
    }

    private static void dequantizePQ2_0(
            MemorySegment w, long rowElemOffset, int dim1, MemorySegment dst, long dstBase) {
        dequantizeRow(128, w, rowElemOffset, dim1, dst, dstBase);
    }

    /**
     * Dequantize one weight row (dim1 % qk == 0) into {@code dst[dstOffset..]} in element order.
     * Lane counts of 4/8/16 take 8/16/32 code bits, so a chunk never straddles a long.
     */
    private static void dequantizeRow(
            int qk,
            MemorySegment w,
            long rowElemOffset,
            int dim1,
            MemorySegment dst,
            long dstBase) {
        int bytes = 2 + qk / 4, kblocks = dim1 / qk;
        long firstBlock = rowElemOffset / qk;
        int lanes = F_SPECIES.length();
        for (int blk = 0; blk < kblocks; blk++) {
            long bo = (firstBlock + blk) * bytes;
            float d = readFloat16(w, bo);
            long base = dstBase + (long) blk * qk * 4;
            for (int g = 0; g < qk; g += lanes) {
                long bits = w.get(LONG_LE, bo + 2 + g / 32 * 8) >>> (2 * g % 64);
                IntVector.broadcast(I_SPECIES, (int) bits)
                        .lanewise(VectorOperators.LSHR, LANE_SHIFT)
                        .and(3)
                        .sub(1)
                        .convert(VectorOperators.I2F, 0)
                        .reinterpretAsFloats()
                        .mul(d)
                        .intoMemorySegment(dst, base + g * 4L, ByteOrder.LITTLE_ENDIAN);
            }
        }
    }
}
