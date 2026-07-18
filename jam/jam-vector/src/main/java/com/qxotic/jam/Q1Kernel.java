package com.qxotic.jam;

import static com.qxotic.jam.VectorSupport.F_SPECIES;
import static com.qxotic.jam.VectorSupport.readFloat16;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Q1_0 gemm. Q1_0 block: 128 elements / 18 bytes ({@code f16 d; 16 sign-bit bytes}, LSB-first),
 * value {@code bit ? +d : -d}. A {@link BandGemm#MR}-row band is dequantized into an F32 scratch,
 * then {@link BandGemm}'s F32 band sweeps the columns - the same shape as the other band kernels.
 * The dequant applies the sign bits branchlessly in vector registers (the xor-on-the-float-sign
 * trick from the decode dot): per lane, {@code ((~(bits >>> lane)) & 1) << 31} XORed onto the
 * broadcast f32 bits of {@code d} yields {@code bit ? d : -d} - roughly 16x the scalar per-bit
 * loop, which otherwise dominates the sweep at small n.
 *
 * <p>A fully FUSED 4x4 tile (weight vectors derived in registers inside the k loop, no scratch) was
 * built and benched against this path and LOST decisively at GEMM shapes - 321 vs 429 GMAC/s at 16T
 * n512 (314 vs 534 on GraalVM 25.1-dev) - because the per-chunk weight derivation (6 vector ops x 4
 * rows) re-runs on every column sweep, while the scratch is written once per band and re-read from
 * L1. Do not re-attempt without changing that math.
 */
public final class Q1Kernel {

    private Q1Kernel() {}

    static final int QK = 128, BYTES = 18;

    private static final java.lang.foreign.ValueLayout.OfLong LONG_LE =
            java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED.withOrder(
                    java.nio.ByteOrder.LITTLE_ENDIAN);

    private static final VectorSpecies<Integer> I_SPECIES =
            VectorSpecies.of(int.class, F_SPECIES.vectorShape());
    private static final IntVector LANE_IOTA = IntVector.zero(I_SPECIES).addIndex(1);

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
                Q1Kernel::dequantizeRow);
    }

    /**
     * Dequantize one Q1_0 weight row (dim1 % 128 == 0) into {@code dst[dstOffset..]} in element
     * order. Lane counts of 4/8/16 all divide 64, so a bit chunk never straddles the two longs.
     */
    private static void dequantizeRow(
            MemorySegment w, long rowElemOffset, int dim1, MemorySegment dst, long dstBase) {
        int kblocks = dim1 / QK;
        long firstBlock = rowElemOffset / QK;
        int lanes = F_SPECIES.length();
        for (int blk = 0; blk < kblocks; blk++) {
            long bo = (firstBlock + blk) * BYTES;
            int dBits = Float.floatToRawIntBits(readFloat16(w, bo));
            // MEASURED (see Q1_0FloatTensor.vectorDot): the shared readLong helper compiles
            // slower in these bit-unpack loops - keep the local little-endian layout read.
            long lo = w.get(LONG_LE, bo + 2);
            long hi = w.get(LONG_LE, bo + 10);
            long base = dstBase + (long) blk * QK * 4;
            for (int g = 0; g < QK; g += lanes) {
                long bits = g < 64 ? lo >>> g : hi >>> (g - 64);
                IntVector.broadcast(I_SPECIES, (int) bits)
                        .lanewise(VectorOperators.LSHR, LANE_IOTA)
                        .not()
                        .and(1)
                        .lanewise(VectorOperators.LSHL, 31)
                        .lanewise(VectorOperators.XOR, dBits)
                        .reinterpretAsFloats()
                        .intoMemorySegment(dst, base + g * 4L, ByteOrder.LITTLE_ENDIAN);
            }
        }
    }
}
