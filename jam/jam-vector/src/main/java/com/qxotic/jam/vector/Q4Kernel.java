package com.qxotic.jam.vector;

import static com.qxotic.jam.vector.VectorSupport.F_SPECIES;
import static com.qxotic.jam.vector.VectorSupport.readFloat16;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

/**
 * Q4_0 gemm. Q4_0 block: 32 elements / 18 bytes ({@code fp16 d; nibble qs[16]}), value {@code
 * d·(q-8)}. A {@link BandGemm#MR}-row band is dequantized into an F32 scratch and {@link
 * BandGemm}'s F32 band sweeps the columns, like every other 32-block quant here, at every vector
 * width: each row decodes once, not once per column (LFM2.5-2.6B pp512 in a 256-bit native image,
 * Zen 5 16T: 49 -> 182 t/s over the per-column dot this replaced).
 */
final class Q4Kernel {

    private Q4Kernel() {}

    private static final int BLOCK = 32, TYPE = 18;

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
                Q4Kernel::dequantizeRow);
    }

    /**
     * Dequantize one Q4_0 weight row ({@code dim1 % 32 == 0}, block-aligned {@code rowElemOffset})
     * into {@code dst} at {@code dstBase}: per 32-elem block the 16 nibble bytes fan out as
     * (lo-8)*d into elements [0,16) and (hi-8)*d into [16,32).
     */
    static void dequantizeRow(
            MemorySegment w, long rowElemOffset, int dim1, MemorySegment dst, long dstBase) {
        // weights through the absolute-address route (see Q8Kernel.dequantizeRow: a vector load
        // straight off the mapped segment miscompiles in the native image)
        final MemorySegment ws = VectorSupport.vectorSegment(w);
        long blkOff = VectorSupport.vectorBase(w) + rowElemOffset / BLOCK * TYPE;
        long d = dstBase;
        for (int j = 0; j < dim1; j += BLOCK, blkOff += TYPE, d += (long) BLOCK * 4) {
            var vd = FloatVector.broadcast(F_SPECIES, readFloat16(ws, blkOff));
            var bytes =
                    ByteVector.fromMemorySegment(
                            ByteVector.SPECIES_128, ws, blkOff + 2, ByteOrder.LITTLE_ENDIAN);
            var lo = bytes.and((byte) 0xF).sub((byte) 8);
            var hi = bytes.lanewise(VectorOperators.LSHR, 4).sub((byte) 8);
            VectorSupport.storeScaled(lo, vd, dst, d);
            VectorSupport.storeScaled(hi, vd, dst, d + 64);
        }
    }
}
