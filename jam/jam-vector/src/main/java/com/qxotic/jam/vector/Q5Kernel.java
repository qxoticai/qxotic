package com.qxotic.jam.vector;

import static com.qxotic.jam.vector.VectorSupport.F_SPECIES;
import static com.qxotic.jam.vector.VectorSupport.readFloat16;

import com.oracle.svm.shared.AlwaysInline;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorOperators;

/**
 * Q5_0 gemm. Q5_0 block: 32 elements / 22 bytes ({@code fp16 d; u32 qh; nibble qs[16]}), value
 * {@code d·(q-16)} with {@code q} the nibble plus one high bit from {@code qh} (bit {@code j} for
 * element {@code j}, bit {@code j+16} for element {@code j+16}). A {@link BandGemm#MR}-row band is
 * dequantized into an F32 scratch and {@link BandGemm}'s F32 band sweeps the columns, like every
 * other 32-block quant here. The high bits are spread to byte lanes in registers: one int
 * broadcast, a per-lane shift, mask, shift into bit 4.
 */
final class Q5Kernel {

    private Q5Kernel() {}

    static final int BLOCK = 32, BYTES = 22;

    private static final ValueLayout.OfInt INT_LE =
            ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

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
                Q5Kernel::dequantizeRow);
    }

    /**
     * Dequantize one Q5_0 weight row ({@code dim1 % 32 == 0}, block-aligned {@code rowElemOffset})
     * into {@code dst} at {@code dstBase}: per block, elements [0,16) from the low nibbles and bits
     * 0..15 of {@code qh}, [16,32) from the high nibbles and bits 16..31.
     */
    static void dequantizeRow(
            MemorySegment w, long rowElemOffset, int dim1, MemorySegment dst, long dstBase) {
        final MemorySegment ws = VectorSupport.vectorSegment(w);
        long blkOff = VectorSupport.vectorBase(w) + rowElemOffset / BLOCK * BYTES;
        long d = dstBase;
        for (int j = 0; j < dim1; j += BLOCK, blkOff += BYTES, d += (long) BLOCK * 4) {
            var vd = FloatVector.broadcast(F_SPECIES, readFloat16(ws, blkOff));
            int qh = ws.get(INT_LE, blkOff + 2);
            var bytes =
                    ByteVector.fromMemorySegment(
                            ByteVector.SPECIES_128, ws, blkOff + 6, ByteOrder.LITTLE_ENDIAN);
            var lo = bytes.and((byte) 0xF).or(highBits(qh)).sub((byte) 16);
            var hi = bytes.lanewise(VectorOperators.LSHR, 4).or(highBits(qh >>> 16)).sub((byte) 16);
            VectorSupport.storeScaled(lo, vd, dst, d);
            VectorSupport.storeScaled(hi, vd, dst, d + 64);
        }
    }

    /**
     * The low 16 bits of {@code bits} as 16 byte lanes holding {@code 0x10} where the bit is set
     * (lane {@code i} = bit {@code i}): each byte of two longs is spread from one bit by a
     * multiply-and-mask, then normalized to 0x10 through the carry into bit 7. Width-agnostic, no
     * wide-int species to emulate on 256-bit hosts.
     */
    @AlwaysInline("hot Vector API leaf: an escaping ByteVector is materialized per call")
    private static ByteVector highBits(int bits) {
        return LongVector.zero(LongVector.SPECIES_128)
                .withLane(0, spread(bits))
                .withLane(1, spread(bits >>> 8))
                .reinterpretAsBytes();
    }

    private static long spread(int bits8) {
        long m = ((long) (bits8 & 0xFF) * 0x0101010101010101L) & 0x8040201008040201L;
        return ((m + 0x7F7F7F7F7F7F7F7FL) & 0x8080808080808080L) >>> 3;
    }
}
