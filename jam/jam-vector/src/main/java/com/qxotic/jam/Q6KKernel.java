package com.qxotic.jam;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

import static com.qxotic.jam.VectorSupport.F_SPECIES;
import static com.qxotic.jam.VectorSupport.readByte;
import static com.qxotic.jam.VectorSupport.readFloat16;

/**
 * Q6_K gemm, relocated from jinfer (segment-based). Q6_K super-block: 256 elements / 210 bytes
 * ({@code ql[128] | qh[64] | scales[16] int8 | fp16 d}); 6-bit quants (4 from ql nibble + 2 from qh),
 * value {@code d·sc·(q6−32)}. Dequantizes a {@link BandGemm#MR}-row band into an F32 scratch, then
 * {@link BandGemm} sweeps the columns.
 */
public final class Q6KKernel {

    private Q6KKernel() {}

    static final int BLOCK = 256, TYPE = 210;

    public static void gemm(MemorySegment w, MemorySegment a, long aBase, MemorySegment o, long oBase,
                            int aStride, int oStride, int n, int m, int k, long wOff, Scratch scratch) {
        BandGemm.gemm(w, a, aBase, o, oBase, aStride, oStride, n, m, k, wOff, scratch, Q6KKernel::dequantizeRow);
    }

    /** Dequantize one Q6_K weight row (dim1 % 256 == 0) into {@code dst} at {@code dstOffset}. 512-bit. */
    static void dequantizeRow(MemorySegment w, long rowElemOffset, int dim1, float[] dst, int dstOffset) {
        int kblocks = dim1 / BLOCK;
        long firstBlock = rowElemOffset / BLOCK;
        for (int blk = 0; blk < kblocks; blk++) {
            long b = (firstBlock + blk) * TYPE;
            long qlOff = b, qhOff = b + 128, scOff = b + 192;
            float d = readFloat16(w, b + 208);
            int blockBase = dstOffset + blk * BLOCK;
            for (int h = 0; h < 2; h++) {
                long qlBase = qlOff + h * 64;
                long qhBase = qhOff + h * 32;
                for (int c = 0; c < 2; c++) {
                    var qlA = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, qlBase + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    var qlB = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, qlBase + 32 + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    var qhV = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, qhBase + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    var q0 = qlA.and((byte) 0xF).or(qhV.and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    var q1 = qlB.and((byte) 0xF).or(qhV.lanewise(VectorOperators.LSHR, 2).and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    var q2 = qlA.lanewise(VectorOperators.LSHR, 4).or(qhV.lanewise(VectorOperators.LSHR, 4).and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    var q3 = qlB.lanewise(VectorOperators.LSHR, 4).or(qhV.lanewise(VectorOperators.LSHR, 6).and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    int hb = blockBase + h * 128 + c * 16;
                    VectorSupport.storeScaled(q0, FloatVector.broadcast(F_SPECIES, d * readByte(w, scOff + h * 8 + c)), dst, hb);
                    VectorSupport.storeScaled(q1, FloatVector.broadcast(F_SPECIES, d * readByte(w, scOff + h * 8 + 2 + c)), dst, hb + 32);
                    VectorSupport.storeScaled(q2, FloatVector.broadcast(F_SPECIES, d * readByte(w, scOff + h * 8 + 4 + c)), dst, hb + 64);
                    VectorSupport.storeScaled(q3, FloatVector.broadcast(F_SPECIES, d * readByte(w, scOff + h * 8 + 6 + c)), dst, hb + 96);
                }
            }
        }
    }
}
