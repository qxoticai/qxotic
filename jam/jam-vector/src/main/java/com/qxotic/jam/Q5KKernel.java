package com.qxotic.jam;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

import static com.qxotic.jam.Q4KKernel.getScaleMinK4;
import static com.qxotic.jam.VectorSupport.F_SPECIES;
import static com.qxotic.jam.VectorSupport.readFloat16;

/**
 * Q5_K gemm, relocated from jinfer (segment-based). Q5_K super-block: 256 elements / 176 bytes
 * ({@code fp16 d, dmin; 12 scale bytes; 32 qh bytes (5th bit); 128 nibble bytes}); value
 * {@code d·sc·quant − dmin·m} with {@code quant = nibble | (qhBit<<4)}. Dequantizes a {@link BandGemm#MR}-row
 * band into an F32 scratch, then {@link BandGemm} sweeps the columns. Reuses {@link Q4KKernel#getScaleMinK4}.
 */
public final class Q5KKernel {

    private Q5KKernel() {}

    static final int BLOCK = 256, TYPE = 176;

    public static void gemm(MemorySegment w, MemorySegment a, long aBase, MemorySegment o, long oBase,
                            int aStride, int oStride, int n, int m, int k, long wOff) {
        int groups = m / BandGemm.MR;
        VectorSupport.parallelFor(0, groups, g -> {
            int row0 = g * BandGemm.MR;
            float[] band = BandGemm.bandScratch(BandGemm.MR * k);
            for (int i = 0; i < BandGemm.MR; i++) dequantizeRow(w, wOff + (long) (row0 + i) * k, k, band, i * k);
            int s = 0;
            for (; s + BandGemm.NR <= n; s += BandGemm.NR) BandGemm.gemm512Band3x3(band, k, a, aBase, o, oBase, aStride, oStride, row0, s);
            for (; s < n; s++) for (int i = 0; i < BandGemm.MR; i++)
                BandGemm.store(o, oBase, (long) s * oStride + row0 + i, BandGemm.dotDeq(band, i * k, k, a, aBase, (long) s * aStride));
        });
        for (int row = groups * BandGemm.MR; row < m; row++) {
            float[] band = BandGemm.bandScratch(k);
            dequantizeRow(w, wOff + (long) row * k, k, band, 0);
            float[] bf = band; int rr = row;
            VectorSupport.parallelFor(0, n, s -> BandGemm.store(o, oBase, (long) s * oStride + rr, BandGemm.dotDeq(bf, 0, k, a, aBase, (long) s * aStride)));
        }
    }

    /** Dequantize one Q5_K weight row (dim1 % 256 == 0) into {@code dst} at {@code dstOffset}. 512-bit. */
    static void dequantizeRow(MemorySegment w, long rowElemOffset, int dim1, float[] dst, int dstOffset) {
        int kblocks = dim1 / BLOCK;
        long firstBlock = rowElemOffset / BLOCK;
        for (int blk = 0; blk < kblocks; blk++) {
            long b = (firstBlock + blk) * TYPE;
            float d = readFloat16(w, b);
            float dmin = readFloat16(w, b + 2);
            var qh0 = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b + 16, ByteOrder.LITTLE_ENDIAN);
            var qh1 = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b + 32, ByteOrder.LITTLE_ENDIAN);
            int blockBase = dstOffset + blk * BLOCK;
            for (int g = 0; g < 4; g++) {
                var vd0 = FloatVector.broadcast(F_SPECIES, d * getScaleMinK4(g * 2, w, b + 4, false));
                var vm0 = FloatVector.broadcast(F_SPECIES, -(dmin * getScaleMinK4(g * 2, w, b + 4, true)));
                var vd1 = FloatVector.broadcast(F_SPECIES, d * getScaleMinK4(g * 2 + 1, w, b + 4, false));
                var vm1 = FloatVector.broadcast(F_SPECIES, -(dmin * getScaleMinK4(g * 2 + 1, w, b + 4, true)));
                int bitLo = 2 * g, bitHi = 2 * g + 1;
                for (int c = 0; c < 2; c++) {
                    var wb = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b + 48 + (long) g * 32 + c * 16, ByteOrder.LITTLE_ENDIAN);
                    var qhb = (c == 0) ? qh0 : qh1;
                    var loB = wb.and((byte) 0xF).or(qhb.lanewise(VectorOperators.LSHR, bitLo).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
                    var hiB = wb.lanewise(VectorOperators.LSHR, 4).or(qhb.lanewise(VectorOperators.LSHR, bitHi).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
                    ((FloatVector) loB.castShape(F_SPECIES, 0)).fma(vd0, vm0).intoArray(dst, blockBase + g * 64 + c * 16);
                    ((FloatVector) hiB.castShape(F_SPECIES, 0)).fma(vd1, vm1).intoArray(dst, blockBase + g * 64 + 32 + c * 16);
                }
            }
        }
    }
}
