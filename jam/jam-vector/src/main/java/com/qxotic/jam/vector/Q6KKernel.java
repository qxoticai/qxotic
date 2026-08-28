package com.qxotic.jam.vector;

import static com.qxotic.jam.vector.VectorSupport.F_SPECIES;
import static com.qxotic.jam.vector.VectorSupport.readByte;
import static com.qxotic.jam.vector.VectorSupport.readFloat16;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Q6_K gemm, relocated from jinfer (segment-based). Q6_K super-block: 256 elements / 210 bytes
 * ({@code ql[128] | qh[64] | scales[16] int8 | fp16 d}); 6-bit quants (4 from ql nibble + 2 from
 * qh), value {@code d·sc·(q6−32)}. Dequantizes a {@link BandGemm#MR}-row band into an F32 scratch,
 * then {@link BandGemm} sweeps the columns.
 */
public final class Q6KKernel {

    private Q6KKernel() {}

    static final int BLOCK = 256, TYPE = 210;

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
                Q6KKernel::dequantizeRow);
    }

    /** Dequantize one Q6_K weight row run (block-aligned) into {@code dst} at {@code dstBase}. */
    static void dequantizeRow(
            MemorySegment w, long rowElemOffset, int dim1, MemorySegment dst, long dstBase) {
        int kblocks = dim1 / BLOCK;
        long firstBlock = rowElemOffset / BLOCK;
        final MemorySegment ws = VectorSupport.vectorSegment(w);
        final long wb = VectorSupport.vectorBase(w);
        for (int blk = 0; blk < kblocks; blk++) {
            long b = (firstBlock + blk) * TYPE;
            float d = readFloat16(w, b + 208);
            long sc = b + 192;
            long o = dstBase + (long) blk * BLOCK * 4;
            // 16 groups of 16 elements. The 6-bit unpack runs in 32-bit lanes: a byte shift
            // costs C2 a 5-uop widen/shift/mask/pack sequence, an int shift one uop (measured
            // 4.9 vs 16 Gelem/s decode).
            for (int h = 0; h < 2; h++) {
                long ql = wb + b + h * 64, qh = wb + b + 128 + h * 32, s8 = sc + h * 8;
                long oh = o + h * 128 * 4L;
                for (int c = 0; c < 2; c++) {
                    long qlc = ql + c * 16, qhc = qh + c * 16, oc = oh + c * 16 * 4L;
                    group(ws, qlc, 0, qhc, 0, d * readByte(w, s8 + c), dst, oc);
                    group(ws, qlc + 32, 0, qhc, 2, d * readByte(w, s8 + 2 + c), dst, oc + 32 * 4L);
                    group(ws, qlc, 4, qhc, 4, d * readByte(w, s8 + 4 + c), dst, oc + 64 * 4L);
                    group(ws, qlc + 32, 4, qhc, 6, d * readByte(w, s8 + 6 + c), dst, oc + 96 * 4L);
                }
            }
        }
    }

    private static final VectorSpecies<Integer> I_SPECIES =
            VectorSpecies.of(int.class, F_SPECIES.vectorShape());

    /**
     * 16 elements: nibble {@code qlShift} (0 = low, 4 = high) of the bytes at {@code ql}, plus bits
     * {@code qhShift}..+1 of the bytes at {@code qh} as bits 4-5, minus 32, times {@code scale}.
     * The shift counts are constants at every call site. Width-generic: the 16-byte chunk is {@link
     * VectorSupport#DECODE_PARTS} int vectors (1 at 512-bit, 2 at 256, 4 at 128).
     */
    private static void group(
            MemorySegment ws,
            long ql,
            int qlShift,
            long qh,
            int qhShift,
            float scale,
            MemorySegment dst,
            long o) {
        var qlB =
                ByteVector.fromMemorySegment(
                        ByteVector.SPECIES_128, ws, ql, ByteOrder.LITTLE_ENDIAN);
        var qhB =
                ByteVector.fromMemorySegment(
                        ByteVector.SPECIES_128, ws, qh, ByteOrder.LITTLE_ENDIAN);
        FloatVector vs = FloatVector.broadcast(F_SPECIES, scale);
        for (int p = 0; p < VectorSupport.DECODE_PARTS; p++) {
            IntVector lo =
                    ((IntVector) qlB.castShape(I_SPECIES, p))
                            .lanewise(VectorOperators.LSHR, qlShift)
                            .and(0xF);
            IntVector hi =
                    ((IntVector) qhB.castShape(I_SPECIES, p))
                            .lanewise(VectorOperators.LSHR, qhShift)
                            .and(3);
            IntVector q = lo.or(hi.lanewise(VectorOperators.LSHL, 4)).sub(32);
            ((FloatVector) q.castShape(F_SPECIES, 0))
                    .mul(vs)
                    .intoMemorySegment(
                            dst, o + (long) p * VectorSupport.F_LEN * 4, ByteOrder.LITTLE_ENDIAN);
        }
    }
}
