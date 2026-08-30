package com.qxotic.jam.vector;

import static com.qxotic.jam.vector.VectorSupport.F_SPECIES;
import static com.qxotic.jam.vector.VectorSupport.readFloat16;

import com.oracle.svm.shared.AlwaysInline;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Q5_K gemm, relocated from jinfer (segment-based). Q5_K super-block: 256 elements / 176 bytes
 * ({@code fp16 d, dmin; 12 scale bytes; 32 qh bytes (5th bit); 128 nibble bytes}); value {@code
 * d·sc·quant − dmin·m} with {@code quant = nibble | (qhBit<<4)}. Dequantizes a {@link
 * BandGemm#MR}-row band into an F32 scratch, then {@link BandGemm} sweeps the columns. Reuses
 * {@link Q4KKernel#getScaleMinK4}.
 */
public final class Q5KKernel {

    private Q5KKernel() {}

    static final int BLOCK = 256, TYPE = 176;

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
                Q5KKernel::dequantizeRow);
    }

    /** Dequantize one Q5_K weight row run (block-aligned) into {@code dst} at {@code dstBase}. */
    static void dequantizeRow(
            MemorySegment w, long rowElemOffset, int dim1, MemorySegment dst, long dstBase) {
        int kblocks = dim1 / BLOCK;
        long firstBlock = rowElemOffset / BLOCK;
        final MemorySegment ws = VectorSupport.vectorSegment(w);
        final long wb = VectorSupport.vectorBase(w);
        for (int blk = 0; blk < kblocks; blk++) {
            long b = (firstBlock + blk) * TYPE;
            float d = readFloat16(w, b);
            float dmin = readFloat16(w, b + 2);
            long sc = Q4KKernel.packedScales(w, b + 4);
            long mn = Q4KKernel.packedMins(w, b + 4);
            long qh = wb + b + 16, qs = wb + b + 48;
            long o = dstBase + (long) blk * BLOCK * 4;
            // Sub-block 2g is the low nibbles of bytes [32g, 32g+32) with bit 2g of qh, 2g+1 the
            // high nibbles with bit 2g+1. Every shift count is a literal at its call site, the
            // vectors are loaded fresh per call: no loop-carried vector, nothing for the JIT to
            // lose on a phi (the rotating-qh form of this loop compiled to the boxed fallback on
            // some runs, halving prefill).
            group(ws, qs, 0, qh, 0, d * (int) (sc & 0xFF), dmin * (int) (mn & 0xFF), dst, o);
            group(
                    ws,
                    qs,
                    4,
                    qh,
                    1,
                    d * (int) ((sc >>> 8) & 0xFF),
                    dmin * (int) ((mn >>> 8) & 0xFF),
                    dst,
                    o + 32 * 4L);
            group(
                    ws,
                    qs + 32,
                    0,
                    qh,
                    2,
                    d * (int) ((sc >>> 16) & 0xFF),
                    dmin * (int) ((mn >>> 16) & 0xFF),
                    dst,
                    o + 64 * 4L);
            group(
                    ws,
                    qs + 32,
                    4,
                    qh,
                    3,
                    d * (int) ((sc >>> 24) & 0xFF),
                    dmin * (int) ((mn >>> 24) & 0xFF),
                    dst,
                    o + 96 * 4L);
            group(
                    ws,
                    qs + 64,
                    0,
                    qh,
                    4,
                    d * (int) ((sc >>> 32) & 0xFF),
                    dmin * (int) ((mn >>> 32) & 0xFF),
                    dst,
                    o + 128 * 4L);
            group(
                    ws,
                    qs + 64,
                    4,
                    qh,
                    5,
                    d * (int) ((sc >>> 40) & 0xFF),
                    dmin * (int) ((mn >>> 40) & 0xFF),
                    dst,
                    o + 160 * 4L);
            group(
                    ws,
                    qs + 96,
                    0,
                    qh,
                    6,
                    d * (int) ((sc >>> 48) & 0xFF),
                    dmin * (int) ((mn >>> 48) & 0xFF),
                    dst,
                    o + 192 * 4L);
            group(
                    ws,
                    qs + 96,
                    4,
                    qh,
                    7,
                    d * (int) ((sc >>> 56) & 0xFF),
                    dmin * (int) ((mn >>> 56) & 0xFF),
                    dst,
                    o + 224 * 4L);
        }
    }

    private static final VectorSpecies<Integer> I_SPECIES =
            VectorSpecies.of(int.class, F_SPECIES.vectorShape());

    /**
     * 32 elements: nibble {@code qsShift} (0 = low, 4 = high) of the 32 bytes at {@code qs}, plus
     * bit {@code qhShift} of the 32 bytes at {@code qh} as bit 4, times {@code scale} minus {@code
     * min}. The shift counts are constants at every call site.
     */
    @AlwaysInline("the shift counts are literals at the call sites; the image must see them")
    private static void group(
            MemorySegment ws,
            long qs,
            int qsShift,
            long qh,
            int qhShift,
            float scale,
            float min,
            MemorySegment dst,
            long o) {
        FloatVector vs = FloatVector.broadcast(F_SPECIES, scale);
        FloatVector vm = FloatVector.broadcast(F_SPECIES, -min);
        for (int c = 0; c < 2; c++) {
            var qsB =
                    ByteVector.fromMemorySegment(
                            ByteVector.SPECIES_128, ws, qs + c * 16, ByteOrder.LITTLE_ENDIAN);
            var qhB =
                    ByteVector.fromMemorySegment(
                            ByteVector.SPECIES_128, ws, qh + c * 16, ByteOrder.LITTLE_ENDIAN);
            for (int p = 0; p < VectorSupport.DECODE_PARTS; p++) {
                IntVector lo =
                        ((IntVector) qsB.castShape(I_SPECIES, p))
                                .lanewise(VectorOperators.LSHR, qsShift)
                                .and(0xF);
                IntVector hi =
                        ((IntVector) qhB.castShape(I_SPECIES, p))
                                .lanewise(VectorOperators.LSHR, qhShift)
                                .and(1);
                IntVector q = lo.or(hi.lanewise(VectorOperators.LSHL, 4));
                ((FloatVector) q.castShape(F_SPECIES, 0))
                        .fma(vs, vm)
                        .intoMemorySegment(
                                dst,
                                o + ((long) c * 16 + (long) p * VectorSupport.F_LEN) * 4,
                                ByteOrder.LITTLE_ENDIAN);
            }
        }
    }
}
