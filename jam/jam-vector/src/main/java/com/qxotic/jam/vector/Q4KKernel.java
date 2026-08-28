package com.qxotic.jam.vector;

import static com.qxotic.jam.vector.VectorSupport.F_SPECIES;
import static com.qxotic.jam.vector.VectorSupport.readByte;
import static com.qxotic.jam.vector.VectorSupport.readFloat16;
import static com.qxotic.jam.vector.VectorSupport.readInt;
import static com.qxotic.jam.vector.VectorSupport.readLong;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Q4_K gemm, relocated from jinfer (segment-based). Q4_K super-block: 256 elements / 144 bytes
 * ({@code fp16 d, dmin; 12 packed scale/min bytes; 128 nibble bytes}); value {@code d·sc·nibble −
 * dmin·m}. Dequantizes a {@link BandGemm#MR}-row band into an F32 scratch, then {@link BandGemm}
 * sweeps the columns - so the 6-bit super-block scale unpack is amortized once per row, not per
 * column tile.
 */
public final class Q4KKernel {

    private Q4KKernel() {}

    static final int BLOCK = 256, TYPE = 144;

    // ---- shared k-quant scale unpack (Q5_K reuses these) ----

    /** Decode scale or min for sub-block j (0..7) from the 12-byte scales array. */
    static int getScaleMinK4(int j, MemorySegment mem, long scalesOffset, boolean isMin) {
        if (j < 4) {
            int idx = isMin ? j + 4 : j;
            return Byte.toUnsignedInt(readByte(mem, scalesOffset + idx)) & 63;
        } else {
            int lowIdx = j + 4;
            int highIdx = isMin ? j : j - 4;
            int low =
                    isMin
                            ? (Byte.toUnsignedInt(readByte(mem, scalesOffset + lowIdx)) >> 4)
                            : (Byte.toUnsignedInt(readByte(mem, scalesOffset + lowIdx)) & 0xF);
            int high = (Byte.toUnsignedInt(readByte(mem, scalesOffset + highIdx)) >> 6) & 0x3;
            return low | (high << 4);
        }
    }

    /**
     * The 8 sub-block scales unpacked branch-free into one byte-per-value long (LSB = sub-block 0).
     */
    static long packedScales(MemorySegment w, long scalesOff) {
        long lo = readLong(w, scalesOff);
        int hi = readInt(w, scalesOff + 8);
        long packed = 0;
        for (int j = 0; j < 4; j++) {
            packed |= ((lo >>> (8 * j)) & 63) << (8 * j);
            long v = ((hi >>> (8 * j)) & 0xF) | (((lo >>> (8 * j + 6)) & 3) << 4);
            packed |= v << (8 * (j + 4));
        }
        return packed;
    }

    /** The 8 sub-block mins, same packing as {@link #packedScales}. */
    static long packedMins(MemorySegment w, long scalesOff) {
        long lo = readLong(w, scalesOff);
        int hi = readInt(w, scalesOff + 8);
        long packed = 0;
        for (int j = 0; j < 4; j++) {
            packed |= ((lo >>> (8 * (j + 4))) & 63) << (8 * j);
            long v = ((hi >>> (8 * j + 4)) & 0xF) | (((lo >>> (8 * (j + 4) + 6)) & 3) << 4);
            packed |= v << (8 * (j + 4));
        }
        return packed;
    }

    // ---- gemm: dequantize the row-band once, then the shared decode-free F32 band ----

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
                Q4KKernel::dequantizeRow);
    }

    /** Dequantize one Q4_K weight row run (block-aligned) into {@code dst} at {@code dstBase}. */
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
            long packedSc = packedScales(w, b + 4);
            long packedMn = packedMins(w, b + 4);
            long o = dstBase + (long) blk * BLOCK * 4;
            for (int g = 0; g < 4; g++) {
                float sc0 = d * (int) ((packedSc >>> (16 * g)) & 0xFF);
                float mn0 = -(dmin * (int) ((packedMn >>> (16 * g)) & 0xFF));
                float sc1 = d * (int) ((packedSc >>> (16 * g + 8)) & 0xFF);
                float mn1 = -(dmin * (int) ((packedMn >>> (16 * g + 8)) & 0xFF));
                long qs = wb + b + 16 + g * 32, og = o + g * 64 * 4L;
                // per 16-byte chunk: low nibbles -> 16 elements, high nibbles -> 16 elements at +32
                pair(ws, qs, sc0, mn0, sc1, mn1, dst, og);
                pair(ws, qs + 16, sc0, mn0, sc1, mn1, dst, og + 16 * 4L);
            }
        }
    }

    private static final VectorSpecies<Integer> I_SPECIES =
            VectorSpecies.of(int.class, F_SPECIES.vectorShape());

    /**
     * One 16-byte chunk of nibbles at {@code qs} of the routed segment: the low nibbles into 16
     * elements at {@code o}, the high nibbles into 16 elements at {@code o + 32} elements. The
     * unpack runs in 32-bit lanes; width-generic via {@link VectorSupport#DECODE_PARTS}.
     */
    private static void pair(
            MemorySegment ws,
            long qs,
            float sc0,
            float mn0,
            float sc1,
            float mn1,
            MemorySegment dst,
            long o) {
        var qb =
                ByteVector.fromMemorySegment(
                        ByteVector.SPECIES_128, ws, qs, ByteOrder.LITTLE_ENDIAN);
        FloatVector vs0 = FloatVector.broadcast(F_SPECIES, sc0),
                vm0 = FloatVector.broadcast(F_SPECIES, mn0);
        FloatVector vs1 = FloatVector.broadcast(F_SPECIES, sc1),
                vm1 = FloatVector.broadcast(F_SPECIES, mn1);
        for (int p = 0; p < VectorSupport.DECODE_PARTS; p++) {
            IntVector q = (IntVector) qb.castShape(I_SPECIES, p);
            long po = o + (long) p * VectorSupport.F_LEN * 4;
            ((FloatVector) q.and(0xF).castShape(F_SPECIES, 0))
                    .fma(vs0, vm0)
                    .intoMemorySegment(dst, po, ByteOrder.LITTLE_ENDIAN);
            ((FloatVector) q.lanewise(VectorOperators.LSHR, 4).and(0xF).castShape(F_SPECIES, 0))
                    .fma(vs1, vm1)
                    .intoMemorySegment(dst, po + 32 * 4L, ByteOrder.LITTLE_ENDIAN);
        }
    }
}
