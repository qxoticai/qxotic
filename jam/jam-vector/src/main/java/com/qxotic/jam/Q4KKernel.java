package com.qxotic.jam;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

import static com.qxotic.jam.VectorSupport.F_SPECIES;
import static com.qxotic.jam.VectorSupport.readByte;
import static com.qxotic.jam.VectorSupport.readFloat16;
import static com.qxotic.jam.VectorSupport.readInt;
import static com.qxotic.jam.VectorSupport.readLong;

/**
 * Q4_K gemm, relocated from jinfer (segment-based). Q4_K super-block: 256 elements / 144 bytes
 * ({@code fp16 d, dmin; 12 packed scale/min bytes; 128 nibble bytes}); value {@code d·sc·nibble − dmin·m}.
 * Dequantizes a {@link BandGemm#MR}-row band into an F32 scratch, then {@link BandGemm} sweeps the columns —
 * so the 6-bit super-block scale unpack is amortized once per row, not per column tile.
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
            int low = isMin
                    ? (Byte.toUnsignedInt(readByte(mem, scalesOffset + lowIdx)) >> 4)
                    : (Byte.toUnsignedInt(readByte(mem, scalesOffset + lowIdx)) & 0xF);
            int high = (Byte.toUnsignedInt(readByte(mem, scalesOffset + highIdx)) >> 6) & 0x3;
            return low | (high << 4);
        }
    }

    /** The 8 sub-block scales unpacked branch-free into one byte-per-value long (LSB = sub-block 0). */
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

    /** Dequantize one Q4_K weight row (dim1 % 256 == 0) into {@code dst} at {@code dstOffset}. 512-bit. */
    static void dequantizeRow(MemorySegment w, long rowElemOffset, int dim1, float[] dst, int dstOffset) {
        int kblocks = dim1 / BLOCK;
        long firstBlock = rowElemOffset / BLOCK;
        for (int blk = 0; blk < kblocks; blk++) {
            long b = (firstBlock + blk) * TYPE;
            float d = readFloat16(w, b);
            float dmin = readFloat16(w, b + 2);
            long packedSc = packedScales(w, b + 4);
            long packedMn = packedMins(w, b + 4);
            int blockBase = dstOffset + blk * BLOCK;
            for (int g = 0; g < 4; g++) {
                var vdsc0 = FloatVector.broadcast(F_SPECIES, d * (int) ((packedSc >>> (16 * g)) & 0xFF));
                var vnegm0 = FloatVector.broadcast(F_SPECIES, -(dmin * (int) ((packedMn >>> (16 * g)) & 0xFF)));
                var vdsc1 = FloatVector.broadcast(F_SPECIES, d * (int) ((packedSc >>> (16 * g + 8)) & 0xFF));
                var vnegm1 = FloatVector.broadcast(F_SPECIES, -(dmin * (int) ((packedMn >>> (16 * g + 8)) & 0xFF)));
                for (int c = 0; c < 2; c++) {
                    var wb = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b + 16 + (long) g * 32 + c * 16, ByteOrder.LITTLE_ENDIAN);
                    ((FloatVector) wb.and((byte) 0xF).castShape(F_SPECIES, 0)).fma(vdsc0, vnegm0).intoArray(dst, blockBase + g * 64 + c * 16);
                    ((FloatVector) wb.lanewise(VectorOperators.LSHR, 4).castShape(F_SPECIES, 0)).fma(vdsc1, vnegm1).intoArray(dst, blockBase + g * 64 + 32 + c * 16);
                }
            }
        }
    }
}
