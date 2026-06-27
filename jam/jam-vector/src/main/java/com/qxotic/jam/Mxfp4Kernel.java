package com.qxotic.jam;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

import static com.qxotic.jam.VectorSupport.F_SPECIES;
import static com.qxotic.jam.VectorSupport.readByte;

/**
 * MXFP4 gemm, relocated from jinfer (segment-based). MXFP4 block: 32 elements / 17 bytes
 * ({@code e8m0 scale; 16 nibble bytes}), value {@code lut[nibble]·e8m0Half(scale)}. A group of
 * {@link BandGemm#MR} output rows is dequantized once into an F32 scratch, then {@link BandGemm}'s
 * decode-free F32 band sweeps the columns. Identical to jinfer's {@code MXFP4FloatTensor.vectorGemmMxfp4}.
 */
public final class Mxfp4Kernel {

    private Mxfp4Kernel() {}

    static final int QK = 32, BYTES = 17;

    private static final byte[] MXFP4_LUT = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

    public static void gemm(MemorySegment w, MemorySegment a, long aBase, MemorySegment o, long oBase,
                            int aStride, int oStride, int n, int m, int k, long wOff) {
        int groups = m / BandGemm.MR;
        VectorSupport.parallelFor(0, groups, g -> {
            int row0 = g * BandGemm.MR;
            float[] band = BandGemm.bandScratch(BandGemm.MR * k);
            for (int i = 0; i < BandGemm.MR; i++) {
                dequantizeRow(w, wOff + (long) (row0 + i) * k, k, band, i * k);
            }
            int s = 0;
            for (; s + BandGemm.NR <= n; s += BandGemm.NR) {
                BandGemm.band(band, k, a, aBase, o, oBase, aStride, oStride, row0, s);
            }
            for (; s < n; s++) {
                for (int i = 0; i < BandGemm.MR; i++) {
                    BandGemm.store(o, oBase, (long) s * oStride + row0 + i, BandGemm.dotDeq(band, i * k, k, a, aBase, (long) s * aStride));
                }
            }
        });
        for (int row = groups * BandGemm.MR; row < m; row++) {  // trailing rows: cheap per-column dots
            float[] band = BandGemm.bandScratch(k);
            dequantizeRow(w, wOff + (long) row * k, k, band, 0);
            float[] bf = band;
            int rr = row;
            VectorSupport.parallelFor(0, n, s -> BandGemm.store(o, oBase, (long) s * oStride + rr, BandGemm.dotDeq(bf, 0, k, a, aBase, (long) s * aStride)));
        }
    }

    /** Dequantize one MXFP4 weight row (dim1 % 32 == 0) into {@code dst} at {@code dstOffset}. 512-bit. */
    private static void dequantizeRow(MemorySegment w, long rowElemOffset, int dim1, float[] dst, int dstOffset) {
        int kblocks = dim1 / QK;
        long firstBlock = rowElemOffset / QK;
        for (int blk = 0; blk < kblocks; blk++) {
            long blockOffset = (firstBlock + blk) * BYTES;
            float d = e8m0ToFp32Half(Byte.toUnsignedInt(readByte(w, blockOffset)));
            ByteVector packed = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, blockOffset + 1, ByteOrder.LITTLE_ENDIAN);
            FloatVector loC = ((FloatVector) mxfp4Decode(packed.and((byte) 0x0F)).castShape(F_SPECIES, 0)).mul(d);
            FloatVector hiC = ((FloatVector) mxfp4Decode(packed.lanewise(VectorOperators.LSHR, 4)).castShape(F_SPECIES, 0)).mul(d);
            int base = dstOffset + blk * QK;
            loC.intoArray(dst, base);                  // block elems 0..15 (low nibbles)
            hiC.intoArray(dst, base + QK / 2);         // block elems 16..31 (high nibbles)
        }
    }

    /** Decode 16 nibble codes to signed MXFP4 values via one in-register table permute (vpshufb). */
    private static ByteVector mxfp4Decode(ByteVector nibbles) {
        return ByteVector.fromArray(ByteVector.SPECIES_128, MXFP4_LUT, 0).rearrange(nibbles.toShuffle());
    }

    private static float e8m0ToFp32Half(int x) {
        int bits;
        if (x < 2) {
            bits = 0x00200000 << x;
        } else {
            bits = (x - 1) << 23;
        }
        return Float.intBitsToFloat(bits);
    }
}
