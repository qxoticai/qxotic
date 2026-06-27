package com.qxotic.jam;

import java.lang.foreign.MemorySegment;

import static com.qxotic.jam.VectorSupport.readByte;

/**
 * NVFP4 gemm, relocated from jinfer (segment-based). NVFP4 block: 64 elements / 36 bytes
 * ({@code 4 ue4m3 sub-block scales; 32 nibble bytes}), value {@code lut[nibble]·ue4m3(d[sub])}. A
 * {@link BandGemm#MR}-row band is dequantized into an F32 scratch, then {@link BandGemm}'s 3x3 F32 band
 * sweeps the columns. Identical to jinfer's {@code NVFP4FloatTensor.vectorGemm512}.
 */
public final class Nvfp4Kernel {

    private Nvfp4Kernel() {}

    static final int QK = 64, BYTES = 36;

    private static final int[] NVFP4_VALUES = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

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
        for (int row = groups * BandGemm.MR; row < m; row++) {   // trailing rows: per-column dots
            float[] band = BandGemm.bandScratch(k);
            dequantizeRow(w, wOff + (long) row * k, k, band, 0);
            float[] bf = band;
            int rr = row;
            VectorSupport.parallelFor(0, n, s -> BandGemm.store(o, oBase, (long) s * oStride + rr, BandGemm.dotDeq(bf, 0, k, a, aBase, (long) s * aStride)));
        }
    }

    /** Dequantize one NVFP4 weight row (dim1 % 64 == 0) into {@code dst[dstOffset..]} in element order. */
    private static void dequantizeRow(MemorySegment w, long rowElemOffset, int dim1, float[] dst, int dstOffset) {
        int kblocks = dim1 / QK;
        long firstBlock = rowElemOffset / QK;
        for (int blk = 0; blk < kblocks; blk++) {
            long bo = (firstBlock + blk) * BYTES;
            int base = dstOffset + blk * QK;
            for (int s = 0; s < 4; s++) {
                float d = ue4m3ToFp32(Byte.toUnsignedInt(readByte(w, bo + s)));
                for (int j = 0; j < 8; j++) {
                    int packed = Byte.toUnsignedInt(readByte(w, bo + 4 + s * 8 + j));
                    dst[base + s * 16 + j]     = NVFP4_VALUES[packed & 0x0F] * d;   // low  -> elem j
                    dst[base + s * 16 + 8 + j] = NVFP4_VALUES[packed >>> 4] * d;    // high -> elem j + 8
                }
            }
        }
    }

    /** UE4M3 (unsigned FP8 E4M3) -> float; matches jam_ue4m3_to_float / ggml_ue4m3_to_fp32 (bit 7 ignored). */
    private static float ue4m3ToFp32(int x) {
        if (x == 0 || x == 0x7F) return 0f;
        int e = (x >>> 3) & 0xF, m = x & 0x7;
        return e != 0 ? (1f + m / 8f) * (float) Math.scalb(1.0, e - 7) : m * (float) Math.scalb(1.0, -9);
    }
}
