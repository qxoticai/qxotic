package com.qxotic.jam.vector;

import static com.qxotic.jam.vector.VectorSupport.readByte;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShuffle;

/**
 * NVFP4 gemm, relocated from jinfer (segment-based). NVFP4 block: 64 elements / 36 bytes ({@code 4
 * ue4m3 sub-block scales; 32 nibble bytes}), value {@code lut[nibble]·ue4m3(d[sub])}. A {@link
 * BandGemm#MR}-row band is dequantized into an F32 scratch, then {@link BandGemm}'s 3x3 F32 band
 * sweeps the columns. Identical to jinfer's {@code NVFP4FloatTensor.vectorGemm512}.
 */
public final class Nvfp4Kernel {

    private Nvfp4Kernel() {}

    static final int QK = 64, BYTES = 36;

    private static final int[] NVFP4_VALUES = {
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
    };

    /** Nibble code -> value, as bytes for the in-register LUT permute (vpshufb). */
    private static final byte[] NVFP4_LUT = new byte[16];

    /** UE4M3 byte -> f32 table (index = raw unsigned byte; identical to {@link #ue4m3ToFp32}). */
    private static final float[] UE4M3 = new float[256];

    static {
        for (int i = 0; i < 16; i++) NVFP4_LUT[i] = (byte) NVFP4_VALUES[i];
        for (int i = 0; i < 256; i++) UE4M3[i] = ue4m3ToFp32(i);
    }

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
                Nvfp4Kernel::dequantizeRow);
    }

    /**
     * Dequantize one NVFP4 weight row (dim1 % 64 == 0) into {@code dst[dstOffset..]} in element
     * order. 512-bit: two SPECIES_128 loads per 64-elem block, nibble codes decoded via one vpshufb
     * LUT each (same idiom as {@link Mxfp4Kernel}); sub-block halves are split into 256-bit stores
     * so element order is exact without cross-lane permutes. Scalar fallback for narrower vectors.
     */
    private static void dequantizeRow(
            MemorySegment w, long rowElemOffset, int dim1, MemorySegment dst, long dstBase) {
        if (VectorSupport.F_SPECIES.vectorBitSize() == 512) {
            dequantizeRow512(w, rowElemOffset, dim1, dst, dstBase);
            return;
        }
        int kblocks = dim1 / QK;
        long firstBlock = rowElemOffset / QK;
        for (int blk = 0; blk < kblocks; blk++) {
            long bo = (firstBlock + blk) * BYTES;
            long base = dstBase + (long) blk * QK * 4;
            for (int s = 0; s < 4; s++) {
                float d = ue4m3ToFp32(Byte.toUnsignedInt(readByte(w, bo + s)));
                for (int j = 0; j < 8; j++) {
                    int packed = Byte.toUnsignedInt(readByte(w, bo + 4 + s * 8 + j));
                    dst.set(
                            JAVA_FLOAT_UNALIGNED,
                            base + (s * 16 + j) * 4L,
                            NVFP4_VALUES[packed & 0x0F] * d); // low  -> elem j
                    dst.set(
                            JAVA_FLOAT_UNALIGNED,
                            base + (s * 16 + 8 + j) * 4L,
                            NVFP4_VALUES[packed >>> 4] * d); // high -> elem j + 8
                }
            }
        }
    }

    /** 512-bit vectorized dequant; see {@link #dequantizeRow}. */
    private static void dequantizeRow512(
            MemorySegment w, long rowElemOffset, int dim1, MemorySegment dst, long dstBase) {
        int kblocks = dim1 / QK;
        long firstBlock = rowElemOffset / QK;
        ByteVector lut = ByteVector.fromArray(ByteVector.SPECIES_128, NVFP4_LUT, 0);
        // Lane orders: [loA(0-7), hiA(0-7)] = [0..7, 16..23]; [loB, hiB] = [8..15, 24..31].
        VectorShuffle<Float> shufA =
                VectorShuffle.fromArray(
                        VectorSupport.F_SPECIES,
                        new int[] {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23},
                        0);
        VectorShuffle<Float> shufB =
                VectorShuffle.fromArray(
                        VectorSupport.F_SPECIES,
                        new int[] {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31},
                        0);
        for (int blk = 0; blk < kblocks; blk++) {
            long bo = (firstBlock + blk) * BYTES;
            long base = dstBase + (long) blk * QK * 4;
            float d0 = UE4M3[Byte.toUnsignedInt(readByte(w, bo))];
            float d1 = UE4M3[Byte.toUnsignedInt(readByte(w, bo + 1))];
            float d2 = UE4M3[Byte.toUnsignedInt(readByte(w, bo + 2))];
            float d3 = UE4M3[Byte.toUnsignedInt(readByte(w, bo + 3))];
            ByteVector p01 =
                    ByteVector.fromMemorySegment(
                            ByteVector.SPECIES_128, w, bo + 4, ByteOrder.LITTLE_ENDIAN);
            ByteVector p23 =
                    ByteVector.fromMemorySegment(
                            ByteVector.SPECIES_128, w, bo + 4 + 16, ByteOrder.LITTLE_ENDIAN);
            storeSubPair(
                    lut,
                    p01.and((byte) 0x0F),
                    p01.lanewise(VectorOperators.LSHR, 4),
                    d0,
                    d1,
                    shufA,
                    shufB,
                    dst,
                    base);
            storeSubPair(
                    lut,
                    p23.and((byte) 0x0F),
                    p23.lanewise(VectorOperators.LSHR, 4),
                    d2,
                    d3,
                    shufA,
                    shufB,
                    dst,
                    base + 128);
        }
    }

    /**
     * Decode two adjacent sub-blocks (16 nibble bytes): {@code lo}/{@code hi} lanes 0-7 are
     * sub-block A's low/high nibbles, lanes 8-15 sub-block B's. Element order per sub-block is
     * [lo(8), hi(8)], restored by two-source 512-bit rearranges ({@link #SHUF_A}/{@link #SHUF_B},
     * vpermt2ps) - a 256-bit extract/store split was ~60x slower (not intrinsified by a jvmci JIT).
     */
    private static void storeSubPair(
            ByteVector lut,
            ByteVector lo,
            ByteVector hi,
            float dA,
            float dB,
            VectorShuffle<Float> shufA,
            VectorShuffle<Float> shufB,
            MemorySegment dst,
            long base) {
        FloatVector loF =
                (FloatVector) lut.rearrange(lo.toShuffle()).castShape(VectorSupport.F_SPECIES, 0);
        FloatVector hiF =
                (FloatVector) lut.rearrange(hi.toShuffle()).castShape(VectorSupport.F_SPECIES, 0);
        loF.rearrange(shufA, hiF)
                .mul(FloatVector.broadcast(VectorSupport.F_SPECIES, dA))
                .intoMemorySegment(dst, base, ByteOrder.LITTLE_ENDIAN);
        loF.rearrange(shufB, hiF)
                .mul(FloatVector.broadcast(VectorSupport.F_SPECIES, dB))
                .intoMemorySegment(dst, base + 64, ByteOrder.LITTLE_ENDIAN);
    }

    /**
     * UE4M3 (unsigned FP8 E4M3) -> float; matches jam_ue4m3_to_float / ggml_ue4m3_to_fp32 (bit 7
     * ignored).
     */
    private static float ue4m3ToFp32(int x) {
        if (x == 0 || x == 0x7F) return 0f;
        int e = (x >>> 3) & 0xF, m = x & 0x7;
        return e != 0
                ? (1f + m / 8f) * (float) Math.scalb(1.0, e - 7)
                : m * (float) Math.scalb(1.0, -9);
    }
}
