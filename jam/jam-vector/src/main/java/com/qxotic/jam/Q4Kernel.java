package com.qxotic.jam;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

import static com.qxotic.jam.VectorSupport.F_SPECIES;
import static com.qxotic.jam.VectorSupport.readFloat16;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;

/**
 * Q4_0 register-tiled gemm, relocated from jinfer (segment-based). {@code C = W @ Aᵀ}: weights are Q4_0
 * blocks ({@code fp16 d; 16 nibble bytes}, value {@code d·(nibble-8)}) read from {@code w}; activation is
 * F32 from {@code (a, aBase)}; output F32 to {@code (o, oBase)}. Strides are ELEMENT strides; {@code wOff}
 * is the weight ELEMENT offset (the byte offset is block-derived). Behaviour is identical to jinfer's
 * {@code Q4_0FloatTensor.vectorGemm512} — jinfer now delegates here.
 */
public final class Q4Kernel {

    private Q4Kernel() {}

    private static final int BLOCK = 32, TYPE = 18, HALF = BLOCK / 2;   // Q4_0: 32 elems, 18 bytes/block

    public static void gemm(MemorySegment w,
                     MemorySegment a, long aBase,
                     MemorySegment o, long oBase,
                     int aStride, int oStride, int n, int m, int k, long wOff) {
        final int seqTile = Math.max(4, VectorSupport.SEQ_TILE);
        final int rowTile = Math.max(2, VectorSupport.ROW_TILE);
        final int seqTileCount = (n + seqTile - 1) / seqTile;
        final int rowTileCount = (m + rowTile - 1) / rowTile;
        int tileCount = rowTileCount * seqTileCount;
        if (tileCount == 0) return;
        int workers = Math.min(tileCount, Math.max(1, VectorSupport.THREADS));
        VectorSupport.parallelFor(0, workers, worker -> {
            int tileStart = (int) ((long) tileCount * worker / workers);
            int tileEnd = (int) ((long) tileCount * (worker + 1) / workers);
            for (int tileIndex = tileStart; tileIndex < tileEnd; tileIndex++) {
                int rowStart = (tileIndex / seqTileCount) * rowTile;
                int s0 = (tileIndex % seqTileCount) * seqTile;
                int rowEnd = Math.min(m, rowStart + rowTile);
                int seqEnd = Math.min(n, s0 + seqTile);
                int row = rowStart;
                for (; row + 1 < rowEnd; row += 2) {
                    int s = s0;
                    for (; s + 3 < seqEnd; s += 4) {
                        tile2x4(w, a, aBase, o, oBase, aStride, oStride, k, wOff, row, s);
                    }
                    for (; s < seqEnd; s++) {
                        store(o, oBase, (long) s * oStride + row,     dot(w, wOff + (long) row * k,       a, aBase, (long) s * aStride, k));
                        store(o, oBase, (long) s * oStride + row + 1, dot(w, wOff + (long) (row + 1) * k, a, aBase, (long) s * aStride, k));
                    }
                }
                for (; row < rowEnd; row++) {
                    for (int s = s0; s < seqEnd; s++) {
                        store(o, oBase, (long) s * oStride + row, dot(w, wOff + (long) row * k, a, aBase, (long) s * aStride, k));
                    }
                }
            }
        });
    }

    // 2 weight rows x 4 activation columns; nibbles decoded once per block, shared by all 4 columns. 512-bit.
    private static void tile2x4(MemorySegment w, MemorySegment a, long aBase, MemorySegment o, long oBase,
                                int aStride, int oStride, int k, long wOff, int row, int s) {
        final long rowStride = (long) (k / BLOCK) * TYPE;
        long b0 = (wOff + (long) row * k) / BLOCK * TYPE;
        long b1 = b0 + rowStride;
        long x0 = aBase + 4L * ((long) s * aStride);
        long x1 = x0 + 4L * aStride, x2 = x1 + 4L * aStride, x3 = x2 + 4L * aStride;
        FloatVector c00 = FloatVector.zero(F_SPECIES), c01 = FloatVector.zero(F_SPECIES), c02 = FloatVector.zero(F_SPECIES), c03 = FloatVector.zero(F_SPECIES);
        FloatVector c10 = FloatVector.zero(F_SPECIES), c11 = FloatVector.zero(F_SPECIES), c12 = FloatVector.zero(F_SPECIES), c13 = FloatVector.zero(F_SPECIES);
        for (int j = 0; j < k; j += BLOCK, b0 += TYPE, b1 += TYPE) {
            var vd0 = FloatVector.broadcast(F_SPECIES, readFloat16(w, b0));
            var vd1 = FloatVector.broadcast(F_SPECIES, readFloat16(w, b1));
            var w0b = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + 2, ByteOrder.LITTLE_ENDIAN);
            var w1b = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b1 + 2, ByteOrder.LITTLE_ENDIAN);
            var w0lo = ((FloatVector) w0b.and((byte) 0xF).sub((byte) 8).castShape(F_SPECIES, 0)).mul(vd0);
            var w0hi = ((FloatVector) w0b.lanewise(VectorOperators.LSHR, 4).sub((byte) 8).castShape(F_SPECIES, 0)).mul(vd0);
            var w1lo = ((FloatVector) w1b.and((byte) 0xF).sub((byte) 8).castShape(F_SPECIES, 0)).mul(vd1);
            var w1hi = ((FloatVector) w1b.lanewise(VectorOperators.LSHR, 4).sub((byte) 8).castShape(F_SPECIES, 0)).mul(vd1);
            long xOff = 4L * j;
            FloatVector aLo, aHi;
            aLo = FloatVector.fromMemorySegment(F_SPECIES, a, x0 + xOff, ByteOrder.LITTLE_ENDIAN);
            aHi = FloatVector.fromMemorySegment(F_SPECIES, a, x0 + xOff + 64, ByteOrder.LITTLE_ENDIAN);
            c00 = c00.add(w0hi.fma(aHi, w0lo.mul(aLo)));
            c10 = c10.add(w1hi.fma(aHi, w1lo.mul(aLo)));
            aLo = FloatVector.fromMemorySegment(F_SPECIES, a, x1 + xOff, ByteOrder.LITTLE_ENDIAN);
            aHi = FloatVector.fromMemorySegment(F_SPECIES, a, x1 + xOff + 64, ByteOrder.LITTLE_ENDIAN);
            c01 = c01.add(w0hi.fma(aHi, w0lo.mul(aLo)));
            c11 = c11.add(w1hi.fma(aHi, w1lo.mul(aLo)));
            aLo = FloatVector.fromMemorySegment(F_SPECIES, a, x2 + xOff, ByteOrder.LITTLE_ENDIAN);
            aHi = FloatVector.fromMemorySegment(F_SPECIES, a, x2 + xOff + 64, ByteOrder.LITTLE_ENDIAN);
            c02 = c02.add(w0hi.fma(aHi, w0lo.mul(aLo)));
            c12 = c12.add(w1hi.fma(aHi, w1lo.mul(aLo)));
            aLo = FloatVector.fromMemorySegment(F_SPECIES, a, x3 + xOff, ByteOrder.LITTLE_ENDIAN);
            aHi = FloatVector.fromMemorySegment(F_SPECIES, a, x3 + xOff + 64, ByteOrder.LITTLE_ENDIAN);
            c03 = c03.add(w0hi.fma(aHi, w0lo.mul(aLo)));
            c13 = c13.add(w1hi.fma(aHi, w1lo.mul(aLo)));
        }
        long base = (long) s * oStride + row;
        store(o, oBase, base,                 c00.reduceLanes(VectorOperators.ADD));
        store(o, oBase, base + 1,             c10.reduceLanes(VectorOperators.ADD));
        store(o, oBase, base + oStride,       c01.reduceLanes(VectorOperators.ADD));
        store(o, oBase, base + oStride + 1,   c11.reduceLanes(VectorOperators.ADD));
        store(o, oBase, base + 2L * oStride,     c02.reduceLanes(VectorOperators.ADD));
        store(o, oBase, base + 2L * oStride + 1, c12.reduceLanes(VectorOperators.ADD));
        store(o, oBase, base + 3L * oStride,     c03.reduceLanes(VectorOperators.ADD));
        store(o, oBase, base + 3L * oStride + 1, c13.reduceLanes(VectorOperators.ADD));
    }

    /** One-row vectorized dot, with scalar fallback for the unaligned head + tail (512/256/128 widths). */
    static float dot(MemorySegment w, long wOff, MemorySegment a, long aBase, long aOff, int size) {
        float result = 0f;
        int j = 0;
        int alignmentBound = (int) Math.min(size, -wOff & (BLOCK - 1));
        if (alignmentBound > 0) { result += scalarDot(w, wOff, a, aBase, aOff, alignmentBound); j += alignmentBound; }

        FloatVector val = FloatVector.zero(F_SPECIES);
        long blockOffset = (wOff + j) / BLOCK * TYPE;
        int upperBound = j + (size - j) / BLOCK * BLOCK;
        for (; j < upperBound; j += BLOCK, blockOffset += TYPE) {
            var wScale = FloatVector.broadcast(F_SPECIES, readFloat16(w, blockOffset));
            var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, blockOffset + 2, ByteOrder.LITTLE_ENDIAN);
            var loBytes = wBytes.and((byte) 0xF).sub((byte) 8);
            var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4).sub((byte) 8);
            int len = F_SPECIES.length();
            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    var s0 = av(a, aBase, aOff + j).mul(loBytes.castShape(F_SPECIES, 0));
                    var s1 = av(a, aBase, aOff + j + len).mul(hiBytes.castShape(F_SPECIES, 0));
                    val = s0.add(s1).fma(wScale, val);
                }
                case 256 -> {
                    var s0 = av(a, aBase, aOff + j).mul(loBytes.castShape(F_SPECIES, 0));
                    var s1 = av(a, aBase, aOff + j + 2 * len).mul(hiBytes.castShape(F_SPECIES, 0));
                    s0 = av(a, aBase, aOff + j + len).fma(loBytes.castShape(F_SPECIES, 1), s0);
                    s1 = av(a, aBase, aOff + j + 3 * len).fma(hiBytes.castShape(F_SPECIES, 1), s1);
                    val = s0.add(s1).fma(wScale, val);
                }
                case 128 -> {
                    for (int i = 0; i < 2; ++i) {
                        var tmp = i == 0 ? loBytes : hiBytes;
                        var s0 = av(a, aBase, aOff + j + (i * 4) * len).mul(tmp.castShape(F_SPECIES, 0));
                        var s1 = av(a, aBase, aOff + j + (i * 4 + 2) * len).mul(tmp.castShape(F_SPECIES, 2));
                        s0 = av(a, aBase, aOff + j + (i * 4 + 1) * len).fma(tmp.castShape(F_SPECIES, 1), s0);
                        s1 = av(a, aBase, aOff + j + (i * 4 + 3) * len).fma(tmp.castShape(F_SPECIES, 3), s1);
                        val = s0.add(s1).fma(wScale, val);
                    }
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }
        result += val.reduceLanes(VectorOperators.ADD);
        if (j < size) result += scalarDot(w, wOff + j, a, aBase, aOff + j, size - j);
        return result;
    }

    /** Scalar Q4_0 · F32 dot over {@code size} elements. */
    static float scalarDot(MemorySegment w, long wOff, MemorySegment a, long aBase, long aOff, int size) {
        double sum = 0;
        for (int i = 0; i < size; i++) {
            long idx = wOff + i;
            long blk = idx / BLOCK * TYPE;
            int within = (int) (idx % BLOCK);
            float scale = readFloat16(w, blk);
            int b = w.get(JAVA_BYTE, blk + 2 + (within % HALF)) & 0xFF;
            int q = (within < HALF ? (b & 0xF) : (b >> 4)) - 8;
            sum += (double) (q * scale) * a.get(JAVA_FLOAT_UNALIGNED, aBase + (aOff + i) * 4);
        }
        return (float) sum;
    }

    private static FloatVector av(MemorySegment a, long aBase, long elem) {
        return FloatVector.fromMemorySegment(F_SPECIES, a, aBase + elem * 4, ByteOrder.LITTLE_ENDIAN);
    }

    private static void store(MemorySegment o, long oBase, long elem, float v) {
        o.set(JAVA_FLOAT_UNALIGNED, oBase + elem * 4, v);
    }
}
