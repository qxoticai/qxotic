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
import static java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;

/**
 * Q4_K register-tiled gemm, relocated from jinfer (segment-based). Q4_K super-block: 256 elements / 144
 * bytes ({@code fp16 d, dmin; 12 packed scale/min bytes; 128 nibble bytes}); value
 * {@code d·sc·nibble − dmin·m}. Identical computation to jinfer's {@code Q4_KFloatTensor.vectorGemm512}.
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

    // ---- gemm ----

    public static void gemm(MemorySegment w, MemorySegment a, long aBase, MemorySegment o, long oBase,
                            int aStride, int oStride, int n, int m, int k, long wOff) {
        final int seqTile = Math.max(4, VectorSupport.SEQ_TILE_QK);
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

    // 2 weight rows x 4 activation columns: sub-scale decode + nibble dequant done once per row block.
    private static void tile2x4(MemorySegment w, MemorySegment a, long aBase, MemorySegment o, long oBase,
                                int aStride, int oStride, int k, long wOff, int row, int s) {
        final long rowStride = (long) (k / BLOCK) * TYPE;
        long b0 = (long) ((wOff + (long) row * k) / BLOCK) * TYPE;
        long b1 = b0 + rowStride;
        long x0 = aBase + 4L * ((long) s * aStride);
        long x1 = x0 + 4L * aStride, x2 = x1 + 4L * aStride, x3 = x2 + 4L * aStride;
        FloatVector c00 = FloatVector.zero(F_SPECIES), c01 = FloatVector.zero(F_SPECIES), c02 = FloatVector.zero(F_SPECIES), c03 = FloatVector.zero(F_SPECIES);
        FloatVector c10 = FloatVector.zero(F_SPECIES), c11 = FloatVector.zero(F_SPECIES), c12 = FloatVector.zero(F_SPECIES), c13 = FloatVector.zero(F_SPECIES);
        for (int j = 0; j < k; j += BLOCK, b0 += TYPE, b1 += TYPE) {
            float d0 = readFloat16(w, b0);
            float dmin0 = readFloat16(w, b0 + 2);
            float d1 = readFloat16(w, b1);
            float dmin1 = readFloat16(w, b1 + 2);
            for (int g = 0; g < 4; g++) {
                float r0dLo = d0 * getScaleMinK4(g * 2, w, b0 + 4, false);
                float r0mLo = -(dmin0 * getScaleMinK4(g * 2, w, b0 + 4, true));
                float r0dHi = d0 * getScaleMinK4(g * 2 + 1, w, b0 + 4, false);
                float r0mHi = -(dmin0 * getScaleMinK4(g * 2 + 1, w, b0 + 4, true));
                float r1dLo = d1 * getScaleMinK4(g * 2, w, b1 + 4, false);
                float r1mLo = -(dmin1 * getScaleMinK4(g * 2, w, b1 + 4, true));
                float r1dHi = d1 * getScaleMinK4(g * 2 + 1, w, b1 + 4, false);
                float r1mHi = -(dmin1 * getScaleMinK4(g * 2 + 1, w, b1 + 4, true));
                var vd0Lo = FloatVector.broadcast(F_SPECIES, r0dLo);
                var vm0Lo = FloatVector.broadcast(F_SPECIES, r0mLo);
                var vd0Hi = FloatVector.broadcast(F_SPECIES, r0dHi);
                var vm0Hi = FloatVector.broadcast(F_SPECIES, r0mHi);
                var vd1Lo = FloatVector.broadcast(F_SPECIES, r1dLo);
                var vm1Lo = FloatVector.broadcast(F_SPECIES, r1mLo);
                var vd1Hi = FloatVector.broadcast(F_SPECIES, r1dHi);
                var vm1Hi = FloatVector.broadcast(F_SPECIES, r1mHi);
                long xLo = 4L * (j + g * 64);
                long xHi = xLo + 4L * 32;
                for (int c = 0; c < 2; c++) {
                    long qOff = (long) g * 32 + c * 16;
                    var w0b = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + 16 + qOff, ByteOrder.LITTLE_ENDIAN);
                    var w1b = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b1 + 16 + qOff, ByteOrder.LITTLE_ENDIAN);
                    var w0lo = ((FloatVector) w0b.and((byte) 0xF).castShape(F_SPECIES, 0)).fma(vd0Lo, vm0Lo);
                    var w0hi = ((FloatVector) w0b.lanewise(VectorOperators.LSHR, 4).castShape(F_SPECIES, 0)).fma(vd0Hi, vm0Hi);
                    var w1lo = ((FloatVector) w1b.and((byte) 0xF).castShape(F_SPECIES, 0)).fma(vd1Lo, vm1Lo);
                    var w1hi = ((FloatVector) w1b.lanewise(VectorOperators.LSHR, 4).castShape(F_SPECIES, 0)).fma(vd1Hi, vm1Hi);
                    long off = c * 16L * 4L;
                    FloatVector aLo, aHi;
                    aLo = FloatVector.fromMemorySegment(F_SPECIES, a, x0 + xLo + off, ByteOrder.LITTLE_ENDIAN);
                    aHi = FloatVector.fromMemorySegment(F_SPECIES, a, x0 + xHi + off, ByteOrder.LITTLE_ENDIAN);
                    c00 = c00.add(w0hi.fma(aHi, w0lo.mul(aLo)));
                    c10 = c10.add(w1hi.fma(aHi, w1lo.mul(aLo)));
                    aLo = FloatVector.fromMemorySegment(F_SPECIES, a, x1 + xLo + off, ByteOrder.LITTLE_ENDIAN);
                    aHi = FloatVector.fromMemorySegment(F_SPECIES, a, x1 + xHi + off, ByteOrder.LITTLE_ENDIAN);
                    c01 = c01.add(w0hi.fma(aHi, w0lo.mul(aLo)));
                    c11 = c11.add(w1hi.fma(aHi, w1lo.mul(aLo)));
                    aLo = FloatVector.fromMemorySegment(F_SPECIES, a, x2 + xLo + off, ByteOrder.LITTLE_ENDIAN);
                    aHi = FloatVector.fromMemorySegment(F_SPECIES, a, x2 + xHi + off, ByteOrder.LITTLE_ENDIAN);
                    c02 = c02.add(w0hi.fma(aHi, w0lo.mul(aLo)));
                    c12 = c12.add(w1hi.fma(aHi, w1lo.mul(aLo)));
                    aLo = FloatVector.fromMemorySegment(F_SPECIES, a, x3 + xLo + off, ByteOrder.LITTLE_ENDIAN);
                    aHi = FloatVector.fromMemorySegment(F_SPECIES, a, x3 + xHi + off, ByteOrder.LITTLE_ENDIAN);
                    c03 = c03.add(w0hi.fma(aHi, w0lo.mul(aLo)));
                    c13 = c13.add(w1hi.fma(aHi, w1lo.mul(aLo)));
                }
            }
        }
        long o0 = (long) s * oStride + row;
        store(o, oBase, o0, c00.reduceLanes(VectorOperators.ADD));
        store(o, oBase, o0 + 1, c10.reduceLanes(VectorOperators.ADD));
        store(o, oBase, o0 + oStride, c01.reduceLanes(VectorOperators.ADD));
        store(o, oBase, o0 + oStride + 1, c11.reduceLanes(VectorOperators.ADD));
        store(o, oBase, o0 + 2L * oStride, c02.reduceLanes(VectorOperators.ADD));
        store(o, oBase, o0 + 2L * oStride + 1, c12.reduceLanes(VectorOperators.ADD));
        store(o, oBase, o0 + 3L * oStride, c03.reduceLanes(VectorOperators.ADD));
        store(o, oBase, o0 + 3L * oStride + 1, c13.reduceLanes(VectorOperators.ADD));
    }

    static float dot(MemorySegment w, long wOff, MemorySegment a, long aBase, long aOff, int size) {
        float result = 0f;
        int j = 0;
        int alignmentBound = (int) Math.min(size, -wOff & (BLOCK - 1));
        if (alignmentBound > 0) { result += scalarDot(w, wOff, a, aBase, aOff, alignmentBound); j += alignmentBound; }

        FloatVector val = FloatVector.zero(F_SPECIES);
        FloatVector val2 = FloatVector.zero(F_SPECIES);
        long blockOffset = (long) (wOff + j) / BLOCK * TYPE;
        int upperBound = j + (size - j) / BLOCK * BLOCK;
        for (; j < upperBound; j += BLOCK, blockOffset += TYPE) {
            float d = readFloat16(w, blockOffset);
            float dmin = readFloat16(w, blockOffset + 2);
            long scalesOff = blockOffset + 4;
            long qsOff = blockOffset + 16;
            long packedSc = packedScales(w, scalesOff);
            long packedMn = packedMins(w, scalesOff);
            for (int g = 0; g < 4; g++) {
                float d1 = d * (int) ((packedSc >>> (16 * g)) & 0xFF);
                float negM1 = -(dmin * (int) ((packedMn >>> (16 * g)) & 0xFF));
                float d2 = d * (int) ((packedSc >>> (16 * g + 8)) & 0xFF);
                float negM2 = -(dmin * (int) ((packedMn >>> (16 * g + 8)) & 0xFF));
                var d1Vec = FloatVector.broadcast(F_SPECIES, d1);
                var negM1Vec = FloatVector.broadcast(F_SPECIES, negM1);
                var d2Vec = FloatVector.broadcast(F_SPECIES, d2);
                var negM2Vec = FloatVector.broadcast(F_SPECIES, negM2);
                long loBase = aOff + j + g * 64;
                long hiBase = aOff + j + g * 64 + 32;
                for (int c = 0; c < 2; c++) {
                    var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, qsOff + (long) g * 32 + c * 16, ByteOrder.LITTLE_ENDIAN);
                    var loBytes = wBytes.and((byte) 0xF);
                    var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4);
                    long loIdx = loBase + c * 16;
                    long hiIdx = hiBase + c * 16;
                    switch (F_SPECIES.vectorBitSize()) {
                        case 512 -> {
                            var loQ = loBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val = loQ.fma(d1Vec, negM1Vec).fma(av(a, aBase, loIdx), val);
                            var hiQ = hiBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val2 = hiQ.fma(d2Vec, negM2Vec).fma(av(a, aBase, hiIdx), val2);
                        }
                        case 256 -> {
                            var loQ0 = loBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var loQ1 = loBytes.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            val = loQ0.fma(d1Vec, negM1Vec).fma(av(a, aBase, loIdx), val);
                            val2 = loQ1.fma(d1Vec, negM1Vec).fma(av(a, aBase, loIdx + F_SPECIES.length()), val2);
                            var hiQ0 = hiBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var hiQ1 = hiBytes.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            val = hiQ0.fma(d2Vec, negM2Vec).fma(av(a, aBase, hiIdx), val);
                            val2 = hiQ1.fma(d2Vec, negM2Vec).fma(av(a, aBase, hiIdx + F_SPECIES.length()), val2);
                        }
                        case 128 -> {
                            for (int p = 0; p < 4; p++) {
                                var loQ = loBytes.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val = loQ.fma(d1Vec, negM1Vec).fma(av(a, aBase, loIdx + p * F_SPECIES.length()), val);
                                var hiQ = hiBytes.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val2 = hiQ.fma(d2Vec, negM2Vec).fma(av(a, aBase, hiIdx + p * F_SPECIES.length()), val2);
                            }
                        }
                        default -> throw new UnsupportedOperationException(F_SPECIES.toString());
                    }
                }
            }
        }
        result += val.add(val2).reduceLanes(VectorOperators.ADD);
        if (j < size) result += scalarDot(w, wOff + j, a, aBase, aOff + j, size - j);
        return result;
    }

    /** Decode element {@code index} of a Q4_K weight (the getFloat formula). */
    static float decode(MemorySegment w, long index) {
        long blockIndex = index / BLOCK;
        int withinBlock = (int) (index % BLOCK);
        long blockOffset = blockIndex * TYPE;
        float d = readFloat16(w, blockOffset);
        float dmin = readFloat16(w, blockOffset + 2);
        long scalesOffset = blockOffset + 4;
        long qsOffset = blockOffset + 16;
        int group = withinBlock / 64;
        int inGroup = withinBlock % 64;
        int subBlock, nibbleIndex;
        boolean isHigh;
        if (inGroup < 32) { subBlock = group * 2; nibbleIndex = inGroup; isHigh = false; }
        else { subBlock = group * 2 + 1; nibbleIndex = inGroup - 32; isHigh = true; }
        int sc = getScaleMinK4(subBlock, w, scalesOffset, false);
        int m = getScaleMinK4(subBlock, w, scalesOffset, true);
        byte qsByte = readByte(w, qsOffset + group * 32 + nibbleIndex);
        int quant = isHigh ? ((Byte.toUnsignedInt(qsByte) >> 4) & 0xF) : (Byte.toUnsignedInt(qsByte) & 0xF);
        return d * sc * quant - dmin * m;
    }

    static float scalarDot(MemorySegment w, long wOff, MemorySegment a, long aBase, long aOff, int size) {
        double sum = 0;
        for (int i = 0; i < size; i++) {
            sum += (double) decode(w, wOff + i) * a.get(JAVA_FLOAT_UNALIGNED, aBase + (aOff + i) * 4);
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
