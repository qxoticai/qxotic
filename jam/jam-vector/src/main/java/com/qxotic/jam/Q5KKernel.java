package com.qxotic.jam;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

import static com.qxotic.jam.Q4KKernel.getScaleMinK4;
import static com.qxotic.jam.VectorSupport.F_SPECIES;
import static com.qxotic.jam.VectorSupport.readByte;
import static com.qxotic.jam.VectorSupport.readFloat16;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;

/**
 * Q5_K register-tiled gemm, relocated from jinfer (segment-based). Q5_K super-block: 256 elements / 176
 * bytes ({@code fp16 d, dmin; 12 scale bytes; 32 qh bytes (5th bit); 128 nibble bytes}); value
 * {@code d·sc·quant − dmin·m} with {@code quant = nibble | (qhBit<<4)}. Identical to jinfer's
 * {@code Q5_KFloatTensor.vectorGemm512}; reuses {@link Q4KKernel#getScaleMinK4}.
 */
public final class Q5KKernel {

    private Q5KKernel() {}

    static final int BLOCK = 256, TYPE = 176;

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
            var qh0r0 = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + 16, ByteOrder.LITTLE_ENDIAN);
            var qh1r0 = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + 32, ByteOrder.LITTLE_ENDIAN);
            var qh0r1 = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b1 + 16, ByteOrder.LITTLE_ENDIAN);
            var qh1r1 = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b1 + 32, ByteOrder.LITTLE_ENDIAN);
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
                int bitLo = 2 * g, bitHi = 2 * g + 1;
                long xLo = 4L * (j + g * 64);
                long xHi = xLo + 4L * 32;
                for (int c = 0; c < 2; c++) {
                    long qOff = (long) g * 32 + c * 16;
                    var w0b = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b0 + 48 + qOff, ByteOrder.LITTLE_ENDIAN);
                    var w1b = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, b1 + 48 + qOff, ByteOrder.LITTLE_ENDIAN);
                    var qhb0 = (c == 0) ? qh0r0 : qh1r0;
                    var qhb1 = (c == 0) ? qh0r1 : qh1r1;
                    var w0loB = w0b.and((byte) 0xF).or(qhb0.lanewise(VectorOperators.LSHR, bitLo).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
                    var w0hiB = w0b.lanewise(VectorOperators.LSHR, 4).or(qhb0.lanewise(VectorOperators.LSHR, bitHi).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
                    var w1loB = w1b.and((byte) 0xF).or(qhb1.lanewise(VectorOperators.LSHR, bitLo).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
                    var w1hiB = w1b.lanewise(VectorOperators.LSHR, 4).or(qhb1.lanewise(VectorOperators.LSHR, bitHi).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
                    var w0lo = ((FloatVector) w0loB.castShape(F_SPECIES, 0)).fma(vd0Lo, vm0Lo);
                    var w0hi = ((FloatVector) w0hiB.castShape(F_SPECIES, 0)).fma(vd0Hi, vm0Hi);
                    var w1lo = ((FloatVector) w1loB.castShape(F_SPECIES, 0)).fma(vd1Lo, vm1Lo);
                    var w1hi = ((FloatVector) w1hiB.castShape(F_SPECIES, 0)).fma(vd1Hi, vm1Hi);
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
            long qhOff = blockOffset + 16;
            long qsOff = blockOffset + 48;
            var qh0 = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, qhOff, ByteOrder.LITTLE_ENDIAN);
            var qh1 = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, qhOff + 16, ByteOrder.LITTLE_ENDIAN);
            for (int g = 0; g < 4; g++) {
                int loSubBlock = g * 2;
                int hiSubBlock = loSubBlock + 1;
                float d1 = d * getScaleMinK4(loSubBlock, w, scalesOff, false);
                float m1 = dmin * getScaleMinK4(loSubBlock, w, scalesOff, true);
                float d2 = d * getScaleMinK4(hiSubBlock, w, scalesOff, false);
                float m2 = dmin * getScaleMinK4(hiSubBlock, w, scalesOff, true);
                int qhBitPosLo = 2 * g;
                int qhBitPosHi = qhBitPosLo + 1;
                long groupQsOff = qsOff + (long) g * 32;
                var d1Vec = FloatVector.broadcast(F_SPECIES, d1);
                var d2Vec = FloatVector.broadcast(F_SPECIES, d2);
                var negM1Vec = FloatVector.broadcast(F_SPECIES, -m1);
                var negM2Vec = FloatVector.broadcast(F_SPECIES, -m2);
                for (int c = 0; c < 2; c++) {
                    long loBase = aOff + j + g * 64 + c * 16;
                    long hiBase = aOff + j + g * 64 + 32 + c * 16;
                    var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, groupQsOff + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    var loQ = wBytes.and((byte) 0xF);
                    var hiQ = wBytes.lanewise(VectorOperators.LSHR, 4);
                    var qhBytes = c == 0 ? qh0 : qh1;
                    loQ = loQ.or(qhBytes.lanewise(VectorOperators.LSHR, qhBitPosLo).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
                    hiQ = hiQ.or(qhBytes.lanewise(VectorOperators.LSHR, qhBitPosHi).and((byte) 1).lanewise(VectorOperators.LSHL, 4));
                    switch (F_SPECIES.vectorBitSize()) {
                        case 512 -> {
                            var loQf = loQ.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var hiQf = hiQ.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val = loQf.fma(d1Vec, negM1Vec).fma(av(a, aBase, loBase), val);
                            val2 = hiQf.fma(d2Vec, negM2Vec).fma(av(a, aBase, hiBase), val2);
                        }
                        case 256 -> {
                            var loQf0 = loQ.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var loQf1 = loQ.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            var hiQf0 = hiQ.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var hiQf1 = hiQ.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            val = loQf0.fma(d1Vec, negM1Vec).fma(av(a, aBase, loBase), val);
                            val = loQf1.fma(d1Vec, negM1Vec).fma(av(a, aBase, loBase + F_SPECIES.length()), val);
                            val2 = hiQf0.fma(d2Vec, negM2Vec).fma(av(a, aBase, hiBase), val2);
                            val2 = hiQf1.fma(d2Vec, negM2Vec).fma(av(a, aBase, hiBase + F_SPECIES.length()), val2);
                        }
                        case 128 -> {
                            for (int p = 0; p < 4; p++) {
                                int off = p * F_SPECIES.length();
                                var loQf = loQ.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var hiQf = hiQ.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val = loQf.fma(d1Vec, negM1Vec).fma(av(a, aBase, loBase + off), val);
                                val2 = hiQf.fma(d2Vec, negM2Vec).fma(av(a, aBase, hiBase + off), val2);
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

    /** Decode element {@code index} of a Q5_K weight (the getFloat formula). */
    static float decode(MemorySegment w, long index) {
        long blockIndex = index / BLOCK;
        int withinBlock = (int) (index % BLOCK);
        long blockOffset = blockIndex * TYPE;
        float d = readFloat16(w, blockOffset);
        float dmin = readFloat16(w, blockOffset + 2);
        long scalesOffset = blockOffset + 4;
        long qhOffset = blockOffset + 16;
        long qsOffset = blockOffset + 48;
        int group = withinBlock / 64;
        int inGroup = withinBlock % 64;
        boolean isHigh = inGroup >= 32;
        int l = isHigh ? inGroup - 32 : inGroup;
        int subBlock = isHigh ? group * 2 + 1 : group * 2;
        int sc = getScaleMinK4(subBlock, w, scalesOffset, false);
        int m = getScaleMinK4(subBlock, w, scalesOffset, true);
        byte qsByte = readByte(w, qsOffset + group * 32 + l);
        int nibble = isHigh ? ((Byte.toUnsignedInt(qsByte) >> 4) & 0xF) : (Byte.toUnsignedInt(qsByte) & 0xF);
        int qhBitPos = isHigh ? 2 * group + 1 : 2 * group;
        int qhBit = (Byte.toUnsignedInt(readByte(w, qhOffset + l)) >> qhBitPos) & 1;
        int quant = nibble | (qhBit << 4);
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
