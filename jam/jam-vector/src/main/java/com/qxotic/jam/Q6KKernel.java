package com.qxotic.jam;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

import static com.qxotic.jam.VectorSupport.F_SPECIES;
import static com.qxotic.jam.VectorSupport.readByte;
import static com.qxotic.jam.VectorSupport.readFloat16;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;

/**
 * Q6_K register-tiled gemm, relocated from jinfer (segment-based). Q6_K super-block: 256 elements / 210
 * bytes ({@code ql[128] | qh[64] | scales[16] int8 | fp16 d}); 6-bit quants (4 from ql nibble + 2 from qh),
 * value {@code d·sc·(q6−32)}. Identical to jinfer's {@code Q6_KFloatTensor.vectorGemm512}.
 */
public final class Q6KKernel {

    private Q6KKernel() {}

    static final int BLOCK = 256, TYPE = 210;

    public static void gemm(MemorySegment w, MemorySegment a, long aBase, MemorySegment o, long oBase,
                            int aStride, int oStride, int n, int m, int k, long wOff) {
        final int seqTile = Math.max(4, VectorSupport.SEQ_TILE_QK);
        final int rowTile = Math.max(1, VectorSupport.ROW_TILE);
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
                for (int row = rowStart; row < rowEnd; row++) {
                    int s = s0;
                    for (; s + 3 < seqEnd; s += 4) {
                        tile1x4(w, a, aBase, o, oBase, aStride, oStride, k, wOff, row, s);
                    }
                    for (; s < seqEnd; s++) {
                        store(o, oBase, (long) s * oStride + row, dot(w, wOff + (long) row * k, a, aBase, (long) s * aStride, k));
                    }
                }
            }
        });
    }

    // 1 weight row x 4 activation columns: the 6-bit unpack + scale is done once per 64-value group.
    private static void tile1x4(MemorySegment w, MemorySegment a, long aBase, MemorySegment o, long oBase,
                                int aStride, int oStride, int k, long wOff, int row, int s) {
        long blockOffset = (long) ((wOff + (long) row * k) / BLOCK) * TYPE;
        long x0 = aBase + 4L * ((long) s * aStride);
        long x1 = x0 + 4L * aStride, x2 = x1 + 4L * aStride, x3 = x2 + 4L * aStride;
        FloatVector c0 = FloatVector.zero(F_SPECIES);
        FloatVector c1 = FloatVector.zero(F_SPECIES);
        FloatVector c2 = FloatVector.zero(F_SPECIES);
        FloatVector c3 = FloatVector.zero(F_SPECIES);
        for (int j = 0; j < k; j += BLOCK, blockOffset += TYPE) {
            long qlOff = blockOffset;
            long qhOff = blockOffset + 128;
            long scOff = blockOffset + 192;
            float d = readFloat16(w, blockOffset + 208);
            for (int h = 0; h < 2; h++) {
                long qlBase = qlOff + h * 64;
                long qhBase = qhOff + h * 32;
                long base = 4L * (j + h * 128);
                for (int c = 0; c < 2; c++) {
                    var qlA = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, qlBase + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    var qlB = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, qlBase + 32 + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    var qhV = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, qhBase + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    var q0 = qlA.and((byte) 0xF).or(qhV.and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    var q1 = qlB.and((byte) 0xF).or(qhV.lanewise(VectorOperators.LSHR, 2).and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    var q2 = qlA.lanewise(VectorOperators.LSHR, 4).or(qhV.lanewise(VectorOperators.LSHR, 4).and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    var q3 = qlB.lanewise(VectorOperators.LSHR, 4).or(qhV.lanewise(VectorOperators.LSHR, 6).and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    var w0 = ((FloatVector) q0.castShape(F_SPECIES, 0)).mul(FloatVector.broadcast(F_SPECIES, d * readByte(w, scOff + h * 8 + c)));
                    var w1 = ((FloatVector) q1.castShape(F_SPECIES, 0)).mul(FloatVector.broadcast(F_SPECIES, d * readByte(w, scOff + h * 8 + 2 + c)));
                    var w2 = ((FloatVector) q2.castShape(F_SPECIES, 0)).mul(FloatVector.broadcast(F_SPECIES, d * readByte(w, scOff + h * 8 + 4 + c)));
                    var w3 = ((FloatVector) q3.castShape(F_SPECIES, 0)).mul(FloatVector.broadcast(F_SPECIES, d * readByte(w, scOff + h * 8 + 6 + c)));
                    long o0 = base + c * 16L * 4L;
                    long o1 = o0 + 32L * 4L;
                    long o2 = o0 + 64L * 4L;
                    long o3 = o0 + 96L * 4L;
                    FloatVector t, t2;
                    t = w0.mul(FloatVector.fromMemorySegment(F_SPECIES, a, x0 + o0, ByteOrder.LITTLE_ENDIAN));
                    t = w1.fma(FloatVector.fromMemorySegment(F_SPECIES, a, x0 + o1, ByteOrder.LITTLE_ENDIAN), t);
                    t2 = w2.mul(FloatVector.fromMemorySegment(F_SPECIES, a, x0 + o2, ByteOrder.LITTLE_ENDIAN));
                    t2 = w3.fma(FloatVector.fromMemorySegment(F_SPECIES, a, x0 + o3, ByteOrder.LITTLE_ENDIAN), t2);
                    c0 = c0.add(t.add(t2));
                    t = w0.mul(FloatVector.fromMemorySegment(F_SPECIES, a, x1 + o0, ByteOrder.LITTLE_ENDIAN));
                    t = w1.fma(FloatVector.fromMemorySegment(F_SPECIES, a, x1 + o1, ByteOrder.LITTLE_ENDIAN), t);
                    t2 = w2.mul(FloatVector.fromMemorySegment(F_SPECIES, a, x1 + o2, ByteOrder.LITTLE_ENDIAN));
                    t2 = w3.fma(FloatVector.fromMemorySegment(F_SPECIES, a, x1 + o3, ByteOrder.LITTLE_ENDIAN), t2);
                    c1 = c1.add(t.add(t2));
                    t = w0.mul(FloatVector.fromMemorySegment(F_SPECIES, a, x2 + o0, ByteOrder.LITTLE_ENDIAN));
                    t = w1.fma(FloatVector.fromMemorySegment(F_SPECIES, a, x2 + o1, ByteOrder.LITTLE_ENDIAN), t);
                    t2 = w2.mul(FloatVector.fromMemorySegment(F_SPECIES, a, x2 + o2, ByteOrder.LITTLE_ENDIAN));
                    t2 = w3.fma(FloatVector.fromMemorySegment(F_SPECIES, a, x2 + o3, ByteOrder.LITTLE_ENDIAN), t2);
                    c2 = c2.add(t.add(t2));
                    t = w0.mul(FloatVector.fromMemorySegment(F_SPECIES, a, x3 + o0, ByteOrder.LITTLE_ENDIAN));
                    t = w1.fma(FloatVector.fromMemorySegment(F_SPECIES, a, x3 + o1, ByteOrder.LITTLE_ENDIAN), t);
                    t2 = w2.mul(FloatVector.fromMemorySegment(F_SPECIES, a, x3 + o2, ByteOrder.LITTLE_ENDIAN));
                    t2 = w3.fma(FloatVector.fromMemorySegment(F_SPECIES, a, x3 + o3, ByteOrder.LITTLE_ENDIAN), t2);
                    c3 = c3.add(t.add(t2));
                }
            }
        }
        long oo = (long) s * oStride + row;
        store(o, oBase, oo, c0.reduceLanes(VectorOperators.ADD));
        store(o, oBase, oo + oStride, c1.reduceLanes(VectorOperators.ADD));
        store(o, oBase, oo + 2L * oStride, c2.reduceLanes(VectorOperators.ADD));
        store(o, oBase, oo + 3L * oStride, c3.reduceLanes(VectorOperators.ADD));
    }

    static float dot(MemorySegment w, long wOff, MemorySegment a, long aBase, long aOff, int size) {
        float result = 0f;
        int j = 0;
        int alignmentBound = (int) Math.min(size, -wOff & (BLOCK - 1));
        if (alignmentBound > 0) { result += scalarDot(w, wOff, a, aBase, aOff, alignmentBound); j += alignmentBound; }

        FloatVector acc0 = FloatVector.zero(F_SPECIES);
        FloatVector acc1 = FloatVector.zero(F_SPECIES);
        FloatVector acc2 = FloatVector.zero(F_SPECIES);
        FloatVector acc3 = FloatVector.zero(F_SPECIES);
        long blockOffset = (long) (wOff + j) / BLOCK * TYPE;
        int upperBound = j + (size - j) / BLOCK * BLOCK;
        for (; j < upperBound; j += BLOCK, blockOffset += TYPE) {
            long qlOff = blockOffset;
            long qhOff = blockOffset + 128;
            long scOff = blockOffset + 192;
            float d = readFloat16(w, blockOffset + 208);
            for (int h = 0; h < 2; h++) {
                long qlBase = qlOff + h * 64;
                long qhBase = qhOff + h * 32;
                long base = aOff + j + h * 128;
                for (int c = 0; c < 2; c++) {
                    var qlA = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, qlBase + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    var qlB = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, qlBase + 32 + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    var qhV = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, w, qhBase + c * 16L, ByteOrder.LITTLE_ENDIAN);
                    var q0 = qlA.and((byte) 0xF).or(qhV.and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    var q1 = qlB.and((byte) 0xF).or(qhV.lanewise(VectorOperators.LSHR, 2).and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    var q2 = qlA.lanewise(VectorOperators.LSHR, 4).or(qhV.lanewise(VectorOperators.LSHR, 4).and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    var q3 = qlB.lanewise(VectorOperators.LSHR, 4).or(qhV.lanewise(VectorOperators.LSHR, 6).and((byte) 3).lanewise(VectorOperators.LSHL, 4)).sub((byte) 32);
                    float ds0 = d * readByte(w, scOff + h * 8 + c);
                    float ds1 = d * readByte(w, scOff + h * 8 + 2 + c);
                    float ds2 = d * readByte(w, scOff + h * 8 + 4 + c);
                    float ds3 = d * readByte(w, scOff + h * 8 + 6 + c);
                    var ds0Vec = FloatVector.broadcast(F_SPECIES, ds0);
                    var ds1Vec = FloatVector.broadcast(F_SPECIES, ds1);
                    var ds2Vec = FloatVector.broadcast(F_SPECIES, ds2);
                    var ds3Vec = FloatVector.broadcast(F_SPECIES, ds3);
                    long sg0Idx = base + c * 16;
                    long sg1Idx = base + 32 + c * 16;
                    long sg2Idx = base + 64 + c * 16;
                    long sg3Idx = base + 96 + c * 16;
                    switch (F_SPECIES.vectorBitSize()) {
                        case 512 -> {
                            var q0f = q0.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var q1f = q1.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var q2f = q2.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var q3f = q3.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            acc0 = q0f.mul(ds0Vec).fma(av(a, aBase, sg0Idx), acc0);
                            acc1 = q1f.mul(ds1Vec).fma(av(a, aBase, sg1Idx), acc1);
                            acc2 = q2f.mul(ds2Vec).fma(av(a, aBase, sg2Idx), acc2);
                            acc3 = q3f.mul(ds3Vec).fma(av(a, aBase, sg3Idx), acc3);
                        }
                        case 256 -> {
                            for (int p = 0; p < 2; p++) {
                                int off = p * F_SPECIES.length();
                                var q0f = q0.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var q1f = q1.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var q2f = q2.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var q3f = q3.castShape(F_SPECIES, p).reinterpretAsFloats();
                                acc0 = q0f.mul(ds0Vec).fma(av(a, aBase, sg0Idx + off), acc0);
                                acc1 = q1f.mul(ds1Vec).fma(av(a, aBase, sg1Idx + off), acc1);
                                acc2 = q2f.mul(ds2Vec).fma(av(a, aBase, sg2Idx + off), acc2);
                                acc3 = q3f.mul(ds3Vec).fma(av(a, aBase, sg3Idx + off), acc3);
                            }
                        }
                        case 128 -> {
                            for (int p = 0; p < 4; p++) {
                                int off = p * F_SPECIES.length();
                                var q0f = q0.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var q1f = q1.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var q2f = q2.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var q3f = q3.castShape(F_SPECIES, p).reinterpretAsFloats();
                                acc0 = q0f.mul(ds0Vec).fma(av(a, aBase, sg0Idx + off), acc0);
                                acc1 = q1f.mul(ds1Vec).fma(av(a, aBase, sg1Idx + off), acc1);
                                acc2 = q2f.mul(ds2Vec).fma(av(a, aBase, sg2Idx + off), acc2);
                                acc3 = q3f.mul(ds3Vec).fma(av(a, aBase, sg3Idx + off), acc3);
                            }
                        }
                        default -> throw new UnsupportedOperationException(F_SPECIES.toString());
                    }
                }
            }
        }
        result += acc0.add(acc1).add(acc2.add(acc3)).reduceLanes(VectorOperators.ADD);
        if (j < size) result += scalarDot(w, wOff + j, a, aBase, aOff + j, size - j);
        return result;
    }

    /** Decode element {@code index} of a Q6_K weight (the getFloat formula). */
    static float decode(MemorySegment w, long index) {
        long blockIndex = index / BLOCK;
        int withinBlock = (int) (index % BLOCK);
        long blockOffset = blockIndex * TYPE;
        long qlOff = blockOffset;
        long qhOff = blockOffset + 128;
        long scOff = blockOffset + 192;
        float d = readFloat16(w, blockOffset + 208);
        int half = withinBlock / 128;
        int rem128 = withinBlock % 128;
        int sub32 = rem128 / 32;
        int l = rem128 % 32;
        long qlBase = qlOff + half * 64;
        long qhBase = qhOff + half * 32;
        int qlNibble, qhShift;
        switch (sub32) {
            case 0 -> { qlNibble = Byte.toUnsignedInt(readByte(w, qlBase + l)) & 0xF; qhShift = 0; }
            case 1 -> { qlNibble = Byte.toUnsignedInt(readByte(w, qlBase + 32 + l)) & 0xF; qhShift = 2; }
            case 2 -> { qlNibble = (Byte.toUnsignedInt(readByte(w, qlBase + l)) >> 4) & 0xF; qhShift = 4; }
            case 3 -> { qlNibble = (Byte.toUnsignedInt(readByte(w, qlBase + 32 + l)) >> 4) & 0xF; qhShift = 6; }
            default -> throw new IllegalStateException();
        }
        int qhBits = (Byte.toUnsignedInt(readByte(w, qhBase + l)) >> qhShift) & 3;
        int q6 = (qlNibble | (qhBits << 4)) - 32;
        int sc = readByte(w, scOff + half * 8 + sub32 * 2 + l / 16);
        return d * sc * q6;
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
