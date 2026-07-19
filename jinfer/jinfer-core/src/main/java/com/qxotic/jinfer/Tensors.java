// Tensors and compute kernels: the FloatTensor hierarchy (one class per GGML quantization)
// and the vector dot/gemm/gemv kernels. The parallel runner / CPU pinning live in Parallel.
package com.qxotic.jinfer;

import com.oracle.svm.shared.AlwaysInline;
import com.qxotic.format.gguf.GGMLType;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

final class Float16 {
    public static final int BYTES = 2;
}

/**
 * Base for tensors backed by a MemorySegment (mapped GGUF weights or native arena allocations).
 * Vector and scalar access go through (vseg, vbase + byteOffset): with GLOBAL_SEGMENT available the
 * bounds/liveness checks constant-fold (vbase = absolute address) and every access site sees a
 * single segment implementation type — required for native-image SIMD (see FloatTensor).
 */
abstract class SegmentFloatTensor extends FloatTensor {

    final long size;

    /**
     * The backing segment. Declared in EVERY leaf class, not here: a shared field would merge the
     * segment implementation types of all tensors (mapped GGUF weights, native arena buffers) into
     * one points-to set, and native-image SIMD needs each vector call site to see a single segment
     * implementation type (see FloatTensor.GLOBAL_SEGMENT). vseg/vbase are safe to share: with
     * GLOBAL_SEGMENT bound they are the same constant for everyone.
     */
    final MemorySegment vseg;

    final long vbase;

    SegmentFloatTensor(long size, MemorySegment memorySegment) {
        this.size = size;
        this.vseg = vectorSegment(memorySegment);
        this.vbase = vectorBase(memorySegment);
    }

    @Override
    public final long size() {
        return size;
    }
}

final class Q4_0FloatTensor extends SegmentFloatTensor {

    final MemorySegment memorySegment;

    public Q4_0FloatTensor(long size, MemorySegment memorySegment) {
        super(size, memorySegment);
        this.memorySegment = memorySegment;
    }

    @Override
    public void setFloat(long index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, long index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q4_0;
    }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        long blockIndex = index / GGMLType.Q4_0.getElementsPerBlock();
        long blockOffset = blockIndex * GGMLType.Q4_0.getBlockByteSize();
        float scale = readFloat16(memorySegment, blockOffset);
        byte quant;
        int modIndex = (int) (index % GGMLType.Q4_0.getElementsPerBlock());
        if (modIndex < GGMLType.Q4_0.getElementsPerBlock() / 2) {
            quant = (byte) (readByte(memorySegment, blockOffset + Float16.BYTES + modIndex) & 0x0F);
        } else {
            quant =
                    (byte)
                            ((readByte(
                                                    memorySegment,
                                                    blockOffset
                                                            + Float16.BYTES
                                                            + modIndex
                                                            - GGMLType.Q4_0.getElementsPerBlock()
                                                                    / 2)
                                            >>> 4)
                                    & 0x0F);
        }
        quant -= 8;
        return quant * scale;
    }

    @Override
    public float dot(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API && that instanceof F32FloatTensor f32) {
            return vectorDot(this, thisOffset, f32, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float vectorDot(
            Q4_0FloatTensor thiz, long thisOffset, F32FloatTensor that, long thatOffset, int size) {
        float result = 0f;
        int j = 0;

        assert Integer.bitCount(GGMLType.Q4_0.getElementsPerBlock()) == 1 : "power of 2";
        int alignmentBound =
                (int) Math.min(size, -thisOffset & (GGMLType.Q4_0.getElementsPerBlock() - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % GGMLType.Q4_0.getElementsPerBlock() == 0;

        FloatVector val = FloatVector.zero(F_SPECIES);
        long blockOffset =
                (long) (thisOffset + j)
                        / GGMLType.Q4_0.getElementsPerBlock()
                        * GGMLType.Q4_0.getBlockByteSize();
        int upperBound =
                j
                        + (size - j)
                                / GGMLType.Q4_0.getElementsPerBlock()
                                * GGMLType.Q4_0.getElementsPerBlock();
        for (;
                j < upperBound;
                j += GGMLType.Q4_0.getElementsPerBlock(),
                        blockOffset += GGMLType.Q4_0.getBlockByteSize()) {
            float wScaleValue = readFloat16(thiz.memorySegment, blockOffset);
            var wScale = FloatVector.broadcast(F_SPECIES, wScaleValue);
            var wBytes =
                    ByteVector.fromMemorySegment(
                            ByteVector.SPECIES_128,
                            thiz.memorySegment,
                            blockOffset + Float16.BYTES,
                            ByteOrder.LITTLE_ENDIAN);
            var loBytes = wBytes.and((byte) 0xF).sub((byte) 8);
            var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4).sub((byte) 8);

            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    var s0 =
                            that.getFloatVector(F_SPECIES, thatOffset + j)
                                    .mul(loBytes.castShape(F_SPECIES, 0));
                    var s1 =
                            that.getFloatVector(F_SPECIES, thatOffset + j + F_SPECIES.length())
                                    .mul(hiBytes.castShape(F_SPECIES, 0));
                    val = s0.add(s1).fma(wScale, val);
                }
                case 256 -> {
                    var s0 =
                            that.getFloatVector(F_SPECIES, thatOffset + j)
                                    .mul(loBytes.castShape(F_SPECIES, 0));
                    var s1 =
                            that.getFloatVector(F_SPECIES, thatOffset + j + 2 * F_SPECIES.length())
                                    .mul(hiBytes.castShape(F_SPECIES, 0));
                    s0 =
                            that.getFloatVector(F_SPECIES, thatOffset + j + F_SPECIES.length())
                                    .fma(loBytes.castShape(F_SPECIES, 1), s0);
                    s1 =
                            that.getFloatVector(F_SPECIES, thatOffset + j + 3 * F_SPECIES.length())
                                    .fma(hiBytes.castShape(F_SPECIES, 1), s1);
                    val = s0.add(s1).fma(wScale, val);
                }
                case 128 -> {
                    for (int i = 0; i < 2; ++i) {
                        var tmp = i == 0 ? loBytes : hiBytes;
                        var s0 =
                                that.getFloatVector(
                                                F_SPECIES,
                                                thatOffset + j + (i * 4) * F_SPECIES.length())
                                        .mul(tmp.castShape(F_SPECIES, 0));
                        var s1 =
                                that.getFloatVector(
                                                F_SPECIES,
                                                thatOffset + j + (i * 4 + 2) * F_SPECIES.length())
                                        .mul(tmp.castShape(F_SPECIES, 2));
                        s0 =
                                that.getFloatVector(
                                                F_SPECIES,
                                                thatOffset + j + (i * 4 + 1) * F_SPECIES.length())
                                        .fma(tmp.castShape(F_SPECIES, 1), s0);
                        s1 =
                                that.getFloatVector(
                                                F_SPECIES,
                                                thatOffset + j + (i * 4 + 3) * F_SPECIES.length())
                                        .fma(tmp.castShape(F_SPECIES, 3), s1);
                        val = s0.add(s1).fma(wScale, val);
                    }
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }
        result += val.reduceLanes(VectorOperators.ADD);

        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}

final class Q4_1FloatTensor extends SegmentFloatTensor {

    final MemorySegment memorySegment;

    public Q4_1FloatTensor(long size, MemorySegment memorySegment) {
        super(size, memorySegment);
        this.memorySegment = memorySegment;
    }

    @Override
    public void setFloat(long index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, long index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q4_1;
    }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        long blockIndex = index / GGMLType.Q4_1.getElementsPerBlock();
        long blockOffset = blockIndex * GGMLType.Q4_1.getBlockByteSize();
        float delta = readFloat16(memorySegment, blockOffset);
        float min = readFloat16(memorySegment, blockOffset + Float16.BYTES);
        int modIndex = (int) (index % GGMLType.Q4_1.getElementsPerBlock());
        int quant;
        if (modIndex < 16) {
            quant =
                    Byte.toUnsignedInt(
                                    readByte(
                                            memorySegment,
                                            blockOffset + 2 * Float16.BYTES + modIndex))
                            & 0x0F;
        } else {
            quant =
                    (Byte.toUnsignedInt(
                                            readByte(
                                                    memorySegment,
                                                    blockOffset
                                                            + 2 * Float16.BYTES
                                                            + modIndex
                                                            - 16))
                                    >>> 4)
                            & 0x0F;
        }
        return delta * quant + min;
    }

    @Override
    public float dot(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API && that instanceof F32FloatTensor f32) {
            return vectorDot(this, thisOffset, f32, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static final java.lang.foreign.ValueLayout.OfLong LONG_LE =
            java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private static float vectorDot(
            Q4_1FloatTensor thiz, long thisOffset, F32FloatTensor that, long thatOffset, int size) {
        float result = 0f;
        int j = 0;

        assert Integer.bitCount(GGMLType.Q4_1.getElementsPerBlock()) == 1 : "power of 2";
        int alignmentBound =
                (int) Math.min(size, -thisOffset & (GGMLType.Q4_1.getElementsPerBlock() - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % GGMLType.Q4_1.getElementsPerBlock() == 0;

        FloatVector val = FloatVector.zero(F_SPECIES);
        long blockOffset =
                (long) (thisOffset + j)
                        / GGMLType.Q4_1.getElementsPerBlock()
                        * GGMLType.Q4_1.getBlockByteSize();
        int upperBound =
                j
                        + (size - j)
                                / GGMLType.Q4_1.getElementsPerBlock()
                                * GGMLType.Q4_1.getElementsPerBlock();
        for (;
                j < upperBound;
                j += GGMLType.Q4_1.getElementsPerBlock(),
                        blockOffset += GGMLType.Q4_1.getBlockByteSize()) {
            float deltaValue = readFloat16(thiz.memorySegment, blockOffset);
            float minValue = readFloat16(thiz.memorySegment, blockOffset + Float16.BYTES);
            var wDelta = FloatVector.broadcast(F_SPECIES, deltaValue);
            var wMin = FloatVector.broadcast(F_SPECIES, minValue);
            var wBytes =
                    ByteVector.fromMemorySegment(
                            ByteVector.SPECIES_128,
                            thiz.memorySegment,
                            blockOffset + 2 * Float16.BYTES,
                            ByteOrder.LITTLE_ENDIAN);
            var loBytes = wBytes.and((byte) 0xF);
            var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4);
            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    var that0 = that.getFloatVector(F_SPECIES, thatOffset + j);
                    var that1 = that.getFloatVector(F_SPECIES, thatOffset + j + F_SPECIES.length());
                    var s0 = that0.mul(loBytes.castShape(F_SPECIES, 0));
                    var s1 = that1.mul(hiBytes.castShape(F_SPECIES, 0));
                    val = s0.add(s1).fma(wDelta, val);
                    val = that0.add(that1).fma(wMin, val);
                }
                case 256 -> {
                    var that0 = that.getFloatVector(F_SPECIES, thatOffset + j);
                    var that1 = that.getFloatVector(F_SPECIES, thatOffset + j + F_SPECIES.length());
                    var that2 =
                            that.getFloatVector(F_SPECIES, thatOffset + j + 2 * F_SPECIES.length());
                    var that3 =
                            that.getFloatVector(F_SPECIES, thatOffset + j + 3 * F_SPECIES.length());
                    var s0 = that0.mul(loBytes.castShape(F_SPECIES, 0));
                    var s1 = that2.mul(hiBytes.castShape(F_SPECIES, 0));
                    s0 = that1.fma(loBytes.castShape(F_SPECIES, 1), s0);
                    s1 = that3.fma(hiBytes.castShape(F_SPECIES, 1), s1);
                    val = s0.add(s1).fma(wDelta, val);
                    val = that0.add(that1).add(that2).add(that3).fma(wMin, val);
                }
                case 128 -> {
                    for (int i = 0; i < 2; ++i) {
                        var tmp = i == 0 ? loBytes : hiBytes;
                        var s0 =
                                that.getFloatVector(
                                                F_SPECIES,
                                                thatOffset + j + (i * 4) * F_SPECIES.length())
                                        .mul(tmp.castShape(F_SPECIES, 0));
                        var s1 =
                                that.getFloatVector(
                                                F_SPECIES,
                                                thatOffset + j + (i * 4 + 2) * F_SPECIES.length())
                                        .mul(tmp.castShape(F_SPECIES, 2));
                        s0 =
                                that.getFloatVector(
                                                F_SPECIES,
                                                thatOffset + j + (i * 4 + 1) * F_SPECIES.length())
                                        .fma(tmp.castShape(F_SPECIES, 1), s0);
                        s1 =
                                that.getFloatVector(
                                                F_SPECIES,
                                                thatOffset + j + (i * 4 + 3) * F_SPECIES.length())
                                        .fma(tmp.castShape(F_SPECIES, 3), s1);
                        val = s0.add(s1).fma(wDelta, val);
                    }
                    // vectorized min contribution
                    var thatSum = FloatVector.zero(F_SPECIES);
                    for (int k = 0;
                            k < GGMLType.Q4_1.getElementsPerBlock();
                            k += F_SPECIES.length()) {
                        thatSum = thatSum.add(that.getFloatVector(F_SPECIES, thatOffset + j + k));
                    }
                    val = thatSum.fma(wMin, val);
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }
        result += val.reduceLanes(VectorOperators.ADD);

        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}

final class Q5_1FloatTensor extends SegmentFloatTensor {

    final MemorySegment memorySegment;

    Q5_1FloatTensor(long size, MemorySegment memorySegment) {
        super(size, memorySegment);
        this.memorySegment = memorySegment;
    }

    @Override
    public void setFloat(long index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, long index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q5_1;
    }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        long blockIndex = index / GGMLType.Q5_1.getElementsPerBlock();
        int inBlockIndex = (int) (index % GGMLType.Q5_1.getElementsPerBlock());
        long blockOffset = blockIndex * GGMLType.Q5_1.getBlockByteSize();

        float d = readFloat16(memorySegment, blockOffset);
        float m = readFloat16(memorySegment, blockOffset + Float16.BYTES);
        int qh = readInt32LE(memorySegment, blockOffset + 2L * Float16.BYTES);

        int j;
        int nibble;
        int xh;
        if (inBlockIndex < GGMLType.Q5_1.getElementsPerBlock() / 2) {
            j = inBlockIndex;
            nibble =
                    Byte.toUnsignedInt(
                                    readByte(
                                            memorySegment,
                                            blockOffset + 2L * Float16.BYTES + Integer.BYTES + j))
                            & 0x0F;
            xh = ((qh >> j) << 4) & 0x10;
        } else {
            j = inBlockIndex - GGMLType.Q5_1.getElementsPerBlock() / 2;
            nibble =
                    (Byte.toUnsignedInt(
                                            readByte(
                                                    memorySegment,
                                                    blockOffset
                                                            + 2L * Float16.BYTES
                                                            + Integer.BYTES
                                                            + j))
                                    >>> 4)
                            & 0x0F;
            xh = (qh >> (j + 12)) & 0x10;
        }

        int q = nibble | xh;
        return q * d + m;
    }

    @Override
    public float dot(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API && that instanceof F32FloatTensor f32) {
            return vectorDot(this, thisOffset, f32, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static final java.lang.foreign.ValueLayout.OfLong LONG_LE =
            java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private static float vectorDot(
            Q5_1FloatTensor thiz, long thisOffset, F32FloatTensor that, long thatOffset, int size) {
        assert Integer.bitCount(GGMLType.Q5_1.getElementsPerBlock()) == 1 : "power of 2";
        int j = 0;
        float result = 0f;

        int alignmentBound =
                (int) Math.min(size, -thisOffset & (GGMLType.Q5_1.getElementsPerBlock() - 1));
        if (alignmentBound > 0) {
            result += scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j = alignmentBound;
        }

        float[] decoded = new float[GGMLType.Q5_1.getElementsPerBlock()];
        int upperBound =
                j
                        + (size - j)
                                / GGMLType.Q5_1.getElementsPerBlock()
                                * GGMLType.Q5_1.getElementsPerBlock();
        int vecUpper = F_SPECIES.loopBound(GGMLType.Q5_1.getElementsPerBlock());
        for (; j < upperBound; j += GGMLType.Q5_1.getElementsPerBlock()) {
            assert (thisOffset + j) % GGMLType.Q5_1.getElementsPerBlock() == 0;
            long blockOffset =
                    (long) (thisOffset + j)
                            / GGMLType.Q5_1.getElementsPerBlock()
                            * GGMLType.Q5_1.getBlockByteSize();
            float d = readFloat16(thiz.memorySegment, blockOffset);
            float m = readFloat16(thiz.memorySegment, blockOffset + Float16.BYTES);
            int qh = readInt32LE(thiz.memorySegment, blockOffset + 2L * Float16.BYTES);
            long qsBase = blockOffset + 2L * Float16.BYTES + Integer.BYTES;

            for (int p = 0; p < GGMLType.Q5_1.getElementsPerBlock() / 2; p++) {
                int packed = Byte.toUnsignedInt(readByte(thiz.memorySegment, qsBase + p));
                int x0 = (packed & 0x0F) | ((((qh >> p) << 4) & 0x10));
                int x1 = ((packed >>> 4) & 0x0F) | ((qh >> (p + 12)) & 0x10);
                decoded[p] = x0 * d + m;
                decoded[p + GGMLType.Q5_1.getElementsPerBlock() / 2] = x1 * d + m;
            }

            FloatVector acc = FloatVector.zero(F_SPECIES);
            for (int i = 0; i < vecUpper; i += F_SPECIES.length()) {
                FloatVector w = FloatVector.fromArray(F_SPECIES, decoded, i);
                FloatVector x = that.getFloatVector(F_SPECIES, thatOffset + j + i);
                acc = w.fma(x, acc);
            }
            result += acc.reduceLanes(VectorOperators.ADD);

            for (int i = vecUpper; i < GGMLType.Q5_1.getElementsPerBlock(); i++) {
                result += decoded[i] * that.getFloat(thatOffset + j + i);
            }
        }

        if (j < size) {
            result += scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }
        return result;
    }

    private static int readInt32LE(MemorySegment memorySegment, long offset) {
        int b0 = Byte.toUnsignedInt(readByte(memorySegment, offset));
        int b1 = Byte.toUnsignedInt(readByte(memorySegment, offset + 1));
        int b2 = Byte.toUnsignedInt(readByte(memorySegment, offset + 2));
        int b3 = Byte.toUnsignedInt(readByte(memorySegment, offset + 3));
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }
}

final class Q4_KFloatTensor extends SegmentFloatTensor {

    static final int BLOCK_SIZE = QK_K;
    static final int TYPE_SIZE = GGMLType.Q4_K.getBlockByteSize();

    final MemorySegment memorySegment;

    public Q4_KFloatTensor(long size, MemorySegment memorySegment) {
        super(size, memorySegment);
        this.memorySegment = memorySegment;
    }

    @Override
    public void setFloat(long index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, long index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q4_K;
    }

    // Decode scale or min for sub-block j (0..7) from the 12-byte scales array
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
     * The 8 sub-block scales of a Q4_K block, unpacked branch-free from the 12 packed bytes into
     * one byte-per-value long (LSB = sub-block 0); a per-row dot otherwise pays 32 branchy {@link
     * #getScaleMinK4} calls per super-block.
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

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int withinBlock = (int) (index % BLOCK_SIZE);
        long blockOffset = blockIndex * TYPE_SIZE;
        float d = readFloat16(memorySegment, blockOffset);
        float dmin = readFloat16(memorySegment, blockOffset + 2);
        long scalesOffset = blockOffset + 4;
        long qsOffset = blockOffset + 16; // 4 + 12

        // Each group of 64 values uses 2 sub-blocks: low nibble (32) + high nibble (32)
        int group = withinBlock / 64; // 0..3
        int inGroup = withinBlock % 64;
        int subBlock;
        int nibbleIndex;
        boolean isHigh;
        if (inGroup < 32) {
            subBlock = group * 2;
            nibbleIndex = inGroup;
            isHigh = false;
        } else {
            subBlock = group * 2 + 1;
            nibbleIndex = inGroup - 32;
            isHigh = true;
        }

        int sc = getScaleMinK4(subBlock, memorySegment, scalesOffset, false);
        int m = getScaleMinK4(subBlock, memorySegment, scalesOffset, true);

        byte qsByte = readByte(memorySegment, qsOffset + group * 32 + nibbleIndex);
        int quant =
                isHigh
                        ? ((Byte.toUnsignedInt(qsByte) >> 4) & 0xF)
                        : (Byte.toUnsignedInt(qsByte) & 0xF);

        return d * sc * quant - dmin * m;
    }

    @Override
    public float dot(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API && that instanceof F32FloatTensor f32) {
            return vectorDot(this, thisOffset, f32, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static final java.lang.foreign.ValueLayout.OfLong LONG_LE =
            java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private static float vectorDot(
            Q4_KFloatTensor thiz, long thisOffset, F32FloatTensor that, long thatOffset, int size) {
        float result = 0f;
        int j = 0;

        // Handle unaligned head
        assert Integer.bitCount(BLOCK_SIZE) == 1 : "power of 2";
        int alignmentBound = (int) Math.min(size, -thisOffset & (BLOCK_SIZE - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }

        FloatVector val = FloatVector.zero(F_SPECIES);
        FloatVector val2 = FloatVector.zero(F_SPECIES);
        long blockOffset = (long) (thisOffset + j) / BLOCK_SIZE * TYPE_SIZE;
        int upperBound = j + (size - j) / BLOCK_SIZE * BLOCK_SIZE;

        for (; j < upperBound; j += BLOCK_SIZE, blockOffset += TYPE_SIZE) {
            float d = readFloat16(thiz.memorySegment, blockOffset);
            float dmin = readFloat16(thiz.memorySegment, blockOffset + 2);
            long scalesOff = blockOffset + 4;
            long qsOff = blockOffset + 16;

            long packedSc = packedScales(thiz.memorySegment, scalesOff);
            long packedMn = packedMins(thiz.memorySegment, scalesOff);

            // 4 groups of 64 values each (2 sub-blocks per group: low nibble + high nibble)
            for (int g = 0; g < 4; g++) {
                float d1 = d * (int) ((packedSc >>> (16 * g)) & 0xFF);
                float negM1 = -(dmin * (int) ((packedMn >>> (16 * g)) & 0xFF));
                float d2 = d * (int) ((packedSc >>> (16 * g + 8)) & 0xFF);
                float negM2 = -(dmin * (int) ((packedMn >>> (16 * g + 8)) & 0xFF));

                var d1Vec = FloatVector.broadcast(F_SPECIES, d1);
                var negM1Vec = FloatVector.broadcast(F_SPECIES, negM1);
                var d2Vec = FloatVector.broadcast(F_SPECIES, d2);
                var negM2Vec = FloatVector.broadcast(F_SPECIES, negM2);

                long loBase = thatOffset + j + g * 64;
                long hiBase = thatOffset + j + g * 64 + 32;

                // Process 32 bytes of qs in 2 chunks of 16 bytes
                for (int c = 0; c < 2; c++) {
                    var wBytes =
                            ByteVector.fromMemorySegment(
                                    ByteVector.SPECIES_128,
                                    thiz.memorySegment,
                                    qsOff + (long) g * 32 + c * 16,
                                    ByteOrder.LITTLE_ENDIAN);
                    var loBytes = wBytes.and((byte) 0xF);
                    var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4);

                    long loIdx = loBase + c * 16;
                    long hiIdx = hiBase + c * 16;

                    switch (F_SPECIES.vectorBitSize()) {
                        case 512 -> {
                            var loQ = loBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val =
                                    loQ.fma(d1Vec, negM1Vec)
                                            .fma(that.getFloatVector(F_SPECIES, loIdx), val);
                            var hiQ = hiBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val2 =
                                    hiQ.fma(d2Vec, negM2Vec)
                                            .fma(that.getFloatVector(F_SPECIES, hiIdx), val2);
                        }
                        case 256 -> {
                            var loQ0 = loBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var loQ1 = loBytes.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            val =
                                    loQ0.fma(d1Vec, negM1Vec)
                                            .fma(that.getFloatVector(F_SPECIES, loIdx), val);
                            val2 =
                                    loQ1.fma(d1Vec, negM1Vec)
                                            .fma(
                                                    that.getFloatVector(
                                                            F_SPECIES, loIdx + F_SPECIES.length()),
                                                    val2);
                            var hiQ0 = hiBytes.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var hiQ1 = hiBytes.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            val =
                                    hiQ0.fma(d2Vec, negM2Vec)
                                            .fma(that.getFloatVector(F_SPECIES, hiIdx), val);
                            val2 =
                                    hiQ1.fma(d2Vec, negM2Vec)
                                            .fma(
                                                    that.getFloatVector(
                                                            F_SPECIES, hiIdx + F_SPECIES.length()),
                                                    val2);
                        }
                        case 128 -> {
                            for (int p = 0; p < 4; p++) {
                                var loQ = loBytes.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val =
                                        loQ.fma(d1Vec, negM1Vec)
                                                .fma(
                                                        that.getFloatVector(
                                                                F_SPECIES,
                                                                loIdx + p * F_SPECIES.length()),
                                                        val);
                                var hiQ = hiBytes.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val2 =
                                        hiQ.fma(d2Vec, negM2Vec)
                                                .fma(
                                                        that.getFloatVector(
                                                                F_SPECIES,
                                                                hiIdx + p * F_SPECIES.length()),
                                                        val2);
                            }
                        }
                        default -> throw new UnsupportedOperationException(F_SPECIES.toString());
                    }
                }
            }
        }
        result += val.add(val2).reduceLanes(VectorOperators.ADD);

        // Handle tail
        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}

final class Q5_KFloatTensor extends SegmentFloatTensor {

    static final int BLOCK_SIZE = QK_K;
    static final int TYPE_SIZE = GGMLType.Q5_K.getBlockByteSize();

    final MemorySegment memorySegment;

    public Q5_KFloatTensor(long size, MemorySegment memorySegment) {
        super(size, memorySegment);
        this.memorySegment = memorySegment;
    }

    @Override
    public void setFloat(long index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, long index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q5_K;
    }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int withinBlock = (int) (index % BLOCK_SIZE);
        long blockOffset = blockIndex * TYPE_SIZE;
        float d = readFloat16(memorySegment, blockOffset);
        float dmin = readFloat16(memorySegment, blockOffset + 2);
        long scalesOffset = blockOffset + 4;
        long qhOffset = blockOffset + 16; // 4 + 12
        long qsOffset = blockOffset + 48; // 4 + 12 + 32

        int group = withinBlock / 64;
        int inGroup = withinBlock % 64;
        boolean isHigh = inGroup >= 32;
        int l = isHigh ? inGroup - 32 : inGroup;
        int subBlock = isHigh ? group * 2 + 1 : group * 2;

        int sc = Q4_KFloatTensor.getScaleMinK4(subBlock, memorySegment, scalesOffset, false);
        int m = Q4_KFloatTensor.getScaleMinK4(subBlock, memorySegment, scalesOffset, true);

        byte qsByte = readByte(memorySegment, qsOffset + group * 32 + l);
        int nibble =
                isHigh
                        ? ((Byte.toUnsignedInt(qsByte) >> 4) & 0xF)
                        : (Byte.toUnsignedInt(qsByte) & 0xF);

        int qhBitPos = isHigh ? 2 * group + 1 : 2 * group;
        int qhBit = (Byte.toUnsignedInt(readByte(memorySegment, qhOffset + l)) >> qhBitPos) & 1;

        int quant = nibble | (qhBit << 4);
        return d * sc * quant - dmin * m;
    }

    @Override
    public float dot(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API && that instanceof F32FloatTensor f32) {
            return vectorDot(this, thisOffset, f32, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static final java.lang.foreign.ValueLayout.OfLong LONG_LE =
            java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private static float vectorDot(
            Q5_KFloatTensor thiz, long thisOffset, F32FloatTensor that, long thatOffset, int size) {
        float result = 0f;
        int j = 0;

        assert Integer.bitCount(BLOCK_SIZE) == 1 : "power of 2";
        int alignmentBound = (int) Math.min(size, -thisOffset & (BLOCK_SIZE - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }

        FloatVector val = FloatVector.zero(F_SPECIES);
        FloatVector val2 = FloatVector.zero(F_SPECIES);
        long blockOffset = (long) (thisOffset + j) / BLOCK_SIZE * TYPE_SIZE;
        int upperBound = j + (size - j) / BLOCK_SIZE * BLOCK_SIZE;

        for (; j < upperBound; j += BLOCK_SIZE, blockOffset += TYPE_SIZE) {
            float d = readFloat16(thiz.memorySegment, blockOffset);
            float dmin = readFloat16(thiz.memorySegment, blockOffset + 2);
            long scalesOff = blockOffset + 4;
            long qhOff = blockOffset + 16;
            long qsOff = blockOffset + 48;
            var qh0 =
                    ByteVector.fromMemorySegment(
                            ByteVector.SPECIES_128,
                            thiz.memorySegment,
                            qhOff,
                            ByteOrder.LITTLE_ENDIAN);
            var qh1 =
                    ByteVector.fromMemorySegment(
                            ByteVector.SPECIES_128,
                            thiz.memorySegment,
                            qhOff + 16,
                            ByteOrder.LITTLE_ENDIAN);

            for (int g = 0; g < 4; g++) {
                int loSubBlock = g * 2;
                int hiSubBlock = loSubBlock + 1;
                float d1 =
                        d
                                * Q4_KFloatTensor.getScaleMinK4(
                                        loSubBlock, thiz.memorySegment, scalesOff, false);
                float m1 =
                        dmin
                                * Q4_KFloatTensor.getScaleMinK4(
                                        loSubBlock, thiz.memorySegment, scalesOff, true);
                float d2 =
                        d
                                * Q4_KFloatTensor.getScaleMinK4(
                                        hiSubBlock, thiz.memorySegment, scalesOff, false);
                float m2 =
                        dmin
                                * Q4_KFloatTensor.getScaleMinK4(
                                        hiSubBlock, thiz.memorySegment, scalesOff, true);
                int qhBitPosLo = 2 * g;
                int qhBitPosHi = qhBitPosLo + 1;
                long groupQsOff = qsOff + (long) g * 32;
                var d1Vec = FloatVector.broadcast(F_SPECIES, d1);
                var d2Vec = FloatVector.broadcast(F_SPECIES, d2);
                var negM1Vec = FloatVector.broadcast(F_SPECIES, -m1);
                var negM2Vec = FloatVector.broadcast(F_SPECIES, -m2);

                for (int c = 0; c < 2; c++) {
                    long loBase = thatOffset + j + g * 64 + c * 16;
                    long hiBase = thatOffset + j + g * 64 + 32 + c * 16;

                    var wBytes =
                            ByteVector.fromMemorySegment(
                                    ByteVector.SPECIES_128,
                                    thiz.memorySegment,
                                    groupQsOff + c * 16L,
                                    ByteOrder.LITTLE_ENDIAN);
                    var loQ = wBytes.and((byte) 0xF);
                    var hiQ = wBytes.lanewise(VectorOperators.LSHR, 4);

                    var qhBytes = c == 0 ? qh0 : qh1;
                    loQ =
                            loQ.or(
                                    qhBytes.lanewise(VectorOperators.LSHR, qhBitPosLo)
                                            .and((byte) 1)
                                            .lanewise(VectorOperators.LSHL, 4));
                    hiQ =
                            hiQ.or(
                                    qhBytes.lanewise(VectorOperators.LSHR, qhBitPosHi)
                                            .and((byte) 1)
                                            .lanewise(VectorOperators.LSHL, 4));

                    switch (F_SPECIES.vectorBitSize()) {
                        case 512 -> {
                            var loQf = loQ.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var hiQf = hiQ.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            val =
                                    loQf.fma(d1Vec, negM1Vec)
                                            .fma(that.getFloatVector(F_SPECIES, loBase), val);
                            val2 =
                                    hiQf.fma(d2Vec, negM2Vec)
                                            .fma(that.getFloatVector(F_SPECIES, hiBase), val2);
                        }
                        case 256 -> {
                            var loQf0 = loQ.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var loQf1 = loQ.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            var hiQf0 = hiQ.castShape(F_SPECIES, 0).reinterpretAsFloats();
                            var hiQf1 = hiQ.castShape(F_SPECIES, 1).reinterpretAsFloats();
                            val =
                                    loQf0.fma(d1Vec, negM1Vec)
                                            .fma(that.getFloatVector(F_SPECIES, loBase), val);
                            val =
                                    loQf1.fma(d1Vec, negM1Vec)
                                            .fma(
                                                    that.getFloatVector(
                                                            F_SPECIES, loBase + F_SPECIES.length()),
                                                    val);
                            val2 =
                                    hiQf0.fma(d2Vec, negM2Vec)
                                            .fma(that.getFloatVector(F_SPECIES, hiBase), val2);
                            val2 =
                                    hiQf1.fma(d2Vec, negM2Vec)
                                            .fma(
                                                    that.getFloatVector(
                                                            F_SPECIES, hiBase + F_SPECIES.length()),
                                                    val2);
                        }
                        case 128 -> {
                            for (int p = 0; p < 4; p++) {
                                int off = p * F_SPECIES.length();
                                var loQf = loQ.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var hiQf = hiQ.castShape(F_SPECIES, p).reinterpretAsFloats();
                                val =
                                        loQf.fma(d1Vec, negM1Vec)
                                                .fma(
                                                        that.getFloatVector(
                                                                F_SPECIES, loBase + off),
                                                        val);
                                val2 =
                                        hiQf.fma(d2Vec, negM2Vec)
                                                .fma(
                                                        that.getFloatVector(
                                                                F_SPECIES, hiBase + off),
                                                        val2);
                            }
                        }
                        default -> throw new UnsupportedOperationException(F_SPECIES.toString());
                    }
                }
            }
        }

        result += val.add(val2).reduceLanes(VectorOperators.ADD);

        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}

final class Q6_KFloatTensor extends SegmentFloatTensor {

    static final int BLOCK_SIZE = QK_K;
    static final int TYPE_SIZE = GGMLType.Q6_K.getBlockByteSize();

    final MemorySegment memorySegment;

    public Q6_KFloatTensor(long size, MemorySegment memorySegment) {
        super(size, memorySegment);
        this.memorySegment = memorySegment;
    }

    @Override
    public void setFloat(long index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, long index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q6_K;
    }

    // Block layout: ql[128] | qh[64] | scales[16] (int8) | d (fp16)
    // 256 elements per block, 6-bit quants: 4 from ql nibble + 2 from qh

    @Override
    public float getFloat(long index) {
        long blockIndex = index / BLOCK_SIZE;
        int withinBlock = (int) (index % BLOCK_SIZE);
        long blockOffset = blockIndex * TYPE_SIZE;
        long qlOff = blockOffset;
        long qhOff = blockOffset + 128;
        long scOff = blockOffset + 192;
        float d = readFloat16(memorySegment, blockOffset + 208);

        int half = withinBlock / 128;
        int rem128 = withinBlock % 128;
        int sub32 = rem128 / 32;
        int l = rem128 % 32;

        long qlBase = qlOff + half * 64;
        long qhBase = qhOff + half * 32;

        int qlNibble, qhShift;
        switch (sub32) {
            case 0 -> {
                qlNibble = Byte.toUnsignedInt(readByte(memorySegment, qlBase + l)) & 0xF;
                qhShift = 0;
            }
            case 1 -> {
                qlNibble = Byte.toUnsignedInt(readByte(memorySegment, qlBase + 32 + l)) & 0xF;
                qhShift = 2;
            }
            case 2 -> {
                qlNibble = (Byte.toUnsignedInt(readByte(memorySegment, qlBase + l)) >> 4) & 0xF;
                qhShift = 4;
            }
            case 3 -> {
                qlNibble =
                        (Byte.toUnsignedInt(readByte(memorySegment, qlBase + 32 + l)) >> 4) & 0xF;
                qhShift = 6;
            }
            default -> throw new IllegalStateException();
        }

        int qhBits = (Byte.toUnsignedInt(readByte(memorySegment, qhBase + l)) >> qhShift) & 3;
        int q6 = (qlNibble | (qhBits << 4)) - 32;
        int sc = readByte(memorySegment, scOff + half * 8 + sub32 * 2 + l / 16); // signed int8

        return d * sc * q6;
    }

    @Override
    public float dot(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API && that instanceof F32FloatTensor f32) {
            return vectorDot(this, thisOffset, f32, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float vectorDot(
            Q6_KFloatTensor thiz, long thisOffset, F32FloatTensor that, long thatOffset, int size) {
        float result = 0f;
        int j = 0;

        assert Integer.bitCount(BLOCK_SIZE) == 1 : "power of 2";
        int alignmentBound = (int) Math.min(size, -thisOffset & (BLOCK_SIZE - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }

        // four independent accumulators, one per q-stream: a single accumulator chains four
        // dependent FMAs per iteration and stalls on FMA latency
        FloatVector acc0 = FloatVector.zero(F_SPECIES);
        FloatVector acc1 = FloatVector.zero(F_SPECIES);
        FloatVector acc2 = FloatVector.zero(F_SPECIES);
        FloatVector acc3 = FloatVector.zero(F_SPECIES);
        long blockOffset = (long) (thisOffset + j) / BLOCK_SIZE * TYPE_SIZE;
        int upperBound = j + (size - j) / BLOCK_SIZE * BLOCK_SIZE;

        for (; j < upperBound; j += BLOCK_SIZE, blockOffset += TYPE_SIZE) {
            long qlOff = blockOffset;
            long qhOff = blockOffset + 128;
            long scOff = blockOffset + 192;
            // NOTE: Deliberately avoid Float.float16ToFloat here.
            // In native-image builds, Graal can lower that intrinsic to VCVTPH2PS with
            // an illegal high XMM operand under heavy vector register pressure in Q6_K
            // vectorDot, causing a compile-time crash. Keep this software conversion
            // until the Graal backend bug is fixed.
            float d = Float.float16ToFloat(readShort(thiz.memorySegment, blockOffset + 208));

            for (int h = 0; h < 2; h++) {
                long qlBase = qlOff + h * 64;
                long qhBase = qhOff + h * 32;

                long base = thatOffset + j + h * 128;
                for (int c = 0; c < 2; c++) {
                    var qlA =
                            ByteVector.fromMemorySegment(
                                    ByteVector.SPECIES_128,
                                    thiz.memorySegment,
                                    qlBase + c * 16L,
                                    ByteOrder.LITTLE_ENDIAN);
                    var qlB =
                            ByteVector.fromMemorySegment(
                                    ByteVector.SPECIES_128,
                                    thiz.memorySegment,
                                    qlBase + 32 + c * 16L,
                                    ByteOrder.LITTLE_ENDIAN);
                    var qhV =
                            ByteVector.fromMemorySegment(
                                    ByteVector.SPECIES_128,
                                    thiz.memorySegment,
                                    qhBase + c * 16L,
                                    ByteOrder.LITTLE_ENDIAN);

                    var q0 =
                            qlA.and((byte) 0xF)
                                    .or(qhV.and((byte) 3).lanewise(VectorOperators.LSHL, 4))
                                    .sub((byte) 32);
                    var q1 =
                            qlB.and((byte) 0xF)
                                    .or(
                                            qhV.lanewise(VectorOperators.LSHR, 2)
                                                    .and((byte) 3)
                                                    .lanewise(VectorOperators.LSHL, 4))
                                    .sub((byte) 32);
                    var q2 =
                            qlA.lanewise(VectorOperators.LSHR, 4)
                                    .or(
                                            qhV.lanewise(VectorOperators.LSHR, 4)
                                                    .and((byte) 3)
                                                    .lanewise(VectorOperators.LSHL, 4))
                                    .sub((byte) 32);
                    var q3 =
                            qlB.lanewise(VectorOperators.LSHR, 4)
                                    .or(
                                            qhV.lanewise(VectorOperators.LSHR, 6)
                                                    .and((byte) 3)
                                                    .lanewise(VectorOperators.LSHL, 4))
                                    .sub((byte) 32);

                    float ds0 = d * readByte(thiz.memorySegment, scOff + h * 8 + c);
                    float ds1 = d * readByte(thiz.memorySegment, scOff + h * 8 + 2 + c);
                    float ds2 = d * readByte(thiz.memorySegment, scOff + h * 8 + 4 + c);
                    float ds3 = d * readByte(thiz.memorySegment, scOff + h * 8 + 6 + c);

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
                            acc0 =
                                    q0f.mul(ds0Vec)
                                            .fma(that.getFloatVector(F_SPECIES, sg0Idx), acc0);
                            acc1 =
                                    q1f.mul(ds1Vec)
                                            .fma(that.getFloatVector(F_SPECIES, sg1Idx), acc1);
                            acc2 =
                                    q2f.mul(ds2Vec)
                                            .fma(that.getFloatVector(F_SPECIES, sg2Idx), acc2);
                            acc3 =
                                    q3f.mul(ds3Vec)
                                            .fma(that.getFloatVector(F_SPECIES, sg3Idx), acc3);
                        }
                        case 256 -> {
                            for (int p = 0; p < 2; p++) {
                                int off = p * F_SPECIES.length();
                                var q0f = q0.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var q1f = q1.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var q2f = q2.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var q3f = q3.castShape(F_SPECIES, p).reinterpretAsFloats();
                                acc0 =
                                        q0f.mul(ds0Vec)
                                                .fma(
                                                        that.getFloatVector(
                                                                F_SPECIES, sg0Idx + off),
                                                        acc0);
                                acc1 =
                                        q1f.mul(ds1Vec)
                                                .fma(
                                                        that.getFloatVector(
                                                                F_SPECIES, sg1Idx + off),
                                                        acc1);
                                acc2 =
                                        q2f.mul(ds2Vec)
                                                .fma(
                                                        that.getFloatVector(
                                                                F_SPECIES, sg2Idx + off),
                                                        acc2);
                                acc3 =
                                        q3f.mul(ds3Vec)
                                                .fma(
                                                        that.getFloatVector(
                                                                F_SPECIES, sg3Idx + off),
                                                        acc3);
                            }
                        }
                        case 128 -> {
                            for (int p = 0; p < 4; p++) {
                                int off = p * F_SPECIES.length();
                                var q0f = q0.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var q1f = q1.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var q2f = q2.castShape(F_SPECIES, p).reinterpretAsFloats();
                                var q3f = q3.castShape(F_SPECIES, p).reinterpretAsFloats();
                                acc0 =
                                        q0f.mul(ds0Vec)
                                                .fma(
                                                        that.getFloatVector(
                                                                F_SPECIES, sg0Idx + off),
                                                        acc0);
                                acc1 =
                                        q1f.mul(ds1Vec)
                                                .fma(
                                                        that.getFloatVector(
                                                                F_SPECIES, sg1Idx + off),
                                                        acc1);
                                acc2 =
                                        q2f.mul(ds2Vec)
                                                .fma(
                                                        that.getFloatVector(
                                                                F_SPECIES, sg2Idx + off),
                                                        acc2);
                                acc3 =
                                        q3f.mul(ds3Vec)
                                                .fma(
                                                        that.getFloatVector(
                                                                F_SPECIES, sg3Idx + off),
                                                        acc3);
                            }
                        }
                        default -> throw new UnsupportedOperationException(F_SPECIES.toString());
                    }
                }
            }
        }

        result += acc0.add(acc1).add(acc2.add(acc3)).reduceLanes(VectorOperators.ADD);

        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }
}

/**
 * Q1_0: 1-bit sign quantization. Block = fp16 scale {@code d} + 16 bytes of sign bits (128
 * elements, LSB-first within each byte); element = bit set ? +d : -d. Mirrors llama.cpp's {@code
 * dequantize_row_q1_0} exactly.
 */
final class Q1_0FloatTensor extends SegmentFloatTensor {

    final MemorySegment memorySegment;

    public Q1_0FloatTensor(long size, MemorySegment memorySegment) {
        super(size, memorySegment);
        this.memorySegment = memorySegment;
    }

    @Override
    public void setFloat(long index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, long index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q1_0;
    }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / GGMLType.Q1_0.getElementsPerBlock();
        int withinBlockIndex = (int) (index % GGMLType.Q1_0.getElementsPerBlock());
        long blockOffset = blockIndex * GGMLType.Q1_0.getBlockByteSize();
        byte bits = readByte(memorySegment, blockOffset + Float16.BYTES + (withinBlockIndex / 8));
        float scale = readFloat16(memorySegment, blockOffset);
        return ((bits >> (withinBlockIndex % 8)) & 1) != 0 ? scale : -scale;
    }

    @Override
    public float dot(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API && that instanceof F32FloatTensor f32) {
            return vectorDot(this, thisOffset, f32, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static final java.lang.foreign.ValueLayout.OfLong LONG_LE =
            java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private static final VectorSpecies<Integer> I_SPECIES =
            VectorSpecies.of(int.class, F_SPECIES.vectorShape());
    private static final IntVector LANE_IOTA = IntVector.zero(I_SPECIES).addIndex(1);

    /**
     * Q1_0·F32 dot: a block's 128 sign bits (two little-endian longs) are applied branchlessly -
     * per lane, {@code ((~(bits >>> lane)) & 1) << 31} XORed onto the activation's float bits flips
     * its sign exactly where the weight bit is clear - the direct analogue of llama.cpp's AVX2
     * xor-negate kernel, against f32 activations. Sums accumulate unscaled and fold in the block
     * scale {@code d} once via fma. Lane counts of 4/8/16 all divide 64, so a bit chunk never
     * straddles the two longs.
     */
    static float vectorDot(
            Q1_0FloatTensor thiz, long thisOffset, F32FloatTensor that, long thatOffset, int size) {
        final int blockSize = GGMLType.Q1_0.getElementsPerBlock();
        final int typeSize = GGMLType.Q1_0.getBlockByteSize();
        final MemorySegment w = thiz.memorySegment;
        final MemorySegment x = that.vseg;
        final long xBase = that.vbase;
        final int lanes = F_SPECIES.length();
        float result = 0f;
        int j = 0;

        int alignmentBound = (int) Math.min(size, -thisOffset & (blockSize - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % blockSize == 0;

        long b0 = (thisOffset + j) / blockSize * (long) typeSize;
        int upperBound = j + (size - j) / blockSize * blockSize;
        FloatVector acc0 = FloatVector.zero(F_SPECIES);
        FloatVector acc1 = FloatVector.zero(F_SPECIES);
        for (; j < upperBound; j += blockSize, b0 += typeSize) {
            FloatVector vd = FloatVector.broadcast(F_SPECIES, readFloat16(w, b0));
            // MEASURED: the shared readLong helper compiles ~20% slower in this loop (17.7 vs
            // 22 Gelem/s single-thread; e2e Bonsai decode 9.4 vs 11.3 t/s) - keep the local
            // little-endian layout read.
            long lo = w.get(LONG_LE, b0 + Float16.BYTES);
            long hi = w.get(LONG_LE, b0 + Float16.BYTES + 8);
            FloatVector sum0 = FloatVector.zero(F_SPECIES);
            FloatVector sum1 = FloatVector.zero(F_SPECIES);
            for (int g = 0; g < blockSize; g += 2 * lanes) {
                int g1 = g + lanes;
                IntVector sign0 =
                        IntVector.broadcast(I_SPECIES, (int) (g < 64 ? lo >>> g : hi >>> (g - 64)))
                                .lanewise(VectorOperators.LSHR, LANE_IOTA)
                                .not()
                                .and(1)
                                .lanewise(VectorOperators.LSHL, 31);
                IntVector sign1 =
                        IntVector.broadcast(
                                        I_SPECIES, (int) (g1 < 64 ? lo >>> g1 : hi >>> (g1 - 64)))
                                .lanewise(VectorOperators.LSHR, LANE_IOTA)
                                .not()
                                .and(1)
                                .lanewise(VectorOperators.LSHL, 31);
                FloatVector x0 =
                        FloatVector.fromMemorySegment(
                                F_SPECIES,
                                x,
                                xBase + 4L * (thatOffset + j + g),
                                ByteOrder.LITTLE_ENDIAN);
                FloatVector x1 =
                        FloatVector.fromMemorySegment(
                                F_SPECIES,
                                x,
                                xBase + 4L * (thatOffset + j + g1),
                                ByteOrder.LITTLE_ENDIAN);
                sum0 =
                        sum0.add(
                                x0.reinterpretAsInts()
                                        .lanewise(VectorOperators.XOR, sign0)
                                        .reinterpretAsFloats());
                sum1 =
                        sum1.add(
                                x1.reinterpretAsInts()
                                        .lanewise(VectorOperators.XOR, sign1)
                                        .reinterpretAsFloats());
            }
            acc0 = sum0.fma(vd, acc0);
            acc1 = sum1.fma(vd, acc1);
        }
        result += acc0.add(acc1).reduceLanes(VectorOperators.ADD);
        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }
        return result;
    }
}

final class Q8_0FloatTensor extends SegmentFloatTensor {

    final MemorySegment memorySegment;

    public Q8_0FloatTensor(long size, MemorySegment memorySegment) {
        super(size, memorySegment);
        this.memorySegment = memorySegment;
    }

    @Override
    public void setFloat(long index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, long index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q8_0;
    }

    @Override
    public float getFloat(long index) {
        long blockIndex = index / GGMLType.Q8_0.getElementsPerBlock();
        long withinBlockIndex = index % GGMLType.Q8_0.getElementsPerBlock();
        long blockOffset = blockIndex * GGMLType.Q8_0.getBlockByteSize();
        byte quant = readByte(memorySegment, blockOffset + Float16.BYTES + withinBlockIndex);
        float scale = readFloat16(memorySegment, blockOffset);
        return quant * scale;
    }

    @Override
    public float dot(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API && that instanceof F32FloatTensor f32) {
            if (F_SPECIES.vectorBitSize() == 512) {
                return vectorDot512F32(this, thisOffset, f32, thatOffset, size);
            }
            return vectorDot(this, thisOffset, f32, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    /**
     * 512-bit Q8_0·F32 dot (the decode/gemv path behind {@link #dot}): two blocks per iteration
     * with a pair of accumulators, weights decoded once (sign-extend + scale) and combined in a
     * mul/fma tree. The register-tiled prefill GEMM that used to sit here was relocated to {@code
     * com.qxotic.jam.Q8Kernel}.
     */
    static float vectorDot512F32(
            Q8_0FloatTensor thiz, long thisOffset, F32FloatTensor that, long thatOffset, int size) {
        final int blockSize = GGMLType.Q8_0.getElementsPerBlock();
        final int typeSize = GGMLType.Q8_0.getBlockByteSize();
        final MemorySegment w = thiz.memorySegment;
        final MemorySegment x = that.vseg;
        final long xBase = that.vbase;
        float result = 0f;
        int j = 0;

        int alignmentBound = (int) Math.min(size, -thisOffset & (blockSize - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % blockSize == 0;

        long b0 = (long) (thisOffset + j) / blockSize * typeSize;
        int upperBound = j + (size - j) / blockSize * blockSize;
        FloatVector c0 = FloatVector.zero(F_SPECIES);
        FloatVector c1 = FloatVector.zero(F_SPECIES);
        for (; j + blockSize < upperBound; j += 2 * blockSize, b0 += 2 * typeSize) {
            var vd0 = FloatVector.broadcast(F_SPECIES, readFloat16(w, b0));
            var vd1 = FloatVector.broadcast(F_SPECIES, readFloat16(w, b0 + typeSize));
            var w00 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    w,
                                                    b0 + Float16.BYTES,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(vd0);
            var w01 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    w,
                                                    b0 + Float16.BYTES + 16,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(vd0);
            var w10 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    w,
                                                    b0 + typeSize + Float16.BYTES,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(vd1);
            var w11 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    w,
                                                    b0 + typeSize + Float16.BYTES + 16,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(vd1);
            c0 =
                    c0.add(
                            w01.fma(
                                    FloatVector.fromMemorySegment(
                                            F_SPECIES,
                                            x,
                                            xBase + 4L * (thatOffset + j + 16),
                                            ByteOrder.LITTLE_ENDIAN),
                                    w00.mul(
                                            FloatVector.fromMemorySegment(
                                                    F_SPECIES,
                                                    x,
                                                    xBase + 4L * (thatOffset + j),
                                                    ByteOrder.LITTLE_ENDIAN))));
            c1 =
                    c1.add(
                            w11.fma(
                                    FloatVector.fromMemorySegment(
                                            F_SPECIES,
                                            x,
                                            xBase + 4L * (thatOffset + j + blockSize + 16),
                                            ByteOrder.LITTLE_ENDIAN),
                                    w10.mul(
                                            FloatVector.fromMemorySegment(
                                                    F_SPECIES,
                                                    x,
                                                    xBase + 4L * (thatOffset + j + blockSize),
                                                    ByteOrder.LITTLE_ENDIAN))));
        }
        result += c0.reduceLanes(VectorOperators.ADD) + c1.reduceLanes(VectorOperators.ADD);
        // remainder block uses a fresh accumulator (see vectorDot512)
        for (; j < upperBound; j += blockSize, b0 += typeSize) {
            var vd0 = FloatVector.broadcast(F_SPECIES, readFloat16(w, b0));
            var w00 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    w,
                                                    b0 + Float16.BYTES,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(vd0);
            var w01 =
                    ((FloatVector)
                                    ByteVector.fromMemorySegment(
                                                    ByteVector.SPECIES_128,
                                                    w,
                                                    b0 + Float16.BYTES + 16,
                                                    ByteOrder.LITTLE_ENDIAN)
                                            .castShape(F_SPECIES, 0))
                            .mul(vd0);
            result +=
                    w01.fma(
                                    FloatVector.fromMemorySegment(
                                            F_SPECIES,
                                            x,
                                            xBase + 4L * (thatOffset + j + 16),
                                            ByteOrder.LITTLE_ENDIAN),
                                    w00.mul(
                                            FloatVector.fromMemorySegment(
                                                    F_SPECIES,
                                                    x,
                                                    xBase + 4L * (thatOffset + j),
                                                    ByteOrder.LITTLE_ENDIAN)))
                            .reduceLanes(VectorOperators.ADD);
        }

        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }

    private static float vectorDot(
            Q8_0FloatTensor thiz, long thisOffset, F32FloatTensor that, long thatOffset, int size) {
        float result = 0f;
        int j = 0;

        assert Integer.bitCount(GGMLType.Q8_0.getElementsPerBlock()) == 1 : "power of 2";
        int alignmentBound =
                (int) Math.min(size, -thisOffset & (GGMLType.Q8_0.getElementsPerBlock() - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % GGMLType.Q8_0.getElementsPerBlock() == 0;

        FloatVector val = FloatVector.zero(F_SPECIES);
        long blockOffset =
                (long) (thisOffset + j)
                        / GGMLType.Q8_0.getElementsPerBlock()
                        * GGMLType.Q8_0.getBlockByteSize();
        int upperBound =
                j
                        + (size - j)
                                / GGMLType.Q8_0.getElementsPerBlock()
                                * GGMLType.Q8_0.getElementsPerBlock();
        for (;
                j < upperBound;
                j += GGMLType.Q8_0.getElementsPerBlock(),
                        blockOffset += GGMLType.Q8_0.getBlockByteSize()) {
            val = q8BlockFma(thiz, blockOffset, that, thatOffset + j, val);
        }
        result += val.reduceLanes(VectorOperators.ADD);

        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }

    static FloatVector q8BlockFma(
            Q8_0FloatTensor thiz,
            long blockOffset,
            F32FloatTensor x,
            long xOffset,
            FloatVector acc) {
        float wScaleValue = readFloat16(thiz.memorySegment, blockOffset);
        var wScale = FloatVector.broadcast(F_SPECIES, wScaleValue);
        return switch (F_SPECIES.vectorBitSize()) {
            case 512 -> {
                // two 128-bit loads with part-0 casts: C2 does not intrinsify castShape with part
                // != 0
                var w0 =
                        ByteVector.fromMemorySegment(
                                ByteVector.SPECIES_128,
                                thiz.memorySegment,
                                blockOffset + Float16.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
                var w1 =
                        ByteVector.fromMemorySegment(
                                ByteVector.SPECIES_128,
                                thiz.memorySegment,
                                blockOffset + Float16.BYTES + 16,
                                ByteOrder.LITTLE_ENDIAN);
                var s0 = x.getFloatVector(F_SPECIES, xOffset).mul(w0.castShape(F_SPECIES, 0));
                var s1 =
                        x.getFloatVector(F_SPECIES, xOffset + F_SPECIES.length())
                                .mul(w1.castShape(F_SPECIES, 0));
                yield s0.add(s1).fma(wScale, acc);
            }
            case 256 -> {
                var wBytes =
                        ByteVector.fromMemorySegment(
                                ByteVector.SPECIES_256,
                                thiz.memorySegment,
                                blockOffset + Float16.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
                var s0 = x.getFloatVector(F_SPECIES, xOffset).mul(wBytes.castShape(F_SPECIES, 0));
                var s1 =
                        x.getFloatVector(F_SPECIES, xOffset + 2 * F_SPECIES.length())
                                .mul(wBytes.castShape(F_SPECIES, 2));
                s0 =
                        x.getFloatVector(F_SPECIES, xOffset + F_SPECIES.length())
                                .fma(wBytes.castShape(F_SPECIES, 1), s0);
                s1 =
                        x.getFloatVector(F_SPECIES, xOffset + 3 * F_SPECIES.length())
                                .fma(wBytes.castShape(F_SPECIES, 3), s1);
                yield s0.add(s1).fma(wScale, acc);
            }
            case 128 -> {
                FloatVector val = acc;
                for (int i = 0; i < 2; ++i) {
                    int off = i * 16;
                    var wBytes =
                            ByteVector.fromMemorySegment(
                                    ByteVector.SPECIES_128,
                                    thiz.memorySegment,
                                    blockOffset
                                            + Float16.BYTES
                                            + i * ByteVector.SPECIES_128.vectorByteSize(),
                                    ByteOrder.LITTLE_ENDIAN);
                    var s0 =
                            x.getFloatVector(F_SPECIES, xOffset + off)
                                    .mul(wBytes.castShape(F_SPECIES, 0));
                    var s1 =
                            x.getFloatVector(F_SPECIES, xOffset + off + 2 * F_SPECIES.length())
                                    .mul(wBytes.castShape(F_SPECIES, 2));
                    s0 =
                            x.getFloatVector(F_SPECIES, xOffset + off + F_SPECIES.length())
                                    .fma(wBytes.castShape(F_SPECIES, 1), s0);
                    s1 =
                            x.getFloatVector(F_SPECIES, xOffset + off + 3 * F_SPECIES.length())
                                    .fma(wBytes.castShape(F_SPECIES, 3), s1);
                    val = s0.add(s1).fma(wScale, val);
                }
                yield val;
            }
            default -> throw new UnsupportedOperationException(F_SPECIES.toString());
        };
    }
}

final class MXFP4FloatTensor extends SegmentFloatTensor {

    private static final int[] MXFP4_VALUES = {
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
    };

    final MemorySegment memorySegment;

    MXFP4FloatTensor(long size, MemorySegment memorySegment) {
        super(size, memorySegment);
        this.memorySegment = memorySegment;
    }

    @Override
    public void setFloat(long index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, long index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.MXFP4;
    }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        long blockIndex = index / QK_MXFP4;
        int inBlockIndex = (int) (index % QK_MXFP4);
        long blockOffset = blockIndex * GGMLType.MXFP4.getBlockByteSize();

        int e8m0 = Byte.toUnsignedInt(readByte(memorySegment, blockOffset));
        float d = e8m0ToFp32Half(e8m0);

        long qsOffset = blockOffset + Byte.BYTES + (inBlockIndex & 0x0F);
        int packed = Byte.toUnsignedInt(readByte(memorySegment, qsOffset));
        int nibble = inBlockIndex < (QK_MXFP4 / 2) ? (packed & 0x0F) : ((packed >>> 4) & 0x0F);

        return MXFP4_VALUES[nibble] * d;
    }

    @Override
    public float dot(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API && that instanceof F32FloatTensor f32) {
            return vectorDot(this, thisOffset, f32, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static final java.lang.foreign.ValueLayout.OfLong LONG_LE =
            java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private static float vectorDot(
            MXFP4FloatTensor thiz,
            long thisOffset,
            F32FloatTensor that,
            long thatOffset,
            int size) {
        assert Integer.bitCount(QK_MXFP4) == 1 : "power of 2";
        int j = 0;
        float result = 0f;

        int alignmentBound = (int) Math.min(size, -thisOffset & (QK_MXFP4 - 1));
        if (alignmentBound > 0) {
            result += scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j = alignmentBound;
        }

        int upperBound = j + (size - j) / QK_MXFP4 * QK_MXFP4;
        // The per-block scale d is folded into the decoded coeff vectors (all 32 weights in a block
        // share
        // it), so the cross-block products accumulate in a single vector and reduce ONCE at the end
        // —
        // dropping the ~one horizontal reduceLanes per block the old per-block-sum path paid.
        FloatVector acc = FloatVector.zero(F_SPECIES);
        for (; j < upperBound; j += QK_MXFP4) {
            assert (thisOffset + j) % QK_MXFP4 == 0;
            long blockOffset =
                    (long) (thisOffset + j) / QK_MXFP4 * GGMLType.MXFP4.getBlockByteSize();
            float d = e8m0ToFp32Half(Byte.toUnsignedInt(readByte(thiz.memorySegment, blockOffset)));

            ByteVector packed =
                    ByteVector.fromMemorySegment(
                            ByteVector.SPECIES_128,
                            thiz.memorySegment,
                            blockOffset + Byte.BYTES,
                            ByteOrder.LITTLE_ENDIAN);
            ByteVector lo = mxfp4Decode(packed.and((byte) 0x0F));
            ByteVector hi = mxfp4Decode(packed.lanewise(VectorOperators.LSHR, 4));

            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    FloatVector loC = ((FloatVector) lo.castShape(F_SPECIES, 0)).mul(d);
                    FloatVector hiC = ((FloatVector) hi.castShape(F_SPECIES, 0)).mul(d);
                    FloatVector xLo = that.getFloatVector(F_SPECIES, thatOffset + j);
                    FloatVector xHi = that.getFloatVector(F_SPECIES, thatOffset + j + QK_MXFP4 / 2);
                    acc = loC.fma(xLo, hiC.fma(xHi, acc));
                }
                case 256 -> {
                    FloatVector lo0 = ((FloatVector) lo.castShape(F_SPECIES, 0)).mul(d);
                    FloatVector lo1 = ((FloatVector) lo.castShape(F_SPECIES, 1)).mul(d);
                    FloatVector hi0 = ((FloatVector) hi.castShape(F_SPECIES, 0)).mul(d);
                    FloatVector hi1 = ((FloatVector) hi.castShape(F_SPECIES, 1)).mul(d);
                    FloatVector x0 = that.getFloatVector(F_SPECIES, thatOffset + j);
                    FloatVector x1 =
                            that.getFloatVector(F_SPECIES, thatOffset + j + F_SPECIES.length());
                    FloatVector x2 = that.getFloatVector(F_SPECIES, thatOffset + j + QK_MXFP4 / 2);
                    FloatVector x3 =
                            that.getFloatVector(
                                    F_SPECIES, thatOffset + j + QK_MXFP4 / 2 + F_SPECIES.length());
                    acc = lo0.fma(x0, lo1.fma(x1, hi0.fma(x2, hi1.fma(x3, acc))));
                }
                case 128 -> {
                    for (int p = 0; p < 4; p++) {
                        FloatVector loPart = ((FloatVector) lo.castShape(F_SPECIES, p)).mul(d);
                        FloatVector hiPart = ((FloatVector) hi.castShape(F_SPECIES, p)).mul(d);
                        FloatVector xLo =
                                that.getFloatVector(
                                        F_SPECIES, thatOffset + j + p * F_SPECIES.length());
                        FloatVector xHi =
                                that.getFloatVector(
                                        F_SPECIES,
                                        thatOffset + j + QK_MXFP4 / 2 + p * F_SPECIES.length());
                        acc = loPart.fma(xLo, hiPart.fma(xHi, acc));
                    }
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }
        result += acc.reduceLanes(VectorOperators.ADD);

        if (j < size) {
            result += scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }
        return result;
    }

    /**
     * Register-tiled MXFP4 GEMM (F32 activations, 512-bit Vector API). A group of {@code MR} output
     * rows is decoded ONCE into a thread-local F32 scratch (nibble unpack via {@code vpshufb} +
     * scale), then a decode-free F32 MRxNR band sweeps the sequence columns FMA-ing straight from
     * the scratch. Decoding to a scratch fully amortizes the unpack over the whole row (vs the
     * {@code dot} path's re-decode per column) and frees the FMA ports of the per-block code->float
     * conversion + scale; the MR-row band then loads each activation column ONCE and feeds all 3
     * rows, cutting activation-load traffic ~3x vs a 1-row tile. The 3x3 shape is bounded twice:
     * MR*NR + MR + NR = 15 vectors fit the 16 zmm, AND the 3 dequantized rows must stay L1-resident
     * — 3 rows of dim1=2880 is 34.5 KB (vs a 32 KB L1); a 4-row band (46 KB) thrashes and
     * collapses, a 2x4 band (23 KB) under-reuses. Parallel over row groups; trailing rows (dim0 %
     * 3) and remainder columns fall back to per-column dots; non-tileable shapes use the generic
     * dot loop.
     */
    private static final int MXFP4_MR = 3, MXFP4_NR = 3;

    /**
     * Per-worker F32 scratch holding the row group's dequantized weights; grown on demand, reused.
     */
    private static final ThreadLocal<float[]> DEQUANT_BAND = new ThreadLocal<>();

    /* package-shared: generic F32 dequant-band machinery, reused by NVFP4FloatTensor (operates on float[]). */
    static float[] bandScratch(int need) {
        float[] w = DEQUANT_BAND.get();
        if (w == null || w.length < need) {
            w = new float[need];
            DEQUANT_BAND.set(w);
        }
        return w;
    }

    /**
     * Flat F32 dot of a dequantized weight row (at {@code w[wOffset..]}) against one activation
     * column.
     */
    static float dotDeq(float[] w, int wOffset, int dim1, F32FloatTensor x, int xbase) {
        FloatVector acc = FloatVector.zero(F_SPECIES);
        int len = F_SPECIES.length();
        for (int k = 0; k < dim1; k += len) {
            acc =
                    FloatVector.fromArray(F_SPECIES, w, wOffset + k)
                            .fma(x.getFloatVector(F_SPECIES, xbase + k), acc);
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    /**
     * MXFP4 e2m1 magnitudes (signed) indexed by 4-bit code; 16 entries fit a single in-register
     * byte table for {@link #mxfp4Decode}. Loaded per block (a vector cannot live in a field).
     */
    private static final byte[] MXFP4_LUT = {
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
    };

    /**
     * Decode 16 nibble codes (0..15 per byte lane) to their signed MXFP4 values via one in-register
     * table permute (compiles to {@code vpshufb}), replacing the per-lane arithmetic
     * reconstruction. Callers {@code castShape} the desired part to float and apply the block scale
     * {@code d}.
     */
    private static ByteVector mxfp4Decode(ByteVector nibbles) {
        return ByteVector.fromArray(ByteVector.SPECIES_128, MXFP4_LUT, 0)
                .rearrange(nibbles.toShuffle());
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

/**
 * NVFP4 (NVIDIA FP4), GGUF block_nvfp4 = { uint8_t d[4] (UE4M3 per-16 sub-block); uint8_t qs[32] }
 * = 36 bytes, 64 elements (4 sub-blocks of 16). value = kvalues_mxfp4[nibble] · ue4m3(d[s]); no
 * global scale. Within sub-block s, byte (s*8+j): low nibble = element s*16+j, high nibble =
 * element s*16+8+j. Matches jam (the int8 path, used for prefill/decode when the native lib is
 * present) and ggml's dequant. The dot here is the scalar floor (correct fallback); jam carries the
 * vectorized weight when loaded.
 */
final class NVFP4FloatTensor extends SegmentFloatTensor {

    private static final int[] NVFP4_VALUES = {
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
    };
    static final int QK_NVFP4 = 64;

    final MemorySegment memorySegment;

    NVFP4FloatTensor(long size, MemorySegment memorySegment) {
        super(size, memorySegment);
        this.memorySegment = memorySegment;
    }

    @Override
    public void setFloat(long index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, long index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.NVFP4;
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

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        long blockOffset = (index / QK_NVFP4) * GGMLType.NVFP4.getBlockByteSize();
        int withinBlock = (int) (index % QK_NVFP4);
        int sub = withinBlock / 16, local = withinBlock % 16; // sub-block + element within it
        float d =
                ue4m3ToFp32(
                        Byte.toUnsignedInt(readByte(memorySegment, blockOffset + sub))); // d[sub]
        long qsOffset = blockOffset + 4 + sub * 8 + (local & 7); // byte s*8 + (j mod 8)
        int packed = Byte.toUnsignedInt(readByte(memorySegment, qsOffset));
        int nibble = local < 8 ? (packed & 0x0F) : ((packed >>> 4) & 0x0F);
        return NVFP4_VALUES[nibble] * d;
    }

    @Override
    public float dot(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API
                && that instanceof F32FloatTensor x
                && (thisOffset % QK_NVFP4) == 0
                && (size % QK_NVFP4) == 0) {
            float[] w =
                    MXFP4FloatTensor.bandScratch(size); // decode the row, then a vectorized F32 dot
            dequantizeRow(this, thisOffset, size, w, 0);
            return MXFP4FloatTensor.dotDeq(w, 0, size, x, (int) thatOffset);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    /**
     * Dequantize one weight row (dim1 % 64 == 0) into dst[dstOffset..] in element order. Scalar
     * decode — the GGUF t|t+8 sub-block packing doesn't vectorize cleanly; in the gemm it is
     * amortized over the band.
     */
    private static void dequantizeRow(
            NVFP4FloatTensor thiz, long rowElemOffset, int dim1, float[] dst, int dstOffset) {
        int kblocks = dim1 / QK_NVFP4;
        long firstBlock = rowElemOffset / QK_NVFP4;
        long blockByteSize = GGMLType.NVFP4.getBlockByteSize();
        for (int blk = 0; blk < kblocks; blk++) {
            long bo = (firstBlock + blk) * blockByteSize;
            int base = dstOffset + blk * QK_NVFP4;
            for (int s = 0; s < 4; s++) {
                float d = ue4m3ToFp32(Byte.toUnsignedInt(readByte(thiz.memorySegment, bo + s)));
                for (int j = 0; j < 8; j++) {
                    int packed =
                            Byte.toUnsignedInt(readByte(thiz.memorySegment, bo + 4 + s * 8 + j));
                    dst[base + s * 16 + j] = NVFP4_VALUES[packed & 0x0F] * d; // low  -> elem j
                    dst[base + s * 16 + 8 + j] =
                            NVFP4_VALUES[packed >>> 4] * d; // high -> elem j + 8
                }
            }
        }
    }
}

final class BF16FloatTensor extends SegmentFloatTensor {

    final MemorySegment memorySegment;

    public BF16FloatTensor(long size, MemorySegment memorySegment) {
        super(size, memorySegment);
        this.memorySegment = memorySegment;
    }

    @Override
    public void setFloat(long index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, long index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.BF16;
    }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        short bits = readShort(memorySegment, (long) index * 2);
        return Float.intBitsToFloat(bits << 16);
    }

    @Override
    public float dot(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API && that instanceof F32FloatTensor f32) {
            return vectorDot(this, thisOffset, f32, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static final java.lang.foreign.ValueLayout.OfLong LONG_LE =
            java.lang.foreign.ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private static float vectorDot(
            BF16FloatTensor thiz, long thisOffset, F32FloatTensor that, long thatOffset, int size) {
        assert S_SPECIES_HALF.length() == F_SPECIES.length();
        FloatVector val = FloatVector.zero(F_SPECIES);
        int upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            FloatVector thatVector = that.getFloatVector(F_SPECIES, thatOffset + i);
            ShortVector bfloat16 =
                    ShortVector.fromMemorySegment(
                            S_SPECIES_HALF,
                            thiz.memorySegment,
                            (thisOffset + i) * 2L,
                            ByteOrder.LITTLE_ENDIAN);
            FloatVector thizVector =
                    bfloat16.castShape(I_SPECIES, 0)
                            .lanewise(VectorOperators.LSHL, 16)
                            .reinterpretAsFloats();
            val = thizVector.fma(thatVector, val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        if (upperBound < size) {
            result +=
                    scalarDot(
                            thiz,
                            thisOffset + upperBound,
                            that,
                            thatOffset + upperBound,
                            size - upperBound);
        }
        return result;
    }
}

final class F16FloatTensor extends SegmentFloatTensor {

    final MemorySegment memorySegment;

    public F16FloatTensor(long size, MemorySegment memorySegment) {
        super(size, memorySegment);
        this.memorySegment = memorySegment;
    }

    static F16FloatTensor allocate(int... dims) {
        int n = FloatTensor.numberOfElements(dims);
        MemorySegment segment = Arena.ofAuto().allocate((long) n * 2);
        return new F16FloatTensor(n, segment);
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, long index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.F16;
    }

    @Override
    public long copyRawTo(long elemOffset, MemorySegment dst, long dstByteOffset, long elemCount) {
        MemorySegment.copy(memorySegment, elemOffset * 2, dst, dstByteOffset, elemCount * 2);
        return elemCount * 2;
    }

    @Override
    public long copyRawFrom(
            MemorySegment src, long srcByteOffset, long elemOffset, long elemCount) {
        MemorySegment.copy(src, srcByteOffset, memorySegment, elemOffset * 2, elemCount * 2);
        return elemCount * 2;
    }

    @AlwaysInline(
            "hot Vector API helper: escaping FloatVector boxes per call (see hotspot_compiler)")
    static FloatVector f16ToF32Vector(MemorySegment memSeg, long byteOffset) {
        ShortVector bits16 =
                ShortVector.fromMemorySegment(
                        S_SPECIES_HALF, memSeg, byteOffset, ByteOrder.LITTLE_ENDIAN);
        var bits32 = bits16.castShape(I_SPECIES, 0).reinterpretAsInts();
        var zeroExponentMask = bits32.and(0x7C00).neg().lanewise(VectorOperators.ASHR, 31);
        bits32 =
                bits32.and(0x8000)
                        .lanewise(VectorOperators.LSHL, 16)
                        .or(
                                bits32.and(0x7FFF)
                                        .add(0x1C000)
                                        .lanewise(VectorOperators.LSHL, 13)
                                        .and(zeroExponentMask));
        return bits32.reinterpretAsFloats();
    }

    @Override
    public float getFloat(long index) {
        assert 0 <= index && index < size;
        return readFloat16(memorySegment, index * 2);
    }

    @Override
    public void setFloat(long index, float value) {
        writeShort(memorySegment, (long) index * 2, Float.floatToFloat16(value));
    }

    @Override
    public float dot(long thisOffset, FloatTensor that, long thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API && that instanceof F32FloatTensor f32) {
            return vectorDotF32(this, thisOffset, f32, thatOffset, size);
        }
        return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
    }

    private static float vectorDotF32(
            F16FloatTensor thiz, long thisOffset, F32FloatTensor that, long thatOffset, int size) {
        assert S_SPECIES_HALF.length() == F_SPECIES.length();
        FloatVector val = FloatVector.zero(F_SPECIES);
        int upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            FloatVector thatVector =
                    FloatVector.fromMemorySegment(
                            F_SPECIES,
                            that.vseg,
                            that.vbase + (long) (thatOffset + i) * Float.BYTES,
                            ByteOrder.LITTLE_ENDIAN);
            // f16ToF32Vector inlined by hand for C2 (see vectorDot below)
            var bits32 =
                    ShortVector.fromMemorySegment(
                                    S_SPECIES_HALF,
                                    thiz.vseg,
                                    thiz.vbase + (thisOffset + i) * 2L,
                                    ByteOrder.LITTLE_ENDIAN)
                            .castShape(I_SPECIES, 0)
                            .reinterpretAsInts();
            var zeroExponentMask = bits32.and(0x7C00).neg().lanewise(VectorOperators.ASHR, 31);
            FloatVector thizVector =
                    bits32.and(0x8000)
                            .lanewise(VectorOperators.LSHL, 16)
                            .or(
                                    bits32.and(0x7FFF)
                                            .add(0x1C000)
                                            .lanewise(VectorOperators.LSHL, 13)
                                            .and(zeroExponentMask))
                            .reinterpretAsFloats();
            val = thizVector.fma(thatVector, val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        if (upperBound < size) {
            result +=
                    scalarDot(
                            thiz,
                            thisOffset + upperBound,
                            that,
                            thatOffset + upperBound,
                            size - upperBound);
        }
        return result;
    }
}
