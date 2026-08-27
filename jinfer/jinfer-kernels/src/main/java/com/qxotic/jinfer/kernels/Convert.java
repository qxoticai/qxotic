package com.qxotic.jinfer.kernels;

import static com.qxotic.jinfer.Segments.F16_BYTES;
import static com.qxotic.jinfer.Segments.F_SPECIES;
import static com.qxotic.jinfer.Segments.I_SPECIES;
import static com.qxotic.jinfer.Segments.S_SPECIES_HALF;
import static com.qxotic.jinfer.Segments.USE_VECTOR_API;
import static com.qxotic.jinfer.Segments.readByte;
import static com.qxotic.jinfer.Segments.readFloat;
import static com.qxotic.jinfer.Segments.readFloat16;
import static com.qxotic.jinfer.Segments.readShort;
import static com.qxotic.jinfer.Segments.writeFloat;
import static com.qxotic.jinfer.Segments.writeShort;

import com.oracle.svm.shared.AlwaysInline;
import com.qxotic.jinfer.Segments;
import com.qxotic.jota.BFloat16;
import com.qxotic.jota.DataType;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

/**
 * Cross-dtype span converters over views. F32→F16 stays scalar; Q8_0 dequantization reads its scale
 * once per block.
 */
public final class Convert {

    private Convert() {}

    /** Decode F_SPECIES.length() consecutive F16 values to an F32 vector (IEEE half -> single). */
    @AlwaysInline(
            "hot Vector API helper: escaping FloatVector boxes per call (see"
                    + " hotspot_compile_commands)")
    public static FloatVector f16ToF32Vector(MemorySegment memSeg, long byteOffset) {
        ShortVector bits16 =
                ShortVector.fromMemorySegment(
                        Segments.S_SPECIES_HALF, memSeg, byteOffset, ByteOrder.LITTLE_ENDIAN);
        return f16BitsToF32(bits16.castShape(Segments.I_SPECIES, 0).reinterpretAsInts());
    }

    /**
     * Exact IEEE half decode of one half per int lane (the low 16 bits): normals rebias the
     * exponent, subnormals scale their mantissa by 2^-24, Inf/NaN keep the all-ones exponent. Bit
     * identical to {@link Float#float16ToFloat} on every one of the 65536 inputs except that a NaN
     * keeps its payload instead of being quieted.
     */
    public static FloatVector f16BitsToF32(IntVector bits32) {
        IntVector exponent = bits32.and(0x7C00);
        IntVector normal = bits32.and(0x7FFF).add(0x1C000).lanewise(VectorOperators.LSHL, 13);
        IntVector subnormal =
                ((FloatVector) bits32.and(0x3FF).convert(VectorOperators.I2F, 0))
                        .mul(0x1p-24f)
                        .reinterpretAsInts();
        IntVector magnitude =
                normal.blend(subnormal, exponent.eq(0))
                        .lanewise(VectorOperators.OR, 0x7F800000, exponent.eq(0x7C00));
        return bits32.and(0x8000)
                .lanewise(VectorOperators.LSHL, 16)
                .or(magnitude)
                .reinterpretAsFloats();
    }

    /**
     * F32 → F32 over an element span (the old F32→F32 {@code copyTo}): one raw {@code
     * MemorySegment.copy}.
     */
    public static void copyF32(
            MemoryView<MemorySegment> src,
            long srcElemOff,
            MemoryView<MemorySegment> dst,
            long dstElemOff,
            long count) {
        Raw s = Raw.f32(src, "src");
        Raw d = Raw.f32(dst, "dst");
        MemorySegment.copy(
                s.vseg(),
                s.vbase() + srcElemOff * Float.BYTES,
                d.vseg(),
                d.vbase() + dstElemOff * Float.BYTES,
                count * Float.BYTES);
    }

    /**
     * F16 → F32 over an element span (the old F16→F32 {@code copyTo} / {@code copyRow}): vector
     * body plus scalar tail, exact for normals with subnormals flushed — the same converter the
     * FlashAttention F16 tiles use, so decode-first and direct paths agree bit for bit.
     */
    public static void f16ToF32(
            MemoryView<MemorySegment> src,
            long srcElemOff,
            MemoryView<MemorySegment> dst,
            long dstElemOff,
            int count) {
        Raw s = Raw.of(src, DataType.FP16, "src");
        Raw d = Raw.f32(dst, "dst");
        int i = 0;
        if (Segments.USE_VECTOR_API) {
            var sp = Segments.S_SPECIES_HALF;
            int bound = sp.loopBound(count);
            for (; i < bound; i += sp.length()) {
                FloatVector f = f16ToF32Vector(s.vseg(), s.vbase() + (srcElemOff + i) * F16_BYTES);
                f.intoMemorySegment(
                        d.vseg(),
                        d.vbase() + (dstElemOff + i) * Float.BYTES,
                        ByteOrder.LITTLE_ENDIAN);
            }
        }
        for (; i < count; i++) {
            writeFloat(
                    d.vseg(),
                    d.vbase() + (dstElemOff + i) * Float.BYTES,
                    readFloat16(s.vseg(), s.vbase() + (srcElemOff + i) * F16_BYTES));
        }
    }

    /** BF16 -> F32 over an element span. */
    public static void bf16ToF32(
            MemoryView<MemorySegment> src,
            long srcElemOff,
            MemoryView<MemorySegment> dst,
            long dstElemOff,
            int count) {
        Raw s = Raw.of(src, DataType.BF16, "src");
        Raw d = Raw.f32(dst, "dst");
        int i = 0;
        if (Segments.USE_VECTOR_API) {
            int bound = S_SPECIES_HALF.loopBound(count);
            for (; i < bound; i += S_SPECIES_HALF.length()) {
                ShortVector bits =
                        ShortVector.fromMemorySegment(
                                S_SPECIES_HALF,
                                s.vseg(),
                                s.vbase() + (srcElemOff + i) * F16_BYTES,
                                ByteOrder.LITTLE_ENDIAN);
                bits.castShape(I_SPECIES, 0)
                        .lanewise(VectorOperators.LSHL, 16)
                        .reinterpretAsFloats()
                        .intoMemorySegment(
                                d.vseg(),
                                d.vbase() + (dstElemOff + i) * Float.BYTES,
                                ByteOrder.LITTLE_ENDIAN);
            }
        }
        for (; i < count; i++) {
            writeFloat(
                    d.vseg(),
                    d.vbase() + (dstElemOff + i) * Float.BYTES,
                    BFloat16.toFloat(
                            readShort(s.vseg(), s.vbase() + (srcElemOff + i) * F16_BYTES)));
        }
    }

    /**
     * F32 → F16 over an element span (the KV-cache write path, old F32→F16 {@code copyTo}):
     * per-element {@code Float.floatToFloat16}, faithful to the old {@code setFloat} loop.
     */
    public static void f32ToF16(
            MemoryView<MemorySegment> src,
            long srcElemOff,
            MemoryView<MemorySegment> dst,
            long dstElemOff,
            int count) {
        Raw s = Raw.f32(src, "src");
        Raw d = Raw.of(dst, DataType.FP16, "dst");
        for (int i = 0; i < count; i++) {
            writeShort(
                    d.vseg(),
                    d.vbase() + (dstElemOff + i) * F16_BYTES,
                    Float.floatToFloat16(
                            readFloat(s.vseg(), s.vbase() + (srcElemOff + i) * Float.BYTES)));
        }
    }

    /**
     * Q8_0 → F32 over an element span (the embedding gather-dequant, old Q8_0 {@code copyTo} via
     * {@code copyRow}): one scale read per 32-element block, {@code byte * scale} per element —
     * bit-identical to the old per-element {@code getFloat}.
     */
    public static void dequantQ8_0(
            MemoryView<MemorySegment> src,
            long srcElemOff,
            MemoryView<MemorySegment> dst,
            long dstElemOff,
            int count) {
        Raw s = Raw.of(src, DataType.Q8_0, "src");
        Raw d = Raw.f32(dst, "dst");
        final int B = 32, BS = 34;
        int di = 0, rem = count;
        long idx = srcElemOff;
        while (rem > 0) {
            long bi = idx / B;
            int wi = (int) (idx % B);
            int chunk = Math.min(B - wi, rem);
            long bo = bi * BS;
            float scale = readFloat16(s.vseg(), s.vbase() + bo);
            long pt = bo + F16_BYTES + wi;
            for (int j = 0; j < chunk; j++) {
                writeFloat(
                        d.vseg(),
                        d.vbase() + (dstElemOff + di++) * Float.BYTES,
                        readByte(s.vseg(), s.vbase() + pt + j) * scale);
            }
            idx += chunk;
            rem -= chunk;
        }
    }

    /** MXFP4 -> F32 over an element span. */
    public static void dequantMxfp4(
            MemoryView<MemorySegment> src,
            long srcElemOff,
            MemoryView<MemorySegment> dst,
            long dstElemOff,
            int count) {
        Raw s = Raw.of(src, DataType.MXFP4, "src");
        Raw d = Raw.f32(dst, "dst");
        int written = 0;
        long index = srcElemOff;
        while (written < count) {
            long blockOffset = index / 32 * 17;
            int inBlock = (int) (index % 32);
            int chunk = Math.min(32 - inBlock, count - written);
            float scale =
                    mxfp4Scale(Byte.toUnsignedInt(readByte(s.vseg(), s.vbase() + blockOffset)));
            for (int i = 0; i < chunk; i++) {
                int lane = inBlock + i;
                int packed =
                        Byte.toUnsignedInt(
                                readByte(s.vseg(), s.vbase() + blockOffset + 1 + (lane & 15)));
                int code = lane < 16 ? packed & 15 : packed >>> 4;
                writeFloat(
                        d.vseg(),
                        d.vbase() + (dstElemOff + written + i) * Float.BYTES,
                        mxfp4Value(code) * scale);
            }
            index += chunk;
            written += chunk;
        }
    }

    private static int mxfp4Value(int code) {
        int magnitude =
                switch (code & 7) {
                    case 0 -> 0;
                    case 1 -> 1;
                    case 2 -> 2;
                    case 3 -> 3;
                    case 4 -> 4;
                    case 5 -> 6;
                    case 6 -> 8;
                    default -> 12;
                };
        return (code & 8) == 0 ? magnitude : -magnitude;
    }

    private static float mxfp4Scale(int value) {
        int bits = value < 2 ? 0x00200000 << value : (value - 1) << 23;
        return Float.intBitsToFloat(bits);
    }

    /**
     * Legacy-quant -> F32 over an element span: the old per-element {@code copyTo} (getFloat per
     * element, any alignment), sharing MatMul's scalar decoders.
     */
    public static void dequantLegacy(
            MemoryView<MemorySegment> src,
            long srcElemOff,
            MemoryView<MemorySegment> dst,
            long dstElemOff,
            int count) {
        Raw s = Raw.of(src, src.dataType(), "src");
        Raw d = Raw.f32(dst, "dst");
        DataType dt = src.dataType();
        for (int i = 0; i < count; i++) {
            writeFloat(
                    d.vseg(),
                    d.vbase() + (dstElemOff + i) * Float.BYTES,
                    MatMul.getLegacy(s.vseg(), s.vbase(), srcElemOff + i, dt));
        }
    }

    /**
     * The static heir of the old virtual {@code copyTo} for the ->F32 direction: one dtype switch
     * per span, routed to the arms above (Q8_0 dequant / F16 vector / F32 raw copy). The dispatch
     * table lives here, next to the arms it selects — a model never re-encodes it, and a cycle-2
     * dtype adds one case in one file.
     */
    public static void copyToF32(
            MemoryView<MemorySegment> src,
            long srcElemOff,
            MemoryView<MemorySegment> dst,
            long dstElemOff,
            int count) {
        DataType dt = src.dataType();
        if (dt == DataType.Q8_0) {
            dequantQ8_0(src, srcElemOff, dst, dstElemOff, count);
        } else if (dt == DataType.MXFP4) {
            dequantMxfp4(src, srcElemOff, dst, dstElemOff, count);
        } else if (dt == DataType.FP16) {
            f16ToF32(src, srcElemOff, dst, dstElemOff, count);
        } else if (dt == DataType.BF16) {
            bf16ToF32(src, srcElemOff, dst, dstElemOff, count);
        } else if (dt == DataType.FP32) {
            copyF32(src, srcElemOff, dst, dstElemOff, count);
        } else if (dt.elementsPerBlock() > 1) {
            dequantLegacy(src, srcElemOff, dst, dstElemOff, count);
        } else {
            throw new UnsupportedOperationException("copyToF32 dtype " + dt);
        }
    }

    /**
     * Batched embedding gather-dequant: {@code rows[rowsOff .. rowsOff+n)} name the table rows
     * (each {@code rowLen} elements), dequantized consecutively into {@code dst} at {@code
     * dstElemOff}. One dtype dispatch per table - the hoisted form of {@code n} per-row {@link
     * #copyToF32} calls - and the Q8_0 arm additionally vectorizes the row dequant. Every other
     * dtype falls back to the per-row spans (bit-identical either way).
     */
    public static void gatherToF32(
            MemoryView<MemorySegment> table,
            int[] rows,
            int rowsOff,
            int n,
            MemoryView<MemorySegment> dst,
            long dstElemOff,
            int rowLen) {
        if (table.dataType() == DataType.Q8_0 && rowLen % 32 == 0 && USE_VECTOR_API) {
            dequantQ8_0Rows(table, rows, rowsOff, n, dst, dstElemOff, rowLen);
            return;
        }
        for (int r = 0; r < n; r++) {
            copyToF32(
                    table,
                    (long) rows[rowsOff + r] * rowLen,
                    dst,
                    dstElemOff + (long) r * rowLen,
                    rowLen);
        }
    }

    /**
     * The Q8_0 gather arm: one scale read + one 32-byte vector dequant per block (B2I sign-extends,
     * I2F is exact for a byte, the f32 multiply matches - bit-identical to {@link #dequantQ8_0}'s
     * per-element {@code byte * scale}).
     */
    private static void dequantQ8_0Rows(
            MemoryView<MemorySegment> table,
            int[] rows,
            int rowsOff,
            int n,
            MemoryView<MemorySegment> dst,
            long dstElemOff,
            int rowLen) {
        Raw t = Raw.of(table, DataType.Q8_0, "table");
        Raw d = Raw.f32(dst, "dst");
        final int B = 32, BS = 34;
        int blocksPerRow = rowLen / B;
        int parts = B / F_SPECIES.length();
        for (int r = 0; r < n; r++) {
            long rowByte = t.vbase() + (long) rows[rowsOff + r] * blocksPerRow * BS;
            long dstByte = d.vbase() + (dstElemOff + (long) r * rowLen) * Float.BYTES;
            for (int blk = 0; blk < blocksPerRow; blk++) {
                long bo = rowByte + (long) blk * BS;
                FloatVector scale = FloatVector.broadcast(F_SPECIES, readFloat16(t.vseg(), bo));
                ByteVector q =
                        ByteVector.fromMemorySegment(
                                ByteVector.SPECIES_256,
                                t.vseg(),
                                bo + F16_BYTES,
                                ByteOrder.LITTLE_ENDIAN);
                for (int p = 0; p < parts; p++) {
                    q.convertShape(VectorOperators.B2I, I_SPECIES, p)
                            .convert(VectorOperators.I2F, 0)
                            .mul(scale)
                            .intoMemorySegment(
                                    d.vseg(),
                                    dstByte
                                            + (long) (blk * B + p * F_SPECIES.length())
                                                    * Float.BYTES,
                                    ByteOrder.LITTLE_ENDIAN);
                }
            }
        }
    }
}
