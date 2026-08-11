package com.qxotic.jinfer.x;

import static com.qxotic.jinfer.x.Segments.F16_BYTES;
import static com.qxotic.jinfer.x.Segments.readByte;
import static com.qxotic.jinfer.x.Segments.readFloat;
import static com.qxotic.jinfer.x.Segments.readFloat16;
import static com.qxotic.jinfer.x.Segments.writeFloat;
import static com.qxotic.jinfer.x.Segments.writeShort;

import com.oracle.svm.shared.AlwaysInline;
import com.qxotic.jinfer.x.Views.Raw;
import com.qxotic.jota.DataType;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

/**
 * Cross-dtype span converters over views, ported byte-for-byte from the jinfer-core tensor methods
 * they replace ({@code F16FloatTensor.f16ToF32Vector}/{@code copyRow}, the {@code copyTo}
 * per-element convert path, {@code Q8_0FloatTensor.getFloat}/{@code copyRow}). Faithful first:
 * F32→F16 stays scalar (the old path was a per-element {@code floatToFloat16}), Q8_0 dequant keeps
 * the one-scale-read-per-block structure of the old {@code copyRow}.
 */
public final class Convert {

    private Convert() {}

    /** Decode F_SPECIES.length() consecutive F16 values to an F32 vector (IEEE half -> single). */
    @AlwaysInline(
            "hot Vector API helper: escaping FloatVector boxes per call (see hotspot_compiler)")
    public static FloatVector f16ToF32Vector(MemorySegment memSeg, long byteOffset) {
        ShortVector bits16 =
                ShortVector.fromMemorySegment(
                        Segments.S_SPECIES_HALF, memSeg, byteOffset, ByteOrder.LITTLE_ENDIAN);
        var bits32 = bits16.castShape(Segments.I_SPECIES, 0).reinterpretAsInts();
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
        Raw s = Views.rawF32(src, "src");
        Raw d = Views.rawF32(dst, "dst");
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
        Raw s = Views.raw(src, DataType.FP16, "src");
        Raw d = Views.rawF32(dst, "dst");
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
        Raw s = Views.rawF32(src, "src");
        Raw d = Views.raw(dst, DataType.FP16, "dst");
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
        Raw s = Views.raw(src, DataType.Q8_0, "src");
        Raw d = Views.rawF32(dst, "dst");
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
        } else if (dt == DataType.FP16) {
            f16ToF32(src, srcElemOff, dst, dstElemOff, count);
        } else if (dt == DataType.FP32) {
            copyF32(src, srcElemOff, dst, dstElemOff, count);
        } else {
            throw new UnsupportedOperationException("no ->F32 copy arm for " + dt);
        }
    }
}
