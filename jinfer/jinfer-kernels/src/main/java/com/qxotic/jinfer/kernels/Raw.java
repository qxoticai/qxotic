package com.qxotic.jinfer.kernels;

import com.qxotic.jinfer.Segments;
import com.qxotic.jinfer.Views;
import com.qxotic.jota.DataType;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;

/**
 * The {@code (vseg, vbase)} pair a kernel runs on — the exact idiom of the old {@code
 * SegmentFloatTensor} fields: with {@link Segments#GLOBAL_SEGMENT}, {@code vseg} is the
 * all-of-memory segment and {@code vbase} the absolute byte base (view's {@code byteOffset} folded
 * in); without it, the segment itself and its plain byte offset. Valid for BOTH vector loads
 * ({@code FloatVector.fromMemorySegment(F_SPECIES, vseg, vbase + i*4, LE)}) and scalar tails
 * ({@code Segments.readFloat(vseg, vbase + i*4)}).
 *
 * <p>KERNEL-ONLY: extraction is unchecked raw access that bypasses JDK bounds and liveness. This
 * type lives in the kernels module so model code cannot compile against it - model code uses the
 * checked {@link Views#getFloat}/{@link Views#toFloatArray}/{@link Views#copyFromArray} accessors
 * or calls a kernel.
 */
public record Raw(MemorySegment vseg, long vbase) {

    /** Entry check (FP32 + row-major contiguous) + extraction, in one call. */
    public static Raw f32(MemoryView<?> view, String name) {
        return of(view, DataType.FP32, name);
    }

    /** Entry check (dtype + row-major contiguous) + extraction, in one call. */
    public static Raw of(MemoryView<?> view, DataType expected, String name) {
        Views.requireDense(view, expected, name);
        MemoryView<MemorySegment> v = Views.castToSegmentBacked(view, name);
        MemorySegment segment = v.memory().base();
        if (Segments.absoluteAddressing() && !segment.isNative()) {
            // a heap segment's address() is its offset inside the array object: absolute
            // addressing would dereference ~16, a segfault with no Java exception
            throw new IllegalArgumentException(
                    name
                            + ": heap-backed view; kernels address memory absolutely, copy it into"
                            + " a native arena first");
        }
        return new Raw(
                Segments.vectorSegment(segment), Segments.vectorBase(segment) + v.byteOffset());
    }
}
