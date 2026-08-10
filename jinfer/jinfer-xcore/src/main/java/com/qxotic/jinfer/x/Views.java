package com.qxotic.jinfer.x;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.MemoryFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * View factories and the entry-check helpers every xcore kernel runs first.
 *
 * <p>Kernel contract (the migration replaces FloatTensor's two implicit safety nets - virtual
 * {@code getFloat} dequantization and {@code instanceof F32} guards - with explicit checks):
 *
 * <ol>
 *   <li><b>dtype</b>: {@link #requireDtype} (identity {@code ==} on the jota singletons). Scalar
 *       tails read raw memory as floats, which is only sound after this check.
 *   <li><b>layout</b>: {@link #requireContiguous} - kernels assume row-major dense storage.
 *   <li><b>liveness</b>: {@link #checkAlive} - raw {@link Segments#GLOBAL_SEGMENT} access bypasses
 *       JDK liveness; ports run this once per forward on the weights/KV views (the {@code
 *       safetyCanary} role), not per kernel.
 * </ol>
 */
public final class Views {

    private Views() {}

    private static final long ALIGN = 64; // cacheline, vector-friendly

    /** Fresh writable FP32 view over an arena-owned native segment. */
    public static MemoryView<MemorySegment> allocateF32(Arena arena, long... dims) {
        return allocate(arena, DataType.FP32, dims);
    }

    /** Fresh writable FP16 view over an arena-owned native segment (KV cache). */
    public static MemoryView<MemorySegment> allocateF16(Arena arena, long... dims) {
        return allocate(arena, DataType.FP16, dims);
    }

    private static MemoryView<MemorySegment> allocate(Arena arena, DataType dtype, long... dims) {
        Shape shape = Shape.flat(dims);
        MemorySegment segment = arena.allocate(dtype.byteSizeFor(shape), ALIGN);
        Memory<MemorySegment> memory = MemoryFactory.ofMemorySegment(segment);
        return MemoryView.of(memory, 0, dtype, Layout.rowMajor(shape));
    }

    /** Wrap an existing segment (e.g. a weights mmap slice) as a typed view. */
    public static MemoryView<MemorySegment> wrap(
            MemorySegment segment, DataType dtype, Shape shape) {
        return MemoryView.of(
                MemoryFactory.ofMemorySegment(segment), 0, dtype, Layout.rowMajor(shape));
    }

    public static void requireF32(MemoryView<?> view, String name) {
        requireDtype(view, DataType.FP32, name);
    }

    public static void requireDtype(MemoryView<?> view, DataType expected, String name) {
        if (view.dataType() != expected) {
            throw new IllegalArgumentException(
                    name + ": expected " + expected.name() + " but was " + view.dataType().name());
        }
    }

    public static void requireContiguous(MemoryView<?> view, String name) {
        if (!view.isRowMajorContiguous()) {
            throw new IllegalArgumentException(
                    name + ": expected row-major contiguous view but was " + view.layout());
        }
    }

    /** The standard kernel entry check: dtype + row-major contiguity. */
    public static void requireDense(MemoryView<?> view, DataType expected, String name) {
        requireDtype(view, expected, name);
        requireContiguous(view, name);
    }

    /**
     * Fail-fast on freed backing memory before raw reads ({@link Segments#GLOBAL_SEGMENT} access
     * bypasses JDK liveness checks). Ports call this once per forward on their weight/KV views.
     */
    public static void checkAlive(MemoryView<MemorySegment> view, String name) {
        if (!view.memory().base().scope().isAlive()) {
            throw new IllegalStateException(name + ": backing memory already released");
        }
    }

    /** Element-offset → absolute byte offset, for kernel entry extraction. */
    public static long byteOffset(MemoryView<MemorySegment> view, long elementOffset) {
        return view.byteOffset() + elementOffset * view.dataType().byteSize();
    }

    /**
     * The {@code (vseg, vbase)} pair a kernel runs on — the exact idiom of the old {@code
     * SegmentFloatTensor} fields: with {@link Segments#GLOBAL_SEGMENT}, {@code vseg} is the
     * all-of-memory segment and {@code vbase} the absolute byte base (view's {@code byteOffset}
     * folded in); without it, the segment itself and its plain byte offset. Valid for BOTH vector
     * loads ({@code FloatVector.fromMemorySegment(F_SPECIES, vseg, vbase + i*4, LE)}) and scalar
     * tails ({@code Segments.readFloat(vseg, vbase + i*4)}).
     */
    public record Raw(MemorySegment vseg, long vbase) {}

    /** Entry check (FP32 + row-major contiguous) + extraction, in one call. */
    public static Raw rawF32(MemoryView<MemorySegment> view, String name) {
        return raw(view, DataType.FP32, name);
    }

    /** Entry check (dtype + row-major contiguous) + extraction, in one call. */
    public static Raw raw(MemoryView<MemorySegment> view, DataType expected, String name) {
        requireDense(view, expected, name);
        MemorySegment segment = view.memory().base();
        return new Raw(
                Segments.vectorSegment(segment), Segments.vectorBase(segment) + view.byteOffset());
    }
}
