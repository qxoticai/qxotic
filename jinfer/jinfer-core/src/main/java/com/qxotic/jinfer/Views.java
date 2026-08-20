package com.qxotic.jinfer;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.MemoryFactory;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * View factories and the entry-check helpers every kernel runs first.
 *
 * <p>Kernel contract (the migration replaces FloatTensor's two implicit safety nets - virtual
 * {@code getFloat} dequantization and {@code instanceof F32} guards - with explicit checks):
 *
 * <ol>
 *   <li><b>dtype</b>: {@link #requireDatatype} (identity {@code ==} on the jota singletons). Scalar
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

    /** Fresh writable FP32 view over a native-segment allocation. */
    public static MemoryView<MemorySegment> allocateF32(
            MemoryAllocator<MemorySegment> allocator, long... dims) {
        return allocate(allocator, DataType.FP32, dims);
    }

    /** Fresh writable FP16 view over a native-segment allocation (KV cache). */
    public static MemoryView<MemorySegment> allocateF16(
            MemoryAllocator<MemorySegment> allocator, long... dims) {
        return allocate(allocator, DataType.FP16, dims);
    }

    /** Fresh FP32 view holding a copy of {@code values} (constants baked at load time). */
    public static MemoryView<MemorySegment> fromFloatArray(
            MemoryAllocator<MemorySegment> allocator, float[] values) {
        MemoryView<MemorySegment> view = allocateF32(allocator, values.length);
        view.memory()
                .base()
                .asSlice(view.byteOffset(), (long) values.length * Float.BYTES)
                .copyFrom(MemorySegment.ofArray(values));
        return view;
    }

    /**
     * Checked bulk write of {@code values[from..from+count)} into {@code view} at {@code
     * elementOffset} (preprocessing fills a native view from decoded heap arrays). Same guarantees
     * as {@link #getFloat}: plain segment copy with JDK bounds and liveness checks.
     */
    public static void copyFromArray(
            MemoryView<MemorySegment> view,
            long elementOffset,
            float[] values,
            int from,
            int count,
            String name) {
        requireF32(view, name);
        requireContiguous(view, name);
        checkAlive(view, name);
        if (elementOffset < 0 || elementOffset + count > view.shape().size()) {
            throw new IndexOutOfBoundsException(
                    name
                            + ": ["
                            + elementOffset
                            + ", "
                            + (elementOffset + count)
                            + ") out of "
                            + view.shape().size());
        }
        view.memory()
                .base()
                .asSlice(view.byteOffset() + elementOffset * Float.BYTES)
                .copyFrom(
                        MemorySegment.ofArray(values)
                                .asSlice((long) from * Float.BYTES, (long) count * Float.BYTES));
    }

    private static MemoryView<MemorySegment> allocate(
            MemoryAllocator<MemorySegment> allocator, DataType dtype, long... dims) {
        Shape shape = Shape.flat(dims);
        Memory<MemorySegment> memory = allocator.allocateMemory(dtype, shape, ALIGN);
        return MemoryView.of(memory, 0, dtype, Layout.rowMajor(shape));
    }

    /** Wrap an existing segment (e.g. a weights mmap slice) as a typed view. */
    public static MemoryView<MemorySegment> wrap(
            MemorySegment segment, DataType dtype, Shape shape) {
        return MemoryView.of(
                MemoryFactory.ofMemorySegment(segment), 0, dtype, Layout.rowMajor(shape));
    }

    public static void requireF32(MemoryView<?> view, String name) {
        requireDatatype(view, DataType.FP32, name);
    }

    public static void requireDatatype(MemoryView<?> view, DataType expected, String name) {
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
        requireDatatype(view, expected, name);
        requireContiguous(view, name);
    }

    /**
     * Fail-fast on freed backing memory before raw reads ({@link Segments#GLOBAL_SEGMENT} access
     * bypasses JDK liveness checks). Ports call this once per forward on their weight/KV views.
     */
    public static void checkAlive(MemoryView<MemorySegment> view, String name) {
        if (!view.memory().base().scope().isAlive()) {
            throw new IllegalStateException(
                    name
                            + ": backing memory already freed - the arena holding these weights"
                            + " was closed while the model is still borrowing them (close your"
                            + " arena LAST). This canary catches the sequential mistake; freeing"
                            + " DURING a request stays a data race.");
        }
    }

    /** Element-offset → absolute byte offset, for kernel entry extraction. */
    public static long byteOffset(MemoryView<MemorySegment> view, long elementOffset) {
        return view.byteOffset() + elementOffset * view.dataType().byteSize();
    }

    /**
     * Checked scalar FP32 read, for model code that needs ONE element (a logit, a learned scalar).
     * Unlike kernel access this keeps the JDK's bounds and liveness checks (plain {@code
     * segment.get} on the real segment, never {@link Segments#GLOBAL_SEGMENT}); it is deliberately
     * not the hot-path idiom - kernels use {@code Raw.f32} (jinfer-kernels).
     */
    public static float getFloat(MemoryView<MemorySegment> view, long elementOffset, String name) {
        requireF32(view, name);
        requireContiguous(view, name);
        checkAlive(view, name);
        if (elementOffset < 0 || elementOffset >= view.shape().size()) {
            throw new IndexOutOfBoundsException(
                    name + ": element offset " + elementOffset + " out of " + view.shape().size());
        }
        return view.memory()
                .base()
                .get(ValueLayout.JAVA_FLOAT, view.byteOffset() + elementOffset * Float.BYTES);
    }

    /**
     * Checked FP32 readback to a heap array (decode heads, parity tests). Same guarantees as {@link
     * #getFloat}.
     */
    public static float[] toFloatArray(MemoryView<MemorySegment> view, String name) {
        requireF32(view, name);
        requireContiguous(view, name);
        checkAlive(view, name);
        return view.memory()
                .base()
                .asSlice(view.byteOffset(), view.shape().size() * Float.BYTES)
                .toArray(ValueLayout.JAVA_FLOAT);
    }

    /**
     * Checked bulk read of {@code view[elementOffset..elementOffset+count)} into {@code
     * dst[dstOff..dstOff+count)} (conv tap staging, stats readback). Companion of {@link
     * #copyFromArray}; same guarantees as {@link #getFloat}.
     */
    public static void copyToArray(
            MemoryView<MemorySegment> view,
            long elementOffset,
            float[] dst,
            int dstOff,
            int count,
            String name) {
        requireF32(view, name);
        requireContiguous(view, name);
        checkAlive(view, name);
        if (elementOffset < 0 || elementOffset + count > view.shape().size()) {
            throw new IndexOutOfBoundsException(
                    name
                            + ": ["
                            + elementOffset
                            + ", "
                            + (elementOffset + count)
                            + ") out of "
                            + view.shape().size());
        }
        if (dstOff < 0 || dstOff + count > dst.length) {
            throw new IndexOutOfBoundsException(
                    name + ": dst [" + dstOff + ", " + (dstOff + count) + ") out of " + dst.length);
        }
        MemorySegment.copy(
                view.memory().base(),
                ValueLayout.JAVA_FLOAT,
                view.byteOffset() + elementOffset * Float.BYTES,
                dst,
                dstOff,
                count);
    }

    /**
     * Checked CAST of an opaque boundary view ({@code MemoryView<?>}) to its segment-backed reality
     * — same reference, no wrap, no copy, fail-fast. The wildcard at the boundary exists for a
     * future non-segment backing (a GPU buffer); today every weight mmap and state scratch in the
     * slice IS MemorySegment-backed, so this is the ONE sanctioned downcast — everywhere else the
     * compiler already knows. (ponytail: when a non-segment backing lands, this seam grows a second
     * arm rather than the cast spreading back out.)
     */
    @SuppressWarnings("unchecked")
    public static MemoryView<MemorySegment> castToSegmentBacked(MemoryView<?> view, String name) {
        if (!(view.memory().base() instanceof MemorySegment)) {
            throw new IllegalArgumentException(
                    name
                            + ": expected a MemorySegment-backed view but was backed by "
                            + view.memory().base().getClass().getName());
        }
        return (MemoryView<MemorySegment>) view;
    }

    /**
     * Split a 3D {@code [groups, rows, cols]} view along its leading axis into {@code groups}
     * zero-copy 2D {@code [rows, cols]} views. This is the common per-expert (or per-head,
     * per-batch) un-stacking step: the leading axis is a logical outer axis, the trailing two are
     * kept and flattened, and no element is copied or reordered.
     *
     * <p>Safety is enforced at the physical-vs-logical boundary: the view must be contiguous and
     * suffix-contiguous on axis 0, so slicing the outer axis leaves each trailing {@code [rows,
     * cols]} span row-major; the innermost (blocked) dtype axis is untouched and stays a physical
     * axis in the result (a quantized weight becomes {@code [rows, cols/elementsPerBlock]}).
     *
     * <p>This is a generic view primitive - it knows nothing about experts or any model tensor
     * naming. Model loaders call it with their stacked weights and read the result by index.
     */
    public static MemoryView<MemorySegment>[] sliceLeadingAxis(MemoryView<MemorySegment> stacked) {
        requireContiguous(stacked, "stacked");
        if (stacked.shape().flatRank() != 3)
            throw new IllegalArgumentException(
                    "stacked: expected a 3D [groups, rows, cols/elementsPerBlock] view but was "
                            + stacked.shape());
        long groups = stacked.shape().flatAt(0);
        long rows = stacked.shape().flatAt(1);
        long cols = stacked.shape().flatAt(2);
        if (groups <= 0 || rows <= 0 || cols <= 0)
            throw new IllegalArgumentException(
                    "stacked: expected positive [groups, rows, cols/elementsPerBlock] but was "
                            + stacked.shape());
        if (!stacked.layout().isSuffixContiguous(0))
            throw new IllegalArgumentException(
                    "stacked: leading axis is not suffix-contiguous; cannot split + flatten"
                            + " zero-copy (layout "
                            + stacked.layout()
                            + ")");
        MemoryView<MemorySegment>[] slices = new MemoryView[Math.toIntExact(groups)];
        for (int g = 0; g < slices.length; g++) {
            MemoryView<MemorySegment> slice =
                    stacked.slice(0, g, g + 1).view(Shape.flat(rows, cols));
            if (slice.shape().flatRank() != 2
                    || slice.shape().flatAt(0) != rows
                    || slice.shape().flatAt(1) != cols
                    || !slice.isRowMajorContiguous())
                throw new IllegalStateException(
                        "stacked: group " + g + " flattened to " + slice.shape());
            slices[g] = slice;
        }
        return slices;
    }
}
