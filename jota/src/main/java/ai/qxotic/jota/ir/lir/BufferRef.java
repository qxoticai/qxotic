package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.Stride;
import java.util.Objects;

/**
 * Reference to a tensor buffer with metadata.
 *
 * <p>BufferRef combines a buffer identifier with its data type and layout (shape + stride). The
 * layout may have nested structure (CuTe-style); use flat accessors for memory operations.
 *
 * <p>The layout uses element strides; use {@link #byteStrides()} to get byte strides for memory
 * access.
 */
public record BufferRef(int id, DataType dataType, Layout layout) implements LIRInput {

    public BufferRef {
        if (id < 0) {
            throw new IllegalArgumentException("Buffer id must be non-negative, got: " + id);
        }
        Objects.requireNonNull(dataType, "dataType cannot be null");
        Objects.requireNonNull(layout, "layout cannot be null");
    }

    /** Returns the shape (may be nested). */
    public Shape shape() {
        return layout.shape();
    }

    /** Returns the element strides (may be nested). */
    public Stride stride() {
        return layout.stride();
    }

    /** Returns the number of modes (top-level rank, may contain nested shapes). */
    public int rank() {
        return layout.shape().rank();
    }

    /** Returns the flattened rank (total number of dimensions for memory access). */
    public int flatRank() {
        return (int) layout.shape().flatRank();
    }

    /** Returns the total number of elements. */
    public long size() {
        return layout.shape().size();
    }

    /** Returns true if this is a scalar (0-dimensional) buffer. */
    public boolean isScalar() {
        return layout.shape().size() == 1 && layout.shape().flatRank() == 0;
    }

    /**
     * Returns the flattened byte strides for memory access.
     *
     * <p>Byte stride = element stride * dataType.byteSize(). The returned array has length equal to
     * {@link #flatRank()}.
     */
    public long[] byteStrides() {
        int rank = flatRank();
        long[] byteStrides = new long[rank];
        long byteSize = dataType.byteSize();
        for (int i = 0; i < rank; i++) {
            byteStrides[i] = layout.stride().flatAt(i) * byteSize;
        }
        return byteStrides;
    }

    /** Creates a contiguous (row-major) buffer reference. */
    public static BufferRef contiguous(int id, DataType dtype, long... shape) {
        Layout layout = Layout.rowMajor(shape);
        return new BufferRef(id, dtype, layout);
    }

    /** Creates a buffer reference from a layout. */
    public static BufferRef of(int id, DataType dtype, Layout layout) {
        return new BufferRef(id, dtype, layout);
    }

    /** Creates a scalar (0-dimensional) buffer reference. */
    public static BufferRef scalar(int id, DataType dtype) {
        return new BufferRef(id, dtype, Layout.scalar());
    }

    @Override
    public String toString() {
        return "BufferRef[id=" + id + ", dataType=" + dataType + ", layout=" + layout + "]";
    }
}
