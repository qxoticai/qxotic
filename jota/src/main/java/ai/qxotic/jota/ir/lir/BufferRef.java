package ai.qxotic.jota.ir.lir;

import ai.qxotic.jota.DataType;
import java.util.Arrays;
import java.util.Objects;

/** Reference to a tensor buffer with metadata. */
public record BufferRef(int id, DataType dataType, long[] shape, long[] strides)
        implements LIRNode {

    public BufferRef {
        if (id < 0) {
            throw new IllegalArgumentException("Buffer id must be non-negative, got: " + id);
        }
        Objects.requireNonNull(dataType, "dataType cannot be null");
        Objects.requireNonNull(shape, "shape cannot be null");
        Objects.requireNonNull(strides, "strides cannot be null");
        if (shape.length != strides.length) {
            throw new IllegalArgumentException(
                    "shape and strides must have same length: "
                            + shape.length
                            + " vs "
                            + strides.length);
        }
        shape = shape.clone();
        strides = strides.clone();
    }

    /** Returns the number of dimensions. */
    public int rank() {
        return shape.length;
    }

    /** Returns the total number of elements. */
    public long size() {
        long size = 1;
        for (long dim : shape) {
            size *= dim;
        }
        return size;
    }

    /** Creates a contiguous (row-major) buffer reference. */
    public static BufferRef contiguous(int id, DataType dtype, long... shape) {
        long[] strides = new long[shape.length];
        long stride = dtype.byteSize();
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return new BufferRef(id, dtype, shape, strides);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof BufferRef that)) return false;
        return id == that.id
                && dataType == that.dataType
                && Arrays.equals(shape, that.shape)
                && Arrays.equals(strides, that.strides);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(id, dataType);
        result = 31 * result + Arrays.hashCode(shape);
        result = 31 * result + Arrays.hashCode(strides);
        return result;
    }

    @Override
    public String toString() {
        return "BufferRef["
                + "id="
                + id
                + ", dataType="
                + dataType
                + ", shape="
                + Arrays.toString(shape)
                + ", strides="
                + Arrays.toString(strides)
                + "]";
    }
}
