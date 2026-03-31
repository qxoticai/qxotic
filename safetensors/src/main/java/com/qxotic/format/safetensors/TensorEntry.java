package com.qxotic.format.safetensors;

import java.util.Arrays;
import java.util.Objects;

/** Metadata about a tensor in a safetensors file. */
public final class TensorEntry {
    private final String name;
    private final DType dtype;
    private final long[] shape;
    private final long byteOffset;

    private TensorEntry(String name, DType dtype, long[] shape, long byteOffset) {
        this.name = name;
        this.dtype = dtype;
        this.shape = shape.clone();
        this.byteOffset = byteOffset;
    }

    /** Creates a tensor entry. */
    public static TensorEntry create(String name, DType dtype, long[] shape, long offset) {
        return new TensorEntry(
                Objects.requireNonNull(name, "name"),
                Objects.requireNonNull(dtype, "dtype"),
                Objects.requireNonNull(shape, "shape"),
                offset);
    }

    public String name() {
        return name;
    }

    public DType dtype() {
        return dtype;
    }

    /** Returns tensor shape (defensive copy). */
    public long[] shape() {
        return shape.clone();
    }

    public long byteOffset() {
        return byteOffset;
    }

    /**
     * Total number of elements (product of shape dimensions, 1 for scalars). Throws on overflow.
     */
    public long totalNumberOfElements() {
        return Arrays.stream(shape).reduce(1L, Math::multiplyExact);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof TensorEntry)) return false;
        TensorEntry that = (TensorEntry) obj;
        return byteOffset == that.byteOffset
                && Objects.equals(name, that.name)
                && dtype == that.dtype
                && Arrays.equals(shape, that.shape);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, dtype, Arrays.hashCode(shape), byteOffset);
    }

    @Override
    public String toString() {
        return "TensorEntry{name="
                + name
                + ", dtype="
                + dtype
                + ", shape="
                + Arrays.toString(shape)
                + ", offset=0x"
                + Long.toHexString(byteOffset)
                + ", byteSize="
                + byteSize()
                + '}';
    }

    public long byteSize() {
        return dtype.byteSizeForShape(shape);
    }

    /** Creates a copy with a different byte offset. */
    public TensorEntry withOffset(long newOffset) {
        return new TensorEntry(this.name, this.dtype, this.shape, newOffset);
    }
}
