package com.qxotic.format.safetensors;

import java.util.Arrays;
import java.util.Objects;

/**
 * Metadata about a tensor in a safetensors file.
 *
 * <p>Offset is relative to the data section start (see {@link Safetensors#getTensorDataOffset()}).
 */
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

    /**
     * Creates a tensor entry.
     *
     * @param name tensor name
     * @param dtype tensor data type
     * @param shape tensor shape (empty array means scalar)
     * @param offset byte offset relative to tensor data section start
     * @return immutable tensor entry
     */
    public static TensorEntry create(String name, DType dtype, long[] shape, long offset) {
        return new TensorEntry(
                Objects.requireNonNull(name, "name"),
                Objects.requireNonNull(dtype, "dtype"),
                Objects.requireNonNull(shape, "shape"),
                offset);
    }

    /**
     * @return tensor name
     */
    public String name() {
        return name;
    }

    /**
     * @return tensor data type
     */
    public DType dtype() {
        return dtype;
    }

    /**
     * Returns tensor shape.
     *
     * @return defensive copy of the shape array
     */
    public long[] shape() {
        return shape.clone();
    }

    /**
     * @return byte offset relative to the tensor data section start
     */
    public long byteOffset() {
        return byteOffset;
    }

    /**
     * Returns the total number of elements in this tensor.
     *
     * <p>Computes the product of all dimensions in the shape. An empty shape (scalar) returns 1.
     * Throws {@link ArithmeticException} on overflow.
     *
     * @return total number of elements
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
        return "TensorEntry{"
                + "name="
                + name
                + ", dtype="
                + dtype
                + ", shape="
                + Arrays.toString(shape)
                + ", offset=0x"
                + Long.toHexString(byteOffset)
                + ", byteSize="
                + dtype.byteSizeForShape(shape)
                + '}';
    }

    /**
     * @return tensor payload size in bytes
     */
    public long byteSize() {
        return dtype.byteSizeForShape(shape);
    }

    /**
     * Creates a copy of this tensor entry with a different byte offset.
     *
     * <p>This is useful when rearranging tensors in a safetensors file. All other properties (name,
     * dtype, shape) remain unchanged.
     *
     * <p>Example usage:
     *
     * <pre>{@code
     * TensorEntry original = safetensors.getTensor("weights");
     * TensorEntry moved = original.withOffset(1024);
     * }</pre>
     *
     * @param newOffset the new byte offset relative to the tensor data section start
     * @return a new TensorEntry with the same name, dtype, and shape but different offset
     */
    public TensorEntry withOffset(long newOffset) {
        return new TensorEntry(this.name, this.dtype, this.shape, newOffset);
    }
}
