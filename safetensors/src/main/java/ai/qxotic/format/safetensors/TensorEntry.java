package ai.qxotic.format.safetensors;

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
     * Returns the absolute byte offset where this tensor's data begins in the safetensors file.
     *
     * <p>This is a convenience method equivalent to {@code safetensors.getTensorDataOffset() +
     * this.byteOffset()}.
     *
     * <p>Example usage when reading tensor data:
     *
     * <pre>{@code
     * TensorEntry tensor = safetensors.getTensor("weights");
     * long absoluteOffset = tensor.absoluteOffset(safetensors);
     * // Use absoluteOffset to read from file channel
     * }</pre>
     *
     * @param safetensors the Safetensors instance containing this tensor
     * @return the absolute byte offset in the file
     */
    public long absoluteOffset(Safetensors safetensors) {
        Objects.requireNonNull(safetensors, "safetensors");
        return safetensors.getTensorDataOffset() + this.byteOffset;
    }

    /**
     * @return total number of elements implied by {@link #shape()}
     */
    public long totalNumberOfElements() {
        return DType.totalNumberOfElements(shape);
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
}
