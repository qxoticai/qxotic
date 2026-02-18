package ai.qxotic.format.gguf;

import java.util.Arrays;
import java.util.Objects;

/**
 * Represents metadata about a tensor stored in a GGUF file.
 *
 * <p>A tensor is a multidimensional array of values used in machine learning models. This class
 * stores key information about a tensor including its name, shape, data type, and location/offset
 * with respect to {@link GGUF#getTensorDataOffset()}.
 */
public final class TensorEntry {
    /** The name identifier of the tensor. */
    private final String name;

    /**
     * The dimensions of the tensor. For example, [768, 32000] represents a 2D tensor with 768 rows
     * and 32000 columns.
     */
    private final long[] shape;

    /** The data type of the tensor elements e.g. {@link GGMLType#F32}, {@link GGMLType#Q4_0}. */
    private final GGMLType ggmlType;

    /**
     * The byte offset where this tensor's data begins with respect to {@link
     * GGUF#getTensorDataOffset()} in the GGUF file.
     */
    private final long offset;

    private TensorEntry(String name, long[] shape, GGMLType ggmlType, long offset) {
        this.name = name;
        this.shape = shape.clone();
        this.ggmlType = ggmlType;
        this.offset = offset;
    }

    /**
     * Constructs a new {@link TensorEntry} with the specified parameters.
     *
     * @param name the name identifier of the tensor
     * @param shape the dimensions of the tensor as an array of longs
     * @param ggmlType the data type of the tensor elements
     * @param offset the byte offset where this tensor's data begins in the file
     */
    public static TensorEntry create(String name, long[] shape, GGMLType ggmlType, long offset) {
        return new TensorEntry(name, shape, ggmlType, offset);
    }

    /**
     * Returns the name identifier of the tensor.
     *
     * @return the tensor name
     */
    public String name() {
        return name;
    }

    /**
     * Returns the dimensions of the tensor.
     *
     * <p>The returned array represents the size of each dimension. For example, [768, 32000]
     * represents a 2D tensor with 768 rows and 32000 columns.
     *
     * @return the tensor's shape as an array of longs
     */
    public long[] shape() {
        return shape.clone();
    }

    /**
     * Returns the data type of the tensor.
     *
     * <p>Tensors can be {@link GGMLType#isQuantized() quantized} e.g. {@link GGMLType#Q8_0}.
     *
     * @return the GGML data type
     */
    public GGMLType ggmlType() {
        return this.ggmlType;
    }

    /**
     * Alias for {@link #ggmlType()}.
     *
     * @return the GGML data type
     */
    public GGMLType type() {
        return this.ggmlType;
    }

    /**
     * Returns the byte offset where this tensor's data begins with respect to {@link
     * GGUF#getTensorDataOffset()}.
     *
     * @return the byte offset of the tensor data
     */
    public long offset() {
        return this.offset;
    }

    /**
     * Returns the absolute byte offset where this tensor's data begins in the GGUF file.
     *
     * <p>This is a convenience method equivalent to {@code gguf.getTensorDataOffset() +
     * this.offset()}.
     *
     * <p>Example usage when reading tensor data:
     *
     * <pre>{@code
     * TensorEntry tensor = gguf.getTensor("weights");
     * long absoluteOffset = tensor.absoluteOffset(gguf);
     * // Use absoluteOffset to read from file channel
     * }</pre>
     *
     * @param gguf the GGUF instance containing this tensor
     * @return the absolute byte offset in the file
     */
    public long absoluteOffset(GGUF gguf) {
        return gguf.getTensorDataOffset() + this.offset;
    }

    /**
     * Returns the byte size required to store this tensor's data.
     *
     * <p>This is a convenience method equivalent to {@code ggmlType().byteSizeForShape(shape())}.
     *
     * @return the byte size of the tensor data
     */
    public long byteSize() {
        return this.ggmlType.byteSizeForShape(this.shape);
    }

    /**
     * Compares this {@link TensorEntry} with another object for equality.
     *
     * <p>Two TensorInfo objects are considered equal if they have the same name, shape, type, and
     * offset.
     *
     * @param other the object to compare with
     * @return true if the objects are equal, false otherwise
     */
    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (other instanceof TensorEntry) {
            TensorEntry that = (TensorEntry) other;
            return offset == that.offset
                    && Objects.equals(name, that.name)
                    && Arrays.equals(shape, that.shape)
                    && ggmlType == that.ggmlType;
        } else {
            return false;
        }
    }

    /**
     * Returns a hash code value for this {@link TensorEntry}.
     *
     * @return a hash code value for this object
     */
    @Override
    public int hashCode() {
        return Objects.hash(name, Arrays.hashCode(shape), ggmlType, offset);
    }

    /**
     * Returns a string representation of this {@link TensorEntry}.
     *
     * <p>The string includes the tensor's name, shape, type, and offset in hexadecimal. For
     * example: <i>"TensorInfo{name='token_embd.weight', shape=[768, 32000], ggmlType=F32,
     * offset=0x12300}"</i>
     *
     * @return a string representation of this object
     */
    @Override
    public String toString() {
        return "TensorEntry{"
                + "name="
                + name
                + ", shape="
                + Arrays.toString(shape)
                + ", ggmlType="
                + ggmlType
                + ", offset="
                + "0x"
                + Long.toHexString(offset)
                + '}';
    }
}
