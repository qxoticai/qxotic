package com.qxotic.format.gguf;

import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable descriptor for a tensor in a GGUF file: name, shape, {@link GGMLType}, and byte offset
 * relative to {@link GGUF#getTensorDataOffset()}.
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

    /** Creates an entry; {@code offset} is relative to {@link GGUF#getTensorDataOffset()}. */
    public static TensorEntry create(String name, long[] shape, GGMLType ggmlType, long offset) {
        return new TensorEntry(name, shape, ggmlType, offset);
    }

    public String name() {
        return name;
    }

    /** Tensor dimensions, e.g. {@code [768, 32000]}; returns a defensive copy. */
    public long[] shape() {
        return shape.clone();
    }

    public GGMLType ggmlType() {
        return this.ggmlType;
    }

    /** Alias for {@link #ggmlType()}. */
    public GGMLType type() {
        return this.ggmlType;
    }

    /** Byte offset relative to {@link GGUF#getTensorDataOffset()}. */
    public long offset() {
        return this.offset;
    }

    /** Byte size of the tensor data: {@code ggmlType().byteSizeFor(totalNumberOfElements())}. */
    public long byteSize() {
        return this.ggmlType.byteSizeFor(totalNumberOfElements());
    }

    /**
     * Product of all shape dimensions (1 for an empty shape).
     *
     * @throws ArithmeticException if the result overflows
     */
    public long totalNumberOfElements() {
        long total = 1;
        for (long dim : this.shape) {
            total = Math.multiplyExact(total, dim);
        }
        return total;
    }

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

    @Override
    public int hashCode() {
        return Objects.hash(name, Arrays.hashCode(shape), ggmlType, offset);
    }

    /** Includes name, shape, type, and the offset in hexadecimal. */
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

    /** Returns a copy with a different offset; useful when rearranging tensor layout. */
    public TensorEntry withOffset(long newOffset) {
        return new TensorEntry(this.name, this.shape, this.ggmlType, newOffset);
    }
}
