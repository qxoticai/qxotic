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

    public static TensorEntry create(
            String name, DType dtype, long[] shape, long offset) {
        return new TensorEntry(
                Objects.requireNonNull(name),
                Objects.requireNonNull(dtype),
                Objects.requireNonNull(shape),
                offset);
    }

    public String name() {
        return name;
    }

    public DType dtype() {
        return dtype;
    }

    public long[] shape() {
        return shape.clone();
    }

    public long byteOffset() {
        return byteOffset;
    }

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
        return "TensorInfo{"
                + "name=" + name
                + ", dtype=" + dtype
                + ", shape=" + Arrays.toString(shape)
                + ", offset=0x" + Long.toHexString(byteOffset)
                + ", byteSize=" + dtype.byteSizeForShape(shape)
                + '}';
    }

    public long byteSize() {
        return dtype.byteSizeForShape(shape);
    }
}
