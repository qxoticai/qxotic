package ai.qxotic.format.safetensors;

import java.util.Arrays;
import java.util.Objects;

public final class HFTensorEntry {
    private final String name;
    private final DType dtype;
    private final long[] shape;
    private final long offset;
    private final long size;

    public HFTensorEntry(String name, DType dtype, long[] shape, long offset, long size) {
        this.name = Objects.requireNonNull(name, "name");
        this.dtype = Objects.requireNonNull(dtype, "dtype");
        Objects.requireNonNull(shape, "shape");
        if (shape.length == 0) {
            throw new IllegalArgumentException("shape must not be empty");
        }
        this.shape = shape.clone();
        this.offset = offset;
        this.size = size;
    }

    public DType type() {
        return dtype();
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

    public long offset() {
        return offset;
    }

    public long size() {
        return size;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == this) return true;
        if (obj == null || obj.getClass() != this.getClass()) return false;
        var that = (HFTensorEntry) obj;
        return Objects.equals(this.name, that.name) &&
                Objects.equals(this.dtype, that.dtype) &&
                Arrays.equals(this.shape, that.shape) &&
                this.offset == that.offset &&
                this.size == that.size;
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, dtype, Arrays.hashCode(shape), offset, size);
    }

    @Override
    public String toString() {
        return "HFTensorEntry[" +
                "name=" + name + ", " +
                "dtype=" + dtype + ", " +
                "shape=" + Arrays.toString(shape) + ", " +
                "offset=" + offset + ", " +
                "size=" + size + ']';
    }

}

