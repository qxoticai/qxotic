package com.llm4j.huggingface;

import com.llm4j.api.BaseTensorInfo;

import java.util.Objects;

public final class HFTensorEntry implements BaseTensorInfo {
    private final String name;
    private final DType dtype;
    private final long[] shape;
    private final long offset;
    private final long size;

    public HFTensorEntry(String name, DType dtype, long[] shape, long offset, long size) {
        this.name = name;
        this.dtype = dtype;
        this.shape = shape;
        this.offset = offset;
        this.size = size;
    }

    @Override
    public DType type() {
        return dtype();
    }

    public String name() {
        return name;
    }

    public DType dtype() {
        return dtype;
    }

    @Override
    public long[] shape() {
        return shape;
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
                Objects.equals(this.shape, that.shape) &&
                this.offset == that.offset &&
                this.size == that.size;
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, dtype, shape, offset, size);
    }

    @Override
    public String toString() {
        return "HFTensorEntry[" +
                "name=" + name + ", " +
                "dtype=" + dtype + ", " +
                "shape=" + shape + ", " +
                "offset=" + offset + ", " +
                "size=" + size + ']';
    }

}

