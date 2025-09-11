package com.llm4j.huggingface;

import com.llm4j.api.BaseType;

public enum DType implements BaseType {
    // Boolean type
    BOOL(1, boolean.class),
    // Unsigned byte
    U8(1, byte.class),
    // Signed byte
    I8(1, byte.class),
    // Signed integer (16-bit)
    I16(2, short.class),
    // Unsigned integer (16-bit)
    U16(2, short.class),
    // Half-precision floating point
    F16(2, short.class),
    // Brain floating point
    BF16(2, short.class),
    // Signed integer (32-bit)
    I32(4, int.class),
    // Unsigned integer (32-bit)
    U32(4, int.class),
    // Floating point (32-bit)
    F32(4, float.class),
    // Floating point (64-bit)
    F64(8, double.class),
    // Signed integer (64-bit)
    I64(8, long.class),
    // Unsigned integer (64-bit)
    U64(8, long.class);

    private final int size;
    private final Class<?> backingType;

    DType(int size, Class<?> backingType) {
        this.size = size;
        this.backingType = backingType;
    }

    public int size() {
        return size;
    }

    @Override
    public boolean isQuantized() {
        return false;
    }

    @Override
    public long byteSizeFor(long numberOfElements) {
        return numberOfElements * (long) size;
    }

    public Class<?> backingType() {
        return backingType;
    }
}