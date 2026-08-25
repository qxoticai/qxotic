package com.qxotic.jota.memory.impl;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.Memory;
import java.lang.reflect.Array;
import java.util.Objects;

final class ArrayMemory<B> implements Memory<B> {

    private static final Device DEVICE = DeviceType.JAVA.deviceIndex(0);

    private final B array;
    private final long byteSize;
    private final int elementByteSize;

    private ArrayMemory(B array) {
        this.array = Objects.requireNonNull(array);
        this.elementByteSize = elementByteSize(array.getClass());
        this.byteSize = Math.multiplyExact(Array.getLength(array), (long) elementByteSize);
    }

    static <B> Memory<B> of(B array) {
        return new ArrayMemory<>(array);
    }

    static int elementByteSize(Class<?> arrayType) {
        if (arrayType == boolean[].class || arrayType == byte[].class) {
            return Byte.BYTES;
        }
        if (arrayType == short[].class) {
            return Short.BYTES;
        }
        if (arrayType == int[].class) {
            return Integer.BYTES;
        }
        if (arrayType == long[].class) {
            return Long.BYTES;
        }
        if (arrayType == float[].class) {
            return Float.BYTES;
        }
        if (arrayType == double[].class) {
            return Double.BYTES;
        }
        throw new IllegalArgumentException("unsupported array type: " + arrayType.getTypeName());
    }

    @Override
    public long byteSize() {
        return byteSize;
    }

    @Override
    public boolean isReadOnly() {
        return false;
    }

    @Override
    public Device device() {
        return DEVICE;
    }

    @Override
    public B base() {
        return array;
    }

    @Override
    public long memoryGranularity() {
        return elementByteSize;
    }

    @Override
    public boolean supportsDataType(DataType dataType) {
        return array instanceof boolean[]
                ? dataType == DataType.BOOL
                : Memory.super.supportsDataType(dataType);
    }

    @Override
    public String toString() {
        return "Memory{"
                + array.getClass().getSimpleName()
                + ", byteSize="
                + byteSize
                + ", device="
                + DEVICE
                + '}';
    }
}
