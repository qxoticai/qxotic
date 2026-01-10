package com.qxotic.jota.memory.impl;

import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccess;
import sun.misc.Unsafe;

final class ByteArrayMemoryAccess implements MemoryAccess<byte[]> {

    private static final Unsafe UNSAFE = UnsafeAccess.get();

    private static final ByteArrayMemoryAccess INSTANCE = new ByteArrayMemoryAccess();

    public static MemoryAccess<byte[]> instance() {
        return INSTANCE;
    }

    private ByteArrayMemoryAccess() {
    }

    @Override
    public byte readByte(Memory<byte[]> memory, long byteOffset) {
        return UNSAFE.getByte(memory.base(), Unsafe.ARRAY_BYTE_BASE_OFFSET + byteOffset);
    }

    @Override
    public short readShort(Memory<byte[]> memory, long byteOffset) {
        return UNSAFE.getShort(memory.base(), Unsafe.ARRAY_BYTE_BASE_OFFSET + byteOffset);
    }

    @Override
    public int readInt(Memory<byte[]> memory, long byteOffset) {
        return UNSAFE.getInt(memory.base(), Unsafe.ARRAY_BYTE_BASE_OFFSET + byteOffset);
    }

    @Override
    public float readFloat(Memory<byte[]> memory, long byteOffset) {
        return UNSAFE.getFloat(memory.base(), Unsafe.ARRAY_BYTE_BASE_OFFSET + byteOffset);
    }

    @Override
    public long readLong(Memory<byte[]> memory, long byteOffset) {
        return UNSAFE.getLong(memory.base(), Unsafe.ARRAY_BYTE_BASE_OFFSET + byteOffset);
    }

    @Override
    public double readDouble(Memory<byte[]> memory, long byteOffset) {
        return UNSAFE.getDouble(memory.base(), Unsafe.ARRAY_BYTE_BASE_OFFSET + byteOffset);
    }

    @Override
    public void writeByte(Memory<byte[]> memory, long byteOffset, byte value) {
        assert !memory.isReadOnly();
        UNSAFE.putByte(memory.base(), Unsafe.ARRAY_BYTE_BASE_OFFSET + byteOffset, value);
    }

    @Override
    public void writeShort(Memory<byte[]> memory, long byteOffset, short value) {
        assert !memory.isReadOnly();
        UNSAFE.putShort(memory.base(), Unsafe.ARRAY_BYTE_BASE_OFFSET + byteOffset, value);
    }

    @Override
    public void writeInt(Memory<byte[]> memory, long byteOffset, int value) {
        assert !memory.isReadOnly();
        UNSAFE.putInt(memory.base(), Unsafe.ARRAY_BYTE_BASE_OFFSET + byteOffset, value);
    }

    @Override
    public void writeFloat(Memory<byte[]> memory, long byteOffset, float value) {
        assert !memory.isReadOnly();
        UNSAFE.putFloat(memory.base(), Unsafe.ARRAY_BYTE_BASE_OFFSET + byteOffset, value);
    }

    @Override
    public void writeLong(Memory<byte[]> memory, long byteOffset, long value) {
        assert !memory.isReadOnly();
        UNSAFE.putLong(memory.base(), Unsafe.ARRAY_BYTE_BASE_OFFSET + byteOffset, value);
    }

    @Override
    public void writeDouble(Memory<byte[]> memory, long byteOffset, double value) {
        assert !memory.isReadOnly();
        UNSAFE.putDouble(memory.base(), Unsafe.ARRAY_BYTE_BASE_OFFSET + byteOffset, value);
    }
}
