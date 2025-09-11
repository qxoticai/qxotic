package com.llm4j.jota.memory.impl;

import com.llm4j.jota.memory.Memory;
import com.llm4j.jota.memory.MemoryAccess;
import sun.misc.Unsafe;

final class UnsafeMemoryAccess implements MemoryAccess<Void> {

    private static final Unsafe UNSAFE = UnsafeAccess.get();

    private static final UnsafeMemoryAccess INSTANCE = new UnsafeMemoryAccess();

    public static MemoryAccess<Void> instance() {
        return INSTANCE;
    }

    private UnsafeMemoryAccess() {
    }

    @Override
    public byte readByte(Memory<Void> memory, long byteOffset) {
        return UNSAFE.getByte(byteOffset);
    }

    @Override
    public short readShort(Memory<Void> memory, long byteOffset) {
        return UNSAFE.getShort(byteOffset);
    }

    @Override
    public int readInt(Memory<Void> memory, long byteOffset) {
        return UNSAFE.getInt(byteOffset);
    }

    @Override
    public float readFloat(Memory<Void> memory, long byteOffset) {
        return UNSAFE.getFloat(byteOffset);
    }

    @Override
    public double readDouble(Memory<Void> memory, long byteOffset) {
        return UNSAFE.getDouble(byteOffset);
    }

    @Override
    public long readLong(Memory<Void> memory, long byteOffset) {
        return UNSAFE.getLong(byteOffset);
    }

    @Override
    public void writeByte(Memory<Void> memory, long byteOffset, byte value) {
        UNSAFE.putByte(byteOffset, value);
    }

    @Override
    public void writeShort(Memory<Void> memory, long byteOffset, short value) {
        UNSAFE.putShort(byteOffset, value);
    }

    @Override
    public void writeInt(Memory<Void> memory, long byteOffset, int value) {
        UNSAFE.putInt(byteOffset, value);
    }

    @Override
    public void writeFloat(Memory<Void> memory, long byteOffset, float value) {
        UNSAFE.putFloat(byteOffset, value);
    }

    @Override
    public void writeDouble(Memory<Void> memory, long byteOffset, double value) {
        UNSAFE.putDouble(byteOffset, value);
    }

    @Override
    public void writeLong(Memory<Void> memory, long byteOffset, long value) {
        UNSAFE.putLong(byteOffset, value);
    }
}
