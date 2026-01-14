package com.qxotic.jota.memory.impl;

import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAccessChecks;

final class ShortsMemoryAccess implements MemoryAccess<short[]> {

    private static final ShortsMemoryAccess INSTANCE = new ShortsMemoryAccess();

    public static MemoryAccess<short[]> instance() {
        return INSTANCE;
    }

    private ShortsMemoryAccess() {
    }

    @Override
    public byte readByte(Memory<short[]> memory, long byteOffset) {
        throw new UnsupportedOperationException();
    }

    @Override
    public short readShort(Memory<short[]> memory, long byteOffset) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Short.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Short.BYTES);
        return memory.base()[(int) (byteOffset / Short.BYTES)];
    }

    @Override
    public int readInt(Memory<short[]> memory, long byteOffset) {
        throw new UnsupportedOperationException();
    }

    @Override
    public float readFloat(Memory<short[]> memory, long byteOffset) {
        throw new UnsupportedOperationException();
    }

    @Override
    public long readLong(Memory<short[]> memory, long byteOffset) {
        throw new UnsupportedOperationException();
    }

    @Override
    public double readDouble(Memory<short[]> memory, long byteOffset) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void writeByte(Memory<short[]> memory, long byteOffset, byte value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void writeShort(Memory<short[]> memory, long byteOffset, short value) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Short.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Short.BYTES);
        MemoryAccessChecks.checkWriteable(memory);
        memory.base()[(int) (byteOffset / Short.BYTES)] = value;
    }

    @Override
    public void writeInt(Memory<short[]> memory, long byteOffset, int value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void writeFloat(Memory<short[]> memory, long byteOffset, float value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void writeLong(Memory<short[]> memory, long byteOffset, long value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void writeDouble(Memory<short[]> memory, long byteOffset, double value) {
        throw new UnsupportedOperationException();
    }
}
