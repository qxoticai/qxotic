package com.qxotic.jota.memory.impl;

import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAccessChecks;

final class LongsMemoryAccess implements MemoryAccess<long[]> {

    private static final LongsMemoryAccess INSTANCE = new LongsMemoryAccess();

    public static MemoryAccess<long[]> instance() {
        return INSTANCE;
    }

    private LongsMemoryAccess() {}

    @Override
    public byte readByte(Memory<long[]> memory, long byteOffset) {
        throw new UnsupportedOperationException();
    }

    @Override
    public short readShort(Memory<long[]> memory, long byteOffset) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int readInt(Memory<long[]> memory, long byteOffset) {
        throw new UnsupportedOperationException();
    }

    @Override
    public float readFloat(Memory<long[]> memory, long byteOffset) {
        throw new UnsupportedOperationException();
    }

    @Override
    public long readLong(Memory<long[]> memory, long byteOffset) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Long.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Long.BYTES);
        return memory.base()[(int) (byteOffset / Long.BYTES)];
    }

    @Override
    public void writeByte(Memory<long[]> memory, long byteOffset, byte value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void writeShort(Memory<long[]> memory, long byteOffset, short value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void writeInt(Memory<long[]> memory, long byteOffset, int value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void writeFloat(Memory<long[]> memory, long byteOffset, float value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void writeLong(Memory<long[]> memory, long byteOffset, long value) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Long.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Long.BYTES);
        MemoryAccessChecks.checkWriteable(memory);
        memory.base()[(int) (byteOffset / Long.BYTES)] = value;
    }
}
