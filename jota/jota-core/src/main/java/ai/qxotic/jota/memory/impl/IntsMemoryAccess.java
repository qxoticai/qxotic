package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryAccessChecks;

final class IntsMemoryAccess implements MemoryAccess<int[]> {

    private static final IntsMemoryAccess INSTANCE = new IntsMemoryAccess();

    public static MemoryAccess<int[]> instance() {
        return INSTANCE;
    }

    private IntsMemoryAccess() {}

    @Override
    public byte readByte(Memory<int[]> memory, long byteOffset) {
        throw new UnsupportedOperationException();
    }

    @Override
    public short readShort(Memory<int[]> memory, long byteOffset) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int readInt(Memory<int[]> memory, long byteOffset) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Integer.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Integer.BYTES);
        return memory.base()[(int) (byteOffset / Integer.BYTES)];
    }

    @Override
    public long readLong(Memory<int[]> memory, long byteOffset) {
        throw new UnsupportedOperationException();
    }

    @Override
    public double readDouble(Memory<int[]> memory, long byteOffset) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void writeByte(Memory<int[]> memory, long byteOffset, byte value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void writeShort(Memory<int[]> memory, long byteOffset, short value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void writeInt(Memory<int[]> memory, long byteOffset, int value) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Integer.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Integer.BYTES);
        MemoryAccessChecks.checkWriteable(memory);
        memory.base()[(int) (byteOffset / Integer.BYTES)] = value;
    }

    @Override
    public void writeLong(Memory<int[]> memory, long byteOffset, long value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void writeDouble(Memory<int[]> memory, long byteOffset, double value) {
        throw new UnsupportedOperationException();
    }
}
