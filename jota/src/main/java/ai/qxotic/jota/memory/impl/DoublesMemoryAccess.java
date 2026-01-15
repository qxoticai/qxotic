package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryAccessChecks;

final class DoublesMemoryAccess implements MemoryAccess<double[]> {

    private static final DoublesMemoryAccess INSTANCE = new DoublesMemoryAccess();

    public static MemoryAccess<double[]> instance() {
        return INSTANCE;
    }

    private DoublesMemoryAccess() {
    }

    @Override
    public byte readByte(Memory<double[]> memory, long byteOffset) {
        throw new UnsupportedOperationException();
    }

    @Override
    public short readShort(Memory<double[]> memory, long byteOffset) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int readInt(Memory<double[]> memory, long byteOffset) {
        throw new UnsupportedOperationException();
    }

    @Override
    public float readFloat(Memory<double[]> memory, long byteOffset) {
        throw new UnsupportedOperationException();
    }

    @Override
    public long readLong(Memory<double[]> memory, long byteOffset) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Double.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Double.BYTES);
        return Double.doubleToRawLongBits(memory.base()[(int) (byteOffset / Double.BYTES)]);
    }

    @Override
    public double readDouble(Memory<double[]> memory, long byteOffset) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Double.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Double.BYTES);
        return memory.base()[(int) (byteOffset / Double.BYTES)];
    }

    @Override
    public void writeByte(Memory<double[]> memory, long byteOffset, byte value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void writeShort(Memory<double[]> memory, long byteOffset, short value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void writeInt(Memory<double[]> memory, long byteOffset, int value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void writeFloat(Memory<double[]> memory, long byteOffset, float value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void writeLong(Memory<double[]> memory, long byteOffset, long value) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Double.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Double.BYTES);
        MemoryAccessChecks.checkWriteable(memory);
        memory.base()[(int) (byteOffset / Double.BYTES)] = Double.longBitsToDouble(value);
    }

    @Override
    public void writeDouble(Memory<double[]> memory, long byteOffset, double value) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Double.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Double.BYTES);
        MemoryAccessChecks.checkWriteable(memory);
        memory.base()[(int) (byteOffset / Double.BYTES)] = value;
    }
}
