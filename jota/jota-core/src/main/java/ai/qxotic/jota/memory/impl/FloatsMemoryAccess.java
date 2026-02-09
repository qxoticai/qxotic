package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryAccessChecks;

final class FloatsMemoryAccess implements MemoryAccess<float[]> {

    private static final FloatsMemoryAccess INSTANCE = new FloatsMemoryAccess();

    public static MemoryAccess<float[]> instance() {
        return INSTANCE;
    }

    private FloatsMemoryAccess() {}

    @Override
    public byte readByte(Memory<float[]> memory, long byteOffset) {
        throw new UnsupportedOperationException();
    }

    @Override
    public short readShort(Memory<float[]> memory, long byteOffset) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int readInt(Memory<float[]> memory, long byteOffset) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Float.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Float.BYTES);
        return Float.floatToRawIntBits(memory.base()[(int) (byteOffset / Float.BYTES)]);
    }

    @Override
    public float readFloat(Memory<float[]> memory, long byteOffset) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Float.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Float.BYTES);
        return memory.base()[(int) (byteOffset / Float.BYTES)];
    }

    @Override
    public long readLong(Memory<float[]> memory, long byteOffset) {
        throw new UnsupportedOperationException();
    }

    @Override
    public double readDouble(Memory<float[]> memory, long byteOffset) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void writeByte(Memory<float[]> memory, long byteOffset, byte value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void writeShort(Memory<float[]> memory, long byteOffset, short value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void writeInt(Memory<float[]> memory, long byteOffset, int value) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Float.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Float.BYTES);
        MemoryAccessChecks.checkWriteable(memory);
        memory.base()[(int) (byteOffset / Float.BYTES)] = Float.intBitsToFloat(value);
    }

    @Override
    public void writeFloat(Memory<float[]> memory, long byteOffset, float value) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Float.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Float.BYTES);
        MemoryAccessChecks.checkWriteable(memory);
        memory.base()[(int) (byteOffset / Float.BYTES)] = value;
    }

    @Override
    public void writeLong(Memory<float[]> memory, long byteOffset, long value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void writeDouble(Memory<float[]> memory, long byteOffset, double value) {
        throw new UnsupportedOperationException();
    }
}
