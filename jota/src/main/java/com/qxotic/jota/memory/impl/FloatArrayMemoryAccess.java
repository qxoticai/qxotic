package com.qxotic.jota.memory.impl;

import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccess;
import sun.misc.Unsafe;

final class FloatArrayMemoryAccess implements MemoryAccess<float[]> {

    private static final Unsafe UNSAFE = UnsafeAccess.get();

    private static final FloatArrayMemoryAccess INSTANCE = new FloatArrayMemoryAccess();

    public static MemoryAccess<float[]> instance() {
        return INSTANCE;
    }

    private FloatArrayMemoryAccess() {
    }

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
        assert (byteOffset & 3) == 0;
        return Float.floatToRawIntBits(memory.base()[(int) (byteOffset / Float.BYTES)]);
    }

    @Override
    public float readFloat(Memory<float[]> memory, long byteOffset) {
        assert (byteOffset & 3) == 0;
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
        assert (byteOffset & 3) == 0;
        assert !memory.isReadOnly();
        memory.base()[(int) (byteOffset / Float.BYTES)] = Float.intBitsToFloat(value);
    }

    @Override
    public void writeFloat(Memory<float[]> memory, long byteOffset, float value) {
        assert (byteOffset & 3) == 0;
        assert !memory.isReadOnly();
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
