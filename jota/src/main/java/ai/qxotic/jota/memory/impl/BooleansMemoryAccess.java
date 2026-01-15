package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryAccessChecks;
import sun.misc.Unsafe;

final class BooleansMemoryAccess implements MemoryAccess<boolean[]> {

    private static final Unsafe UNSAFE = UnsafeAccess.get();

    private static final BooleansMemoryAccess INSTANCE = new BooleansMemoryAccess();

    public static MemoryAccess<boolean[]> instance() {
        return INSTANCE;
    }

    private BooleansMemoryAccess() {
    }

    @Override
    public byte readByte(Memory<boolean[]> memory, long byteOffset) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Byte.BYTES);
        boolean value = UNSAFE.getBoolean(memory.base(), Unsafe.ARRAY_BOOLEAN_BASE_OFFSET + byteOffset);
        return (byte) (value ? 1 : 0);
    }

    @Override
    public short readShort(Memory<boolean[]> memory, long byteOffset) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Short.BYTES);
        return UNSAFE.getShort(memory.base(), Unsafe.ARRAY_BOOLEAN_BASE_OFFSET + byteOffset);
    }

    @Override
    public int readInt(Memory<boolean[]> memory, long byteOffset) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Integer.BYTES);
        return UNSAFE.getInt(memory.base(), Unsafe.ARRAY_BOOLEAN_BASE_OFFSET + byteOffset);
    }

    @Override
    public float readFloat(Memory<boolean[]> memory, long byteOffset) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Float.BYTES);
        return UNSAFE.getFloat(memory.base(), Unsafe.ARRAY_BOOLEAN_BASE_OFFSET + byteOffset);
    }

    @Override
    public long readLong(Memory<boolean[]> memory, long byteOffset) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Long.BYTES);
        return UNSAFE.getLong(memory.base(), Unsafe.ARRAY_BOOLEAN_BASE_OFFSET + byteOffset);
    }

    @Override
    public double readDouble(Memory<boolean[]> memory, long byteOffset) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Double.BYTES);
        return UNSAFE.getDouble(memory.base(), Unsafe.ARRAY_BOOLEAN_BASE_OFFSET + byteOffset);
    }

    @Override
    public void writeByte(Memory<boolean[]> memory, long byteOffset, byte value) {
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkBounds(memory, byteOffset, Byte.BYTES);
        UNSAFE.putBoolean(memory.base(), Unsafe.ARRAY_BOOLEAN_BASE_OFFSET + byteOffset, value != 0);
    }

    @Override
    public void writeShort(Memory<boolean[]> memory, long byteOffset, short value) {
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkBounds(memory, byteOffset, Short.BYTES);
        UNSAFE.putShort(memory.base(), Unsafe.ARRAY_BOOLEAN_BASE_OFFSET + byteOffset, value);
    }

    @Override
    public void writeInt(Memory<boolean[]> memory, long byteOffset, int value) {
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkBounds(memory, byteOffset, Integer.BYTES);
        UNSAFE.putInt(memory.base(), Unsafe.ARRAY_BOOLEAN_BASE_OFFSET + byteOffset, value);
    }

    @Override
    public void writeFloat(Memory<boolean[]> memory, long byteOffset, float value) {
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkBounds(memory, byteOffset, Float.BYTES);
        UNSAFE.putFloat(memory.base(), Unsafe.ARRAY_BOOLEAN_BASE_OFFSET + byteOffset, value);
    }

    @Override
    public void writeLong(Memory<boolean[]> memory, long byteOffset, long value) {
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkBounds(memory, byteOffset, Long.BYTES);
        UNSAFE.putLong(memory.base(), Unsafe.ARRAY_BOOLEAN_BASE_OFFSET + byteOffset, value);
    }

    @Override
    public void writeDouble(Memory<boolean[]> memory, long byteOffset, double value) {
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkBounds(memory, byteOffset, Double.BYTES);
        UNSAFE.putDouble(memory.base(), Unsafe.ARRAY_BOOLEAN_BASE_OFFSET + byteOffset, value);
    }
}
