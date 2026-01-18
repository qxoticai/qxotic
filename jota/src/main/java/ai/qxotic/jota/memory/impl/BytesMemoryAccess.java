package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryAccessChecks;
import sun.misc.Unsafe;

final class BytesMemoryAccess implements MemoryAccess<byte[]> {

    private static final Unsafe UNSAFE = UnsafeAccess.get();

    private static final BytesMemoryAccess INSTANCE = new BytesMemoryAccess();

    public static MemoryAccess<byte[]> instance() {
        return INSTANCE;
    }

    private BytesMemoryAccess() {}

    @Override
    public byte readByte(Memory<byte[]> memory, long byteOffset) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Byte.BYTES);
        return UNSAFE.getByte(memory.base(), Unsafe.ARRAY_BYTE_BASE_OFFSET + byteOffset);
    }

    @Override
    public short readShort(Memory<byte[]> memory, long byteOffset) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Short.BYTES);
        return UNSAFE.getShort(memory.base(), Unsafe.ARRAY_BYTE_BASE_OFFSET + byteOffset);
    }

    @Override
    public int readInt(Memory<byte[]> memory, long byteOffset) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Integer.BYTES);
        return UNSAFE.getInt(memory.base(), Unsafe.ARRAY_BYTE_BASE_OFFSET + byteOffset);
    }

    @Override
    public float readFloat(Memory<byte[]> memory, long byteOffset) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Float.BYTES);
        return UNSAFE.getFloat(memory.base(), Unsafe.ARRAY_BYTE_BASE_OFFSET + byteOffset);
    }

    @Override
    public long readLong(Memory<byte[]> memory, long byteOffset) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Long.BYTES);
        return UNSAFE.getLong(memory.base(), Unsafe.ARRAY_BYTE_BASE_OFFSET + byteOffset);
    }

    @Override
    public double readDouble(Memory<byte[]> memory, long byteOffset) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Double.BYTES);
        return UNSAFE.getDouble(memory.base(), Unsafe.ARRAY_BYTE_BASE_OFFSET + byteOffset);
    }

    @Override
    public void writeByte(Memory<byte[]> memory, long byteOffset, byte value) {
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkBounds(memory, byteOffset, Byte.BYTES);
        UNSAFE.putByte(memory.base(), Unsafe.ARRAY_BYTE_BASE_OFFSET + byteOffset, value);
    }

    @Override
    public void writeShort(Memory<byte[]> memory, long byteOffset, short value) {
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkBounds(memory, byteOffset, Short.BYTES);
        UNSAFE.putShort(memory.base(), Unsafe.ARRAY_BYTE_BASE_OFFSET + byteOffset, value);
    }

    @Override
    public void writeInt(Memory<byte[]> memory, long byteOffset, int value) {
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkBounds(memory, byteOffset, Integer.BYTES);
        UNSAFE.putInt(memory.base(), Unsafe.ARRAY_BYTE_BASE_OFFSET + byteOffset, value);
    }

    @Override
    public void writeFloat(Memory<byte[]> memory, long byteOffset, float value) {
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkBounds(memory, byteOffset, Float.BYTES);
        UNSAFE.putFloat(memory.base(), Unsafe.ARRAY_BYTE_BASE_OFFSET + byteOffset, value);
    }

    @Override
    public void writeLong(Memory<byte[]> memory, long byteOffset, long value) {
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkBounds(memory, byteOffset, Long.BYTES);
        UNSAFE.putLong(memory.base(), Unsafe.ARRAY_BYTE_BASE_OFFSET + byteOffset, value);
    }

    @Override
    public void writeDouble(Memory<byte[]> memory, long byteOffset, double value) {
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkBounds(memory, byteOffset, Double.BYTES);
        UNSAFE.putDouble(memory.base(), Unsafe.ARRAY_BYTE_BASE_OFFSET + byteOffset, value);
    }
}
