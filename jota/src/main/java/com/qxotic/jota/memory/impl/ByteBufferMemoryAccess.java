package com.qxotic.jota.memory.impl;

import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAccessChecks;

import java.nio.ByteBuffer;

final class ByteBufferMemoryAccess implements MemoryAccess<ByteBuffer> {

    private static final ByteBufferMemoryAccess INSTANCE = new ByteBufferMemoryAccess();

    public static MemoryAccess<ByteBuffer> instance() {
        return INSTANCE;
    }

    private ByteBufferMemoryAccess() {
    }

    @Override
    public byte readByte(Memory<ByteBuffer> memory, long byteOffset) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Byte.BYTES);
        int intByteOffset = Math.toIntExact(byteOffset);
        return memory.base().get(intByteOffset);
    }

    @Override
    public short readShort(Memory<ByteBuffer> memory, long byteOffset) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Short.BYTES);
        int intByteOffset = Math.toIntExact(byteOffset);
        return memory.base().getShort(intByteOffset);
    }

    @Override
    public int readInt(Memory<ByteBuffer> memory, long byteOffset) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Integer.BYTES);
        int intByteOffset = Math.toIntExact(byteOffset);
        return memory.base().getInt(intByteOffset);
    }

    @Override
    public float readFloat(Memory<ByteBuffer> memory, long byteOffset) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Float.BYTES);
        int intByteOffset = Math.toIntExact(byteOffset);
        return memory.base().getFloat(intByteOffset);
    }

    @Override
    public long readLong(Memory<ByteBuffer> memory, long byteOffset) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Long.BYTES);
        int intByteOffset = Math.toIntExact(byteOffset);
        return memory.base().getLong(intByteOffset);
    }

    @Override
    public double readDouble(Memory<ByteBuffer> memory, long byteOffset) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Double.BYTES);
        int intByteOffset = Math.toIntExact(byteOffset);
        return memory.base().getDouble(intByteOffset);
    }

    @Override
    public void writeByte(Memory<ByteBuffer> memory, long byteOffset, byte byteValue) {
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkBounds(memory, byteOffset, Byte.BYTES);
        int intByteOffset = Math.toIntExact(byteOffset);
        memory.base().put(intByteOffset, byteValue);
    }

    @Override
    public void writeShort(Memory<ByteBuffer> memory, long byteOffset, short shortValue) {
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkBounds(memory, byteOffset, Short.BYTES);
        int intByteOffset = Math.toIntExact(byteOffset);
        memory.base().putShort(intByteOffset, shortValue);
    }

    @Override
    public void writeInt(Memory<ByteBuffer> memory, long byteOffset, int intValue) {
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkBounds(memory, byteOffset, Integer.BYTES);
        int intByteOffset = Math.toIntExact(byteOffset);
        memory.base().putInt(intByteOffset, intValue);
    }

    @Override
    public void writeFloat(Memory<ByteBuffer> memory, long byteOffset, float floatValue) {
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkBounds(memory, byteOffset, Float.BYTES);
        int intByteOffset = Math.toIntExact(byteOffset);
        memory.base().putFloat(intByteOffset, floatValue);
    }

    @Override
    public void writeLong(Memory<ByteBuffer> memory, long byteOffset, long longValue) {
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkBounds(memory, byteOffset, Long.BYTES);
        int intByteOffset = Math.toIntExact(byteOffset);
        memory.base().putLong(intByteOffset, longValue);
    }

    @Override
    public void writeDouble(Memory<ByteBuffer> memory, long byteOffset, double doubleValue) {
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkBounds(memory, byteOffset, Double.BYTES);
        int intByteOffset = Math.toIntExact(byteOffset);
        memory.base().putDouble(intByteOffset, doubleValue);
    }
}
