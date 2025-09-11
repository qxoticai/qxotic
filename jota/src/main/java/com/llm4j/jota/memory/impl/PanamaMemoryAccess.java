package com.llm4j.jota.memory.impl;

import com.llm4j.jota.memory.Memory;
import com.llm4j.jota.memory.MemoryAccess;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

final class PanamaMemoryAccess implements MemoryAccess<MemorySegment> {

    private static final PanamaMemoryAccess INSTANCE = new PanamaMemoryAccess();

    public static MemoryAccess<MemorySegment> instance() {
        return INSTANCE;
    }

    private PanamaMemoryAccess() {
    }

    @Override
    public byte readByte(Memory<MemorySegment> memory, long byteOffset) {
        return memory.base().get(ValueLayout.JAVA_BYTE, byteOffset);
    }

    @Override
    public short readShort(Memory<MemorySegment> memory, long byteOffset) {
        return memory.base().get(ValueLayout.JAVA_SHORT_UNALIGNED, byteOffset);
    }

    @Override
    public int readInt(Memory<MemorySegment> memory, long byteOffset) {
        return memory.base().get(ValueLayout.JAVA_INT_UNALIGNED, byteOffset);
    }

    @Override
    public float readFloat(Memory<MemorySegment> memory, long byteOffset) {
        return memory.base().get(ValueLayout.JAVA_FLOAT_UNALIGNED, byteOffset);
    }

    @Override
    public long readLong(Memory<MemorySegment> memory, long byteOffset) {
        return memory.base().get(ValueLayout.JAVA_LONG_UNALIGNED, byteOffset);
    }

    @Override
    public double readDouble(Memory<MemorySegment> memory, long byteOffset) {
        return memory.base().get(ValueLayout.JAVA_DOUBLE_UNALIGNED, byteOffset);
    }

    @Override
    public void writeByte(Memory<MemorySegment> memory, long byteOffset, byte value) {
        // assert !memory.isReadOnly();
        memory.base().set(ValueLayout.JAVA_BYTE, byteOffset, value);
    }

    @Override
    public void writeShort(Memory<MemorySegment> memory, long byteOffset, short value) {
        // assert !memory.isReadOnly();
        memory.base().set(ValueLayout.JAVA_SHORT_UNALIGNED, byteOffset, value);
    }

    @Override
    public void writeInt(Memory<MemorySegment> memory, long byteOffset, int value) {
        // assert !memory.isReadOnly();
        memory.base().set(ValueLayout.JAVA_INT_UNALIGNED, byteOffset, value);
    }

    @Override
    public void writeFloat(Memory<MemorySegment> memory, long byteOffset, float value) {
        // assert !memory.isReadOnly();
        memory.base().set(ValueLayout.JAVA_FLOAT_UNALIGNED, byteOffset, value);
    }

    @Override
    public void writeLong(Memory<MemorySegment> memory, long byteOffset, long value) {
        // assert !memory.isReadOnly();
        memory.base().set(ValueLayout.JAVA_LONG_UNALIGNED, byteOffset, value);
    }

    @Override
    public void writeDouble(Memory<MemorySegment> memory, long byteOffset, double value) {
        // assert !memory.isReadOnly();
        memory.base().set(ValueLayout.JAVA_DOUBLE_UNALIGNED, byteOffset, value);
    }
}
