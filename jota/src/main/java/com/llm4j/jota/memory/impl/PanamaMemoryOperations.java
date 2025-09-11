package com.llm4j.jota.memory.impl;

import com.llm4j.jota.memory.Memory;
import com.llm4j.jota.memory.MemoryOperations;

import java.lang.foreign.MemorySegment;

final class PanamaMemoryOperations implements MemoryOperations<MemorySegment> {

    private static final PanamaMemoryOperations INSTANCE = new PanamaMemoryOperations();

    public static MemoryOperations<MemorySegment> instance() {
        return INSTANCE;
    }

    private PanamaMemoryOperations() {
    }

    @Override
    public void copy(Memory<MemorySegment> src, long srcByteOffset, Memory<MemorySegment> dst, long dstByteOffset, long byteSize) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("negative size");
        }
        MemorySegment.copy(src.base(), srcByteOffset, dst.base(), dstByteOffset, byteSize);
    }

    @Override
    public void copyFromNative(Memory<MemorySegment> src, long srcByteOffset, Memory<MemorySegment> dst, long dstByteOffset, long byteSize) {
        copy(src, srcByteOffset, dst, dstByteOffset, byteSize);
    }

    @Override
    public void copyToNative(Memory<MemorySegment> src, long srcByteOffset, Memory<MemorySegment> dst, long dstByteOffset, long byteSize) {
        copy(src, srcByteOffset, dst, dstByteOffset, byteSize);
    }

    @Override
    public void fillByte(Memory<MemorySegment> memory, long byteOffset, long byteSize, byte value) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("negative size");
        }
        memory.base().asSlice(byteOffset, byteSize).fill(value);
    }

    @Override
    public void fillShort(Memory<MemorySegment> memory, long byteOffset, long byteSize, short shortValue) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void fillInt(Memory<MemorySegment> memory, long byteOffset, long byteSize, int intValue) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void fillLong(Memory<MemorySegment> memory, long byteOffset, long byteSize, long longValue) {
        throw new UnsupportedOperationException();
    }
}
