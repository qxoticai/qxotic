package com.qxotic.jota.memory.impl;

import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccessChecks;
import com.qxotic.jota.memory.MemoryOperations;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;

class BooleansMemoryOperations implements MemoryOperations<boolean[]> {

    private static final BooleansMemoryOperations INSTANCE = new BooleansMemoryOperations();

    public static MemoryOperations<boolean[]> instance() {
        return INSTANCE;
    }

    private BooleansMemoryOperations() {
    }

    @Override
    public void copy(Memory<boolean[]> src, long srcByteOffset, Memory<boolean[]> dst, long dstByteOffset, long byteSize) {
        MemoryAccessChecks.checkBounds(src, srcByteOffset, byteSize);
        MemoryAccessChecks.checkBounds(dst, dstByteOffset, byteSize);
        if (byteSize == 0) {
            return;
        }
        System.arraycopy(
                src.base(), Math.toIntExact(srcByteOffset),
                dst.base(), Math.toIntExact(dstByteOffset),
                Math.toIntExact(byteSize)
        );
    }

    @Override
    public void copyFromNative(Memory<MemorySegment> src, long srcByteOffset, Memory<boolean[]> dst, long dstByteOffset, long byteSize) {
        // MemorySegment.ofArray() doesn't support boolean[]
        throw new UnsupportedOperationException("copyFromNative not supported for boolean[]");
    }

    @Override
    public void copyToNative(Memory<boolean[]> src, long srcByteOffset, Memory<MemorySegment> dst, long dstByteOffset, long byteSize) {
        // MemorySegment.ofArray() doesn't support boolean[]
        throw new UnsupportedOperationException("copyToNative not supported for boolean[]");
    }

    @Override
    public void fillByte(Memory<boolean[]> memory, long byteOffset, long byteSize, byte byteValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        if (byteSize == 0) {
            return;
        }
        int intByteOffset = Math.toIntExact(byteOffset);
        int intByteSize = Math.toIntExact(byteSize);
        Arrays.fill(memory.base(), intByteOffset, intByteOffset + intByteSize, byteValue != 0);
    }

    @Override
    public void fillShort(Memory<boolean[]> memory, long byteOffset, long byteSize, short shortValue) {
        throw new UnsupportedOperationException("fillShort not supported for boolean[]");
    }

    @Override
    public void fillInt(Memory<boolean[]> memory, long byteOffset, long byteSize, int intValue) {
        throw new UnsupportedOperationException("fillInt not supported for boolean[]");
    }

    @Override
    public void fillLong(Memory<boolean[]> memory, long byteOffset, long byteSize, long longValue) {
        throw new UnsupportedOperationException("fillLong not supported for boolean[]");
    }
}
