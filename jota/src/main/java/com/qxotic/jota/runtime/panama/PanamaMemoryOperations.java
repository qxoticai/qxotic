package com.qxotic.jota.runtime.panama;

import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccessChecks;
import com.qxotic.jota.memory.MemoryOperations;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public final class PanamaMemoryOperations implements MemoryOperations<MemorySegment> {

    private static final PanamaMemoryOperations INSTANCE = new PanamaMemoryOperations();

    public static MemoryOperations<MemorySegment> instance() {
        return INSTANCE;
    }

    private PanamaMemoryOperations() {}

    @Override
    public void copy(
            Memory<MemorySegment> src,
            long srcByteOffset,
            Memory<MemorySegment> dst,
            long dstByteOffset,
            long byteSize) {
        MemoryAccessChecks.checkBounds(src, srcByteOffset, byteSize);
        MemoryAccessChecks.checkBounds(dst, dstByteOffset, byteSize);
        if (byteSize == 0) {
            return;
        }
        MemorySegment.copy(src.base(), srcByteOffset, dst.base(), dstByteOffset, byteSize);
    }

    @Override
    public void copyFromNative(
            Memory<MemorySegment> src,
            long srcByteOffset,
            Memory<MemorySegment> dst,
            long dstByteOffset,
            long byteSize) {
        copy(src, srcByteOffset, dst, dstByteOffset, byteSize);
    }

    @Override
    public void copyToNative(
            Memory<MemorySegment> src,
            long srcByteOffset,
            Memory<MemorySegment> dst,
            long dstByteOffset,
            long byteSize) {
        copy(src, srcByteOffset, dst, dstByteOffset, byteSize);
    }

    @Override
    public void fillByte(Memory<MemorySegment> memory, long byteOffset, long byteSize, byte value) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        if (byteSize == 0) {
            return;
        }
        memory.base().asSlice(byteOffset, byteSize).fill(value);
    }

    @Override
    public void fillShort(
            Memory<MemorySegment> memory, long byteOffset, long byteSize, short shortValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        if (byteSize == 0) {
            return;
        }
        long end = byteOffset + byteSize;
        for (long offset = byteOffset; offset < end; offset += Short.BYTES) {
            memory.base().set(ValueLayout.JAVA_SHORT_UNALIGNED, offset, shortValue);
        }
    }

    @Override
    public void fillInt(
            Memory<MemorySegment> memory, long byteOffset, long byteSize, int intValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        if (byteSize == 0) {
            return;
        }
        long end = byteOffset + byteSize;
        for (long offset = byteOffset; offset < end; offset += Integer.BYTES) {
            memory.base().set(ValueLayout.JAVA_INT_UNALIGNED, offset, intValue);
        }
    }

    @Override
    public void fillLong(
            Memory<MemorySegment> memory, long byteOffset, long byteSize, long longValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        if (byteSize == 0) {
            return;
        }
        long end = byteOffset + byteSize;
        for (long offset = byteOffset; offset < end; offset += Long.BYTES) {
            memory.base().set(ValueLayout.JAVA_LONG_UNALIGNED, offset, longValue);
        }
    }
}
