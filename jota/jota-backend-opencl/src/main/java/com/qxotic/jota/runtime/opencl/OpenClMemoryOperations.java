package com.qxotic.jota.runtime.opencl;

import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccessChecks;
import com.qxotic.jota.memory.MemoryOperations;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

final class OpenClMemoryOperations implements MemoryOperations<OpenClDevicePtr> {

    private static final OpenClMemoryOperations INSTANCE = new OpenClMemoryOperations();

    static OpenClMemoryOperations instance() {
        return INSTANCE;
    }

    private OpenClMemoryOperations() {}

    @Override
    public void copy(
            Memory<OpenClDevicePtr> src,
            long srcByteOffset,
            Memory<OpenClDevicePtr> dst,
            long dstByteOffset,
            long byteSize) {
        if (byteSize == 0) {
            return;
        }
        OpenClRuntime.requireAvailable();
        OpenClRuntime.memcpyDtoD(
                dst.base().handle(), dstByteOffset, src.base().handle(), srcByteOffset, byteSize);
    }

    @Override
    public void copyFromNative(
            Memory<MemorySegment> src,
            long srcByteOffset,
            Memory<OpenClDevicePtr> dst,
            long dstByteOffset,
            long byteSize) {
        if (byteSize == 0) {
            return;
        }
        OpenClRuntime.requireAvailable();
        MemorySegment srcSegment = src.base();
        if (srcSegment.isNative()) {
            OpenClRuntime.memcpyHtoD(
                    dst.base().handle(),
                    dstByteOffset,
                    srcSegment.address() + srcByteOffset,
                    byteSize);
            return;
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment staging = arena.allocate(byteSize, 1);
            MemorySegment.copy(srcSegment, srcByteOffset, staging, 0, byteSize);
            OpenClRuntime.memcpyHtoD(
                    dst.base().handle(), dstByteOffset, staging.address(), byteSize);
        }
    }

    @Override
    public void copyToNative(
            Memory<OpenClDevicePtr> src,
            long srcByteOffset,
            Memory<MemorySegment> dst,
            long dstByteOffset,
            long byteSize) {
        if (byteSize == 0) {
            return;
        }
        OpenClRuntime.requireAvailable();
        MemorySegment dstSegment = dst.base();
        if (dstSegment.isNative()) {
            OpenClRuntime.memcpyDtoH(
                    dstSegment.address() + dstByteOffset,
                    src.base().handle(),
                    srcByteOffset,
                    byteSize);
            return;
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment staging = arena.allocate(byteSize, 1);
            OpenClRuntime.memcpyDtoH(
                    staging.address(), src.base().handle(), srcByteOffset, byteSize);
            MemorySegment.copy(staging, 0, dstSegment, dstByteOffset, byteSize);
        }
    }

    @Override
    public void fillByte(
            Memory<OpenClDevicePtr> memory, long byteOffset, long byteSize, byte byteValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        if (byteSize == 0) {
            return;
        }
        OpenClRuntime.requireAvailable();
        OpenClRuntime.fillByte(memory.base().handle(), byteOffset, byteSize, byteValue);
    }

    @Override
    public void fillShort(
            Memory<OpenClDevicePtr> memory, long byteOffset, long byteSize, short shortValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteSize, Short.BYTES);
        if (byteSize == 0) {
            return;
        }
        OpenClRuntime.requireAvailable();
        OpenClRuntime.fillShort(memory.base().handle(), byteOffset, byteSize, shortValue);
    }

    @Override
    public void fillInt(
            Memory<OpenClDevicePtr> memory, long byteOffset, long byteSize, int intValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteSize, Integer.BYTES);
        if (byteSize == 0) {
            return;
        }
        OpenClRuntime.requireAvailable();
        OpenClRuntime.fillInt(memory.base().handle(), byteOffset, byteSize, intValue);
    }

    @Override
    public void fillLong(
            Memory<OpenClDevicePtr> memory, long byteOffset, long byteSize, long longValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteSize, Long.BYTES);
        if (byteSize == 0) {
            return;
        }
        OpenClRuntime.requireAvailable();
        OpenClRuntime.fillLong(memory.base().handle(), byteOffset, byteSize, longValue);
    }
}
