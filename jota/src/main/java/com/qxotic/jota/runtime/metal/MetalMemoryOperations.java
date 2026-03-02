package com.qxotic.jota.runtime.metal;

import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccessChecks;
import com.qxotic.jota.memory.MemoryOperations;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

final class MetalMemoryOperations implements MemoryOperations<MetalDevicePtr> {

    private static final MetalMemoryOperations INSTANCE = new MetalMemoryOperations();

    static MetalMemoryOperations instance() {
        return INSTANCE;
    }

    private MetalMemoryOperations() {}

    @Override
    public void copy(
            Memory<MetalDevicePtr> src,
            long srcByteOffset,
            Memory<MetalDevicePtr> dst,
            long dstByteOffset,
            long byteSize) {
        if (byteSize == 0) {
            return;
        }
        MetalRuntime.requireAvailable();
        MetalRuntime.memcpyDtoD(
                dst.base().handle(), dstByteOffset, src.base().handle(), srcByteOffset, byteSize);
    }

    @Override
    public void copyFromNative(
            Memory<MemorySegment> src,
            long srcByteOffset,
            Memory<MetalDevicePtr> dst,
            long dstByteOffset,
            long byteSize) {
        if (byteSize == 0) {
            return;
        }
        MetalRuntime.requireAvailable();
        MemorySegment srcSegment = src.base();
        if (srcSegment.isNative()) {
            MetalRuntime.memcpyHtoD(
                    dst.base().handle(),
                    dstByteOffset,
                    srcSegment.address() + srcByteOffset,
                    byteSize);
            return;
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment staging = arena.allocate(byteSize, 1);
            MemorySegment.copy(srcSegment, srcByteOffset, staging, 0, byteSize);
            MetalRuntime.memcpyHtoD(
                    dst.base().handle(), dstByteOffset, staging.address(), byteSize);
        }
    }

    @Override
    public void copyToNative(
            Memory<MetalDevicePtr> src,
            long srcByteOffset,
            Memory<MemorySegment> dst,
            long dstByteOffset,
            long byteSize) {
        if (byteSize == 0) {
            return;
        }
        MetalRuntime.requireAvailable();
        MemorySegment dstSegment = dst.base();
        if (dstSegment.isNative()) {
            MetalRuntime.memcpyDtoH(
                    dstSegment.address() + dstByteOffset,
                    src.base().handle(),
                    srcByteOffset,
                    byteSize);
            return;
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment staging = arena.allocate(byteSize, 1);
            MetalRuntime.memcpyDtoH(
                    staging.address(), src.base().handle(), srcByteOffset, byteSize);
            MemorySegment.copy(staging, 0, dstSegment, dstByteOffset, byteSize);
        }
    }

    @Override
    public void fillByte(
            Memory<MetalDevicePtr> memory, long byteOffset, long byteSize, byte byteValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        if (byteSize == 0) {
            return;
        }
        MetalRuntime.requireAvailable();
        MetalRuntime.fillByte(memory.base().handle(), byteOffset, byteSize, byteValue);
    }

    @Override
    public void fillShort(
            Memory<MetalDevicePtr> memory, long byteOffset, long byteSize, short shortValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteSize, Short.BYTES);
        if (byteSize == 0) {
            return;
        }
        MetalRuntime.requireAvailable();
        MetalRuntime.fillShort(memory.base().handle(), byteOffset, byteSize, shortValue);
    }

    @Override
    public void fillInt(
            Memory<MetalDevicePtr> memory, long byteOffset, long byteSize, int intValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteSize, Integer.BYTES);
        if (byteSize == 0) {
            return;
        }
        MetalRuntime.requireAvailable();
        MetalRuntime.fillInt(memory.base().handle(), byteOffset, byteSize, intValue);
    }

    @Override
    public void fillLong(
            Memory<MetalDevicePtr> memory, long byteOffset, long byteSize, long longValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteSize, Long.BYTES);
        if (byteSize == 0) {
            return;
        }
        MetalRuntime.requireAvailable();
        MetalRuntime.fillLong(memory.base().handle(), byteOffset, byteSize, longValue);
    }
}
