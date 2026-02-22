package com.qxotic.jota.runtime.hip;

import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryOperations;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

final class HipMemoryOperations implements MemoryOperations<HipDevicePtr> {

    private static final HipMemoryOperations INSTANCE = new HipMemoryOperations();

    static HipMemoryOperations instance() {
        return INSTANCE;
    }

    private HipMemoryOperations() {}

    @Override
    public void copy(
            Memory<HipDevicePtr> src,
            long srcByteOffset,
            Memory<HipDevicePtr> dst,
            long dstByteOffset,
            long byteSize) {
        if (byteSize == 0) {
            return;
        }
        HipRuntime.requireAvailable();
        HipRuntime.memcpyDtoD(
                dst.base().address(), dstByteOffset, src.base().address(), srcByteOffset, byteSize);
    }

    @Override
    public void copyFromNative(
            Memory<MemorySegment> src,
            long srcByteOffset,
            Memory<HipDevicePtr> dst,
            long dstByteOffset,
            long byteSize) {
        if (byteSize == 0) {
            return;
        }
        HipRuntime.requireAvailable();
        MemorySegment srcSegment = src.base();
        if (srcSegment.isNative()) {
            long srcAddress = srcSegment.address() + srcByteOffset;
            HipRuntime.memcpyHtoD(dst.base().address(), dstByteOffset, srcAddress, byteSize);
            return;
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment staging = arena.allocate(byteSize, 1);
            MemorySegment.copy(srcSegment, srcByteOffset, staging, 0, byteSize);
            long srcAddress = staging.address();
            HipRuntime.memcpyHtoD(dst.base().address(), dstByteOffset, srcAddress, byteSize);
        }
    }

    @Override
    public void copyToNative(
            Memory<HipDevicePtr> src,
            long srcByteOffset,
            Memory<MemorySegment> dst,
            long dstByteOffset,
            long byteSize) {
        if (byteSize == 0) {
            return;
        }
        HipRuntime.requireAvailable();
        MemorySegment dstSegment = dst.base();
        if (dstSegment.isNative()) {
            long dstAddress = dstSegment.address() + dstByteOffset;
            HipRuntime.memcpyDtoH(dstAddress, src.base().address(), srcByteOffset, byteSize);
            return;
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment staging = arena.allocate(byteSize, 1);
            HipRuntime.memcpyDtoH(staging.address(), src.base().address(), srcByteOffset, byteSize);
            MemorySegment.copy(staging, 0, dstSegment, dstByteOffset, byteSize);
        }
    }

    @Override
    public void fillByte(
            Memory<HipDevicePtr> memory, long byteOffset, long byteSize, byte byteValue) {
        throw new UnsupportedOperationException("HIP memory fillByte not implemented");
    }

    @Override
    public void fillShort(
            Memory<HipDevicePtr> memory, long byteOffset, long byteSize, short shortValue) {
        throw new UnsupportedOperationException("HIP memory fillShort not implemented");
    }

    @Override
    public void fillInt(Memory<HipDevicePtr> memory, long byteOffset, long byteSize, int intValue) {
        throw new UnsupportedOperationException("HIP memory fillInt not implemented");
    }

    @Override
    public void fillLong(
            Memory<HipDevicePtr> memory, long byteOffset, long byteSize, long longValue) {
        throw new UnsupportedOperationException("HIP memory fillLong not implemented");
    }
}
