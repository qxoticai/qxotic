package com.qxotic.jota.runtime.hip;

import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccessChecks;
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
    public void fillByte(Memory<HipDevicePtr> m, long offset, long size, byte value) {
        validateFill(m, offset, size);
        HipRuntime.memsetD8(m.base().address(), offset, size, value);
    }

    @Override
    public void fillShort(Memory<HipDevicePtr> m, long offset, long size, short value) {
        validateFill(m, offset, size, Short.BYTES);
        HipRuntime.memsetD16(m.base().address(), offset, size / Short.BYTES, value);
    }

    @Override
    public void fillInt(Memory<HipDevicePtr> m, long offset, long size, int value) {
        validateFill(m, offset, size, Integer.BYTES);
        HipRuntime.memsetD32(m.base().address(), offset, size / Integer.BYTES, value);
    }

    @Override
    public void fillLong(Memory<HipDevicePtr> m, long offset, long size, long value) {
        validateFill(m, offset, size, Long.BYTES);
        HipRuntime.memsetD64(m.base().address(), offset, size / Long.BYTES, value);
    }

    private static void validateFill(
            Memory<HipDevicePtr> m, long offset, long size, int alignment) {
        MemoryAccessChecks.checkBounds(m, offset, size);
        MemoryAccessChecks.checkWriteable(m);
        if (size == 0) return;
        if (size % alignment != 0) {
            throw new IllegalArgumentException(
                    "Size must be multiple of " + alignment + ": " + size);
        }
        HipRuntime.requireAvailable();
    }

    private static void validateFill(Memory<HipDevicePtr> m, long offset, long size) {
        MemoryAccessChecks.checkBounds(m, offset, size);
        MemoryAccessChecks.checkWriteable(m);
        HipRuntime.requireAvailable();
    }
}
