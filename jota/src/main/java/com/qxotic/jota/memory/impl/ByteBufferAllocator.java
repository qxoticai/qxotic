package com.qxotic.jota.memory.impl;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAllocator;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

final class ByteBufferAllocator implements MemoryAllocator<ByteBuffer> {

    private final boolean direct;
    private final ByteOrder byteOrder;

    private ByteBufferAllocator(boolean direct, ByteOrder byteOrder) {
        this.direct = direct;
        this.byteOrder = Objects.requireNonNull(byteOrder);
    }

    public static MemoryAllocator<ByteBuffer> create(boolean direct, ByteOrder byteOrder) {
        return new ByteBufferAllocator(direct, byteOrder);
    }

    @Override
    public Device device() {
        return Device.NATIVE;
    }

    @Override
    public long memoryGranularity() {
        return Byte.BYTES;
    }

    @Override
    public Memory<ByteBuffer> allocateMemory(long byteSize, long byteAlignment) {
        if (!Util.isPowerOf2(byteAlignment)) {
            throw new IllegalArgumentException("invalid byteAlignment, not a power of 2");
        }
        int intByteSize = Math.toIntExact(byteSize + byteAlignment - 1);
        int intByteAlignment = Math.toIntExact(byteAlignment);
        ByteBuffer byteBuffer = direct
                ? ByteBuffer.allocateDirect(intByteSize)
                : ByteBuffer.allocate(intByteSize);
        byteBuffer = byteBuffer.alignedSlice(intByteAlignment).order(byteOrder);
        return MemoryFactory.ofByteBuffer(byteBuffer);
    }
}
