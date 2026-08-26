package com.qxotic.jota.memory.internal;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAllocator;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

final class ByteBufferAllocator implements MemoryAllocator<ByteBuffer> {

    private final boolean direct;

    private ByteBufferAllocator(boolean direct) {
        this.direct = direct;
    }

    /**
     * Buffers are always native-order: the byte copies through {@code MemoryOperations} assume it.
     */
    public static MemoryAllocator<ByteBuffer> create(boolean direct) {
        return new ByteBufferAllocator(direct);
    }

    @Override
    public Device device() {
        return DeviceType.JAVA.deviceIndex(0);
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
        if (!direct && byteAlignment > 1) {
            // A heap array can be moved by the GC, so a heap buffer has no address to align to.
            throw new IllegalArgumentException(
                    "heap ByteBuffer cannot be aligned to " + byteAlignment);
        }
        int size = Math.toIntExact(byteSize);
        int align = Math.toIntExact(byteAlignment);
        // Over-allocate by align-1 and slice exactly `size` bytes from the first aligned index.
        // NOT alignedSlice(align): that rounds the limit DOWN to a multiple of align too, so the
        // slice comes back shorter than requested.
        int capacity = Math.addExact(size, align - 1);
        ByteBuffer raw =
                direct ? ByteBuffer.allocateDirect(capacity) : ByteBuffer.allocate(capacity);
        int start = (align - raw.alignmentOffset(0, align)) & (align - 1);
        return MemoryFactory.ofByteBuffer(raw.slice(start, size).order(ByteOrder.nativeOrder()));
    }
}
