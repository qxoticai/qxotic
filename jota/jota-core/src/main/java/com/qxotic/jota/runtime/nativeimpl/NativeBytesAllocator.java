package com.qxotic.jota.runtime.nativeimpl;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryArena;
import java.lang.foreign.MemorySegment;

final class NativeBytesAllocator
        implements MemoryAllocator<MemorySegment>, MemoryArena<MemorySegment> {

    private static final MemoryAllocator<MemorySegment> INSTANCE = new NativeBytesAllocator();

    public static MemoryAllocator<MemorySegment> instance() {
        return INSTANCE;
    }

    @Override
    public Device device() {
        return Device.PANAMA;
    } // JAVA?

    @Override
    public long memoryGranularity() {
        return Byte.BYTES;
    }

    @Override
    public Memory<MemorySegment> allocateMemory(long byteSize, long byteAlignment) {
        if (defaultByteAlignment() % byteAlignment != 0) {
            // Cannot guarantee more than 1 alignment.
            throw new IllegalArgumentException("unsupported byteAlignment");
        }
        int byteSizeInt = Math.toIntExact(byteSize);
        return NativeMemorySegmentMemory.of(MemorySegment.ofArray(new byte[byteSizeInt]));
    }

    @Override
    public void close() {
        // nop
    }
}
