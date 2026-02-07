package ai.qxotic.jota.runtime.c;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryAllocator;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.memory.MemoryOperations;
import ai.qxotic.jota.panama.PanamaMemoryAccess;
import ai.qxotic.jota.panama.PanamaMemoryOperations;
import java.lang.foreign.MemorySegment;

final class CMemoryDomain implements MemoryDomain<MemorySegment> {

    private final MemoryAllocator<MemorySegment> allocator = new CMemoryAllocator();

    @Override
    public Device device() {
        return Device.C;
    }

    @Override
    public MemoryAllocator<MemorySegment> memoryAllocator() {
        return allocator;
    }

    @Override
    public MemoryAccess<MemorySegment> directAccess() {
        return PanamaMemoryAccess.instance();
    }

    @Override
    public MemoryOperations<MemorySegment> memoryOperations() {
        return PanamaMemoryOperations.instance();
    }

    @Override
    public void close() {
        if (allocator instanceof AutoCloseable closeable) {
            try {
                closeable.close();
            } catch (Exception e) {
                throw new IllegalStateException("Failed to close C memory allocator", e);
            }
        }
    }
}
