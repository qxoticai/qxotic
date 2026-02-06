package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.*;
import java.nio.ByteBuffer;
import java.util.Objects;

class ByteBufferDomain implements MemoryDomain<ByteBuffer> {

    private final MemoryAllocator<ByteBuffer> memoryAllocator;

    ByteBufferDomain(MemoryAllocator<ByteBuffer> memoryAllocator) {
        this.memoryAllocator = Objects.requireNonNull(memoryAllocator);
    }

    @Override
    public Device device() {
        return memoryAllocator.device();
    }

    @Override
    public MemoryAllocator<ByteBuffer> memoryAllocator() {
        return memoryAllocator;
    }

    @Override
    public MemoryAccess<ByteBuffer> directAccess() {
        return ByteBufferMemoryAccess.instance();
    }

    @Override
    public MemoryOperations<ByteBuffer> memoryOperations() {
        return ByteBufferMemoryOperations.instance();
    }

    @Override
    public void close() {
        // Nothing to do, memory is managed by the GC.
    }

    @Override
    public String toString() {
        return new StringBuilder("Context{ByteBuffer, device=")
                .append(device())
                .append(", directAccess=")
                .append(directAccess() != null)
                .append('}')
                .toString();
    }
}
