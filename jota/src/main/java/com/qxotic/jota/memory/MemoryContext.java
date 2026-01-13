package com.qxotic.jota.memory;

import com.qxotic.jota.Device;

public interface MemoryContext<B> extends AutoCloseable {
    Device device();

    MemoryAllocator<B> memoryAllocator();

    /**
     * Optional capability, can be null for opaque memory implementations e.g. GPUs.
     */
    MemoryAccess<B> memoryAccess();

    MemoryOperations<B> memoryOperations();

    FloatOperations<B> floatOperations();

    @Override
    void close();

    String toString();
}

