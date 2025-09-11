package com.llm4j.jota.memory;

import com.llm4j.jota.Device;

public interface Context<B> extends AutoCloseable {
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
}

