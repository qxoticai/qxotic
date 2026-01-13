package com.qxotic.jota.memory.impl;

import com.qxotic.jota.memory.MemoryContext;
import com.qxotic.jota.memory.MemoryAllocator;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;

public class ContextFactory {

    private ContextFactory() {
        // no instances
    }

    public static MemoryContext<float[]> ofFloats() {
        return new FloatsContext();
    }

    public static MemoryContext<MemorySegment> ofMemorySegment() {
        return new PanamaContext();
    }

    public static MemoryContext<ByteBuffer> ofByteBuffer(MemoryAllocator<ByteBuffer> memoryAllocator) {
        return new ByteBufferContext(memoryAllocator);
    }
}
