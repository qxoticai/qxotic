package com.qxotic.jota.memory.impl;

import com.qxotic.jota.memory.Context;
import com.qxotic.jota.memory.MemoryAllocator;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;

public class ContextFactory {

    private ContextFactory() {
        // no instances
    }

    public static Context<float[]> ofFloats() {
        return new FloatsContext();
    }

    public static Context<MemorySegment> ofMemorySegment() {
        return new PanamaContext();
    }

    public static Context<ByteBuffer> ofByteBuffer(MemoryAllocator<ByteBuffer> memoryAllocator) {
        return new ByteBufferContext(memoryAllocator);
    }
}
