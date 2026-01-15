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
        return FloatsContext.instance();
    }

    public static MemoryContext<int[]> ofInts() {
        return IntsContext.instance();
    }

    public static MemoryContext<byte[]> ofBytes() {
        return BytesContext.instance();
    }

    public static MemoryContext<boolean[]> ofBooleans() {
        return BooleansContext.instance();
    }

    public static MemoryContext<short[]> ofShorts() {
        return ShortsContext.instance();
    }

    public static MemoryContext<long[]> ofLongs() {
        return LongsContext.instance();
    }

    public static MemoryContext<double[]> ofDoubles() {
        return DoublesContext.instance();
    }

    public static MemoryContext<MemorySegment> ofMemorySegment() {
        return new PanamaContext();
    }

    public static MemoryContext<ByteBuffer> ofByteBuffer(MemoryAllocator<ByteBuffer> memoryAllocator) {
        return new ByteBufferContext(memoryAllocator);
    }
}
