package com.qxotic.jota.memory.impl;

import com.qxotic.jota.memory.Memory;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;

public final class MemoryFactory {

    private MemoryFactory() {
        // no instances
    }

    public static Memory<float[]> ofFloats(float... floats) {
        return FloatArrayMemory.of(floats);
    }

    public static Memory<byte[]> ofBytes(byte... bytes) {
        return ByteArrayMemory.of(bytes);
    }

    public static Memory<int[]> ofInts(int... ints) {
        return IntArrayMemory.of(ints);
    }

    public static Memory<ByteBuffer> ofByteBuffer(ByteBuffer byteBuffer) {
        return ByteBufferMemory.of(byteBuffer);
    }

    public static Memory<MemorySegment> ofMemorySegment(MemorySegment memorySegment) {
        return PanamaMemory.of(memorySegment);
    }
}
