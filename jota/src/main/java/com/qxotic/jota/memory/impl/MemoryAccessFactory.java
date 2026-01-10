package com.qxotic.jota.memory.impl;

import com.qxotic.jota.memory.MemoryAccess;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;

public class MemoryAccessFactory {
    private MemoryAccessFactory() {
        // no instances
    }

    public static MemoryAccess<float[]> ofFloats() {
        return FloatArrayMemoryAccess.instance();
    }

    public static MemoryAccess<byte[]> ofBytes() {
        return ByteArrayMemoryAccess.instance();
    }

    public static MemoryAccess<ByteBuffer> ofByteBuffer() {
        return ByteBufferMemoryAccess.instance();
    }

    public static MemoryAccess<MemorySegment> ofMemorySegment() {
        return PanamaMemoryAccess.instance();
    }
}
