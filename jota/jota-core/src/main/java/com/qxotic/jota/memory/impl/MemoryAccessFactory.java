package com.qxotic.jota.memory.impl;

import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.runtime.panama.PanamaFactory;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;

public class MemoryAccessFactory {
    private MemoryAccessFactory() {
        // no instances
    }

    public static MemoryAccess<float[]> ofFloats() {
        return FloatsMemoryAccess.instance();
    }

    public static MemoryAccess<byte[]> ofBytes() {
        return BytesMemoryAccess.instance();
    }

    public static MemoryAccess<ByteBuffer> ofByteBuffer() {
        return ByteBufferMemoryAccess.instance();
    }

    public static MemoryAccess<MemorySegment> ofMemorySegment() {
        return PanamaFactory.memoryAccess();
    }
}
