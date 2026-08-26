package com.qxotic.jota.memory.internal;

import com.qxotic.jota.memory.Memory;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;

public final class MemoryFactory {

    private MemoryFactory() {
        // no instances
    }

    public static Memory<boolean[]> ofBooleans(boolean... booleans) {
        return ArrayMemory.of(booleans);
    }

    public static Memory<byte[]> ofBytes(byte... bytes) {
        return ArrayMemory.of(bytes);
    }

    public static Memory<short[]> ofShorts(short... shorts) {
        return ArrayMemory.of(shorts);
    }

    public static Memory<int[]> ofInts(int... ints) {
        return ArrayMemory.of(ints);
    }

    public static Memory<long[]> ofLongs(long... longs) {
        return ArrayMemory.of(longs);
    }

    public static Memory<float[]> ofFloats(float... floats) {
        return ArrayMemory.of(floats);
    }

    public static Memory<double[]> ofDoubles(double... doubles) {
        return ArrayMemory.of(doubles);
    }

    public static Memory<ByteBuffer> ofByteBuffer(ByteBuffer byteBuffer) {
        return ByteBufferMemory.of(byteBuffer);
    }

    public static Memory<MemorySegment> ofMemorySegment(MemorySegment memorySegment) {
        return NativeMemoryFactory.memory(memorySegment);
    }
}
