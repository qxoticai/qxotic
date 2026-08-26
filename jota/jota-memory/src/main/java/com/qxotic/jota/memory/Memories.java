package com.qxotic.jota.memory;

import com.qxotic.jota.memory.internal.MemoryFactory;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;

/**
 * {@link Memory} over storage you already have. Every method wraps; nothing is allocated. Arrays
 * only, no varargs: {@code of(42)} must not silently pick a primitive type.
 */
public final class Memories {

    private Memories() {}

    public static Memory<boolean[]> of(boolean[] booleans) {
        return MemoryFactory.ofBooleans(booleans);
    }

    public static Memory<byte[]> of(byte[] bytes) {
        return MemoryFactory.ofBytes(bytes);
    }

    public static Memory<short[]> of(short[] shorts) {
        return MemoryFactory.ofShorts(shorts);
    }

    public static Memory<int[]> of(int[] ints) {
        return MemoryFactory.ofInts(ints);
    }

    public static Memory<long[]> of(long[] longs) {
        return MemoryFactory.ofLongs(longs);
    }

    public static Memory<float[]> of(float[] floats) {
        return MemoryFactory.ofFloats(floats);
    }

    public static Memory<double[]> of(double[] doubles) {
        return MemoryFactory.ofDoubles(doubles);
    }

    public static Memory<ByteBuffer> of(ByteBuffer byteBuffer) {
        return MemoryFactory.ofByteBuffer(byteBuffer);
    }

    public static Memory<MemorySegment> of(MemorySegment memorySegment) {
        return MemoryFactory.ofMemorySegment(memorySegment);
    }
}
