package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.memory.Memory;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;

public final class MemoryFactory {

    private MemoryFactory() {
        // no instances
    }

    public static Memory<float[]> ofFloats(float... floats) {
        return FloatsMemory.of(floats);
    }

    public static Memory<byte[]> ofBytes(byte... bytes) {
        return BytesMemory.of(bytes);
    }

    public static Memory<boolean[]> ofBooleans(boolean... booleans) {
        return BooleansMemory.of(booleans);
    }

    public static Memory<int[]> ofInts(int... ints) {
        return IntsMemory.of(ints);
    }

    public static Memory<short[]> ofShorts(short... shorts) {
        return ShortsMemory.of(shorts);
    }

    public static Memory<long[]> ofLongs(long... longs) {
        return LongsMemory.of(longs);
    }

    public static Memory<double[]> ofDoubles(double... doubles) {
        return DoublesMemory.of(doubles);
    }

    public static Memory<ByteBuffer> ofByteBuffer(ByteBuffer byteBuffer) {
        return ByteBufferMemory.of(byteBuffer);
    }

    public static Memory<MemorySegment> ofMemorySegment(MemorySegment memorySegment) {
        return PanamaMemory.of(memorySegment);
    }
}
