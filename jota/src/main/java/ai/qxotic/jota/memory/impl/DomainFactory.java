package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.memory.MemoryAllocator;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.runtime.panama.PanamaFactory;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;

public class DomainFactory {

    private DomainFactory() {
        // no instances
    }

    public static MemoryDomain<boolean[]> ofBooleans() {
        return BooleansDomain.instance();
    }

    public static MemoryDomain<byte[]> ofBytes() {
        return BytesDomain.instance();
    }

    public static MemoryDomain<short[]> ofShorts() {
        return ShortsDomain.instance();
    }

    public static MemoryDomain<int[]> ofInts() {
        return IntsDomain.instance();
    }

    public static MemoryDomain<long[]> ofLongs() {
        return LongsDomain.instance();
    }

    public static MemoryDomain<float[]> ofFloats() {
        return FloatsDomain.instance();
    }

    public static MemoryDomain<double[]> ofDoubles() {
        return DoublesDomain.instance();
    }

    public static MemoryDomain<MemorySegment> ofMemorySegment(
            MemoryAllocator<MemorySegment> memoryAllocator) {
        return PanamaFactory.createDomain(memoryAllocator);
    }

    public static MemoryDomain<MemorySegment> ofMemorySegment() {
        return PanamaFactory.createDomain();
    }

    public static MemoryDomain<ByteBuffer> ofByteBuffer(
            MemoryAllocator<ByteBuffer> memoryAllocator) {
        return new ByteBufferDomain(memoryAllocator);
    }
}
