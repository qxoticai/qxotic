package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.memory.MemoryAllocator;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.panama.PanamaFactory;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;

public class ContextFactory {

    private ContextFactory() {
        // no instances
    }

    public static MemoryContext<boolean[]> ofBooleans() {
        return BooleansContext.instance();
    }

    public static MemoryContext<byte[]> ofBytes() {
        return BytesContext.instance();
    }

    public static MemoryContext<short[]> ofShorts() {
        return ShortsContext.instance();
    }

    public static MemoryContext<int[]> ofInts() {
        return IntsContext.instance();
    }

    public static MemoryContext<long[]> ofLongs() {
        return LongsContext.instance();
    }

    public static MemoryContext<float[]> ofFloats() {
        return FloatsContext.instance();
    }

    public static MemoryContext<double[]> ofDoubles() {
        return DoublesContext.instance();
    }

    public static MemoryContext<MemorySegment> ofMemorySegment(
            MemoryAllocator<MemorySegment> memoryAllocator) {
        return PanamaFactory.createContext(memoryAllocator);
    }

    public static MemoryContext<MemorySegment> ofMemorySegment() {
        return PanamaFactory.createContext();
    }

    public static MemoryContext<ByteBuffer> ofByteBuffer(
            MemoryAllocator<ByteBuffer> memoryAllocator) {
        return new ByteBufferContext(memoryAllocator);
    }
}
