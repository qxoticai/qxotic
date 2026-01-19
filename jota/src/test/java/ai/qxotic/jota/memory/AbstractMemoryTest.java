package ai.qxotic.jota.memory;

import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.memory.impl.ContextFactory;
import ai.qxotic.jota.memory.impl.MemoryAllocatorFactory;
import java.util.function.Supplier;
import java.util.stream.Stream;

public abstract class AbstractMemoryTest {

    public static Stream<MemoryContext<?>> onHeapContexts() {
        return suppliedBy(
                ContextFactory::ofBytes,
                ContextFactory::ofShorts,
                ContextFactory::ofInts,
                ContextFactory::ofLongs,
                ContextFactory::ofFloats,
                ContextFactory::ofDoubles);
    }

    public static Stream<MemoryContext<?>> nativeContexts() {
        return suppliedBy(
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true)),
                ContextFactory::ofMemorySegment);
    }

    public static Stream<MemoryContext<?>> allContexts() {
        return Stream.concat(onHeapContexts(), nativeContexts());
    }

    public static Stream<MemoryContext<?>> contextsSupportingF32() {
        return suppliedBy(
                ContextFactory::ofBytes,
                ContextFactory::ofFloats,
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false)),
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true)),
                ContextFactory::ofMemorySegment);
    }

    public static Stream<MemoryContext<?>> contextsSupportingF64() {
        return suppliedBy(
                ContextFactory::ofBytes,
                ContextFactory::ofDoubles,
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false)),
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true)),
                ContextFactory::ofMemorySegment);
    }

    public static Stream<MemoryContext<?>> contextsSupportingI8() {
        return suppliedBy(
                ContextFactory::ofBytes,
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false)),
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true)),
                ContextFactory::ofMemorySegment);
    }

    public static Stream<MemoryContext<?>> contextsSupportingI16() {
        return suppliedBy(
                ContextFactory::ofBytes,
                ContextFactory::ofShorts,
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false)),
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true)),
                ContextFactory::ofMemorySegment);
    }

    public static Stream<MemoryContext<?>> contextsSupportingI32() {
        return suppliedBy(
                ContextFactory::ofInts,
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false)),
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true)),
                ContextFactory::ofMemorySegment);
    }

    public static Stream<MemoryContext<?>> contextsSupportingI64() {
        return suppliedBy(
                ContextFactory::ofLongs,
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false)),
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true)),
                ContextFactory::ofMemorySegment);
    }

    public static Stream<MemoryContext<?>> contextsSupportingBool() {
        return suppliedBy(
                ContextFactory::ofBooleans,
                ContextFactory::ofBytes,
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false)),
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true)),
                ContextFactory::ofMemorySegment);
    }

    @SafeVarargs
    private static <T> Stream<T> suppliedBy(
            Supplier<T>... suppliers) {
        return Stream.of(suppliers).map(Supplier::get);
    }

    public static <B> float readFloat(
            MemoryAccess<B> memoryAccess, MemoryView<B> view, long... coords) {
        return memoryAccess.readFloat(view.memory(), Indexing.coordToOffset(view, coords));
    }

    public static <B> void writeFloat(
            MemoryAccess<B> memoryAccess, MemoryView<B> view, float floatValue, long... coords) {
        memoryAccess.writeFloat(view.memory(), Indexing.coordToOffset(view, coords), floatValue);
    }
}
