package com.qxotic.jota.memory;

import com.qxotic.jota.*;
import com.qxotic.jota.memory.impl.ContextFactory;
import com.qxotic.jota.memory.impl.MemoryAllocatorFactory;

import java.util.function.Supplier;
import java.util.stream.Stream;

public abstract class AbstractMemoryTest {

    public static Stream<MemoryContext<?>> onHeapContexts() {
        return contextSuppliers(
                ContextFactory::ofBytes,
                ContextFactory::ofShorts,
                ContextFactory::ofInts,
                ContextFactory::ofLongs,
                ContextFactory::ofFloats,
                ContextFactory::ofDoubles
        );
    }

    public static Stream<MemoryContext<?>> nativeContexts() {
        return contextSuppliers(
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true)),
                ContextFactory::ofMemorySegment
        );
    }

    public static Stream<MemoryContext<?>> allContexts() {
        return Stream.concat(onHeapContexts(), nativeContexts());
    }

    public static Stream<MemoryContext<?>> contextsSupportingF32() {
        return contextSuppliers(
                ContextFactory::ofBytes,
                ContextFactory::ofFloats,
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false)),
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true)),
                ContextFactory::ofMemorySegment
        );
    }

    public static Stream<MemoryContext<?>> contextsSupportingF64() {
        return contextSuppliers(
                ContextFactory::ofBytes,
                ContextFactory::ofDoubles,
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false)),
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true)),
                ContextFactory::ofMemorySegment
        );
    }

    public static Stream<MemoryContext<?>> contextsSupportingI8() {
        return contextSuppliers(
                ContextFactory::ofBytes,
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false)),
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true)),
                ContextFactory::ofMemorySegment
        );
    }

    public static Stream<MemoryContext<?>> contextsSupportingI16() {
        return contextSuppliers(
                ContextFactory::ofBytes,
                ContextFactory::ofShorts,
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false)),
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true)),
                ContextFactory::ofMemorySegment
        );
    }

    public static Stream<MemoryContext<?>> contextsSupportingI32() {
        return contextSuppliers(
                ContextFactory::ofInts,
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false)),
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true)),
                ContextFactory::ofMemorySegment
        );
    }

    public static Stream<MemoryContext<?>> contextsSupportingI64() {
        return contextSuppliers(
                ContextFactory::ofLongs,
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false)),
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true)),
                ContextFactory::ofMemorySegment
        );
    }

    @SafeVarargs
    private static Stream<MemoryContext<?>> contextSuppliers(Supplier<MemoryContext<?>>... suppliers) {
        return Stream.of(suppliers).map(Supplier::get);
    }

    public static <B> float readFloat(MemoryAccess<B> memoryAccess, MemoryView<B> view, long... coords) {
        return memoryAccess.readFloat(view.memory(), Indexing.coordToOffset(view, coords));
    }

    public static <B> void writeFloat(MemoryAccess<B> memoryAccess, MemoryView<B> view, float floatValue, long... coords) {
        memoryAccess.writeFloat(view.memory(), Indexing.coordToOffset(view, coords), floatValue);
    }

}
