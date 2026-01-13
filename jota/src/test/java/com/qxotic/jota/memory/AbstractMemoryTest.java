package com.qxotic.jota.memory;

import com.qxotic.jota.*;
import com.qxotic.jota.memory.impl.ContextFactory;
import com.qxotic.jota.memory.impl.MemoryAllocatorFactory;

import java.util.function.Supplier;
import java.util.stream.Stream;

public abstract class AbstractMemoryTest {

    public static Stream<MemoryContext<?>> contextProvider() {
        Stream<Supplier<MemoryContext<?>>> lazy = Stream.of(
                ContextFactory::ofFloats,
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false)),
                () -> ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(true)),
                ContextFactory::ofMemorySegment
        );
        // Lazy context creation.
        return lazy.map(Supplier::get);
    }

    public static <B> float readFloat(MemoryAccess<B> memoryAccess, MemoryView<B> view, long... coords) {
        return memoryAccess.readFloat(view.memory(), Indexing.coordToOffset(view, coords));
    }

    public static <B> void writeFloat(MemoryAccess<B> memoryAccess, MemoryView<B> view, float floatValue, long... coords) {
        memoryAccess.writeFloat(view.memory(), Indexing.coordToOffset(view, coords), floatValue);
    }

}
