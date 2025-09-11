package com.llm4j.jota.memory;

import com.llm4j.jota.DataType;
import com.llm4j.jota.Shape;
import com.llm4j.jota.memory.impl.MemoryAllocatorFactory;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.nio.ByteOrder;
import java.util.function.Supplier;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MemoryAllocatorTest {

    static Stream<Supplier<MemoryAllocator<?>>> managedAllocatorProvider() {
        return Stream.of(
                () -> MemoryAllocatorFactory.ofBytes(),
                () -> MemoryAllocatorFactory.ofFloats(),
                () -> MemoryAllocatorFactory.ofByteBuffer(true, ByteOrder.nativeOrder()),
                () -> MemoryAllocatorFactory.ofByteBuffer(false, ByteOrder.nativeOrder()),
                () -> MemoryAllocatorFactory.newPanamaAuto()
        );
    }

    static Stream<Supplier<ScopedMemoryAllocator<?>>> scopedAllocatorProvider() {
        return Stream.of(
                () -> MemoryAllocatorFactory.ofPanama(),
                () -> MemoryAllocatorFactory.newPanamaArena()
        );
    }

    static Stream<Supplier<ScopedMemoryAllocatorArena<?>>> scopedArenaAllocatorProvider() {
        return Stream.of(
                () -> MemoryAllocatorFactory.newPanamaArena()
        );
    }

    @ParameterizedTest
    @MethodSource("managedAllocatorProvider")
    <B> void testAllocateScalar(Supplier<MemoryAllocator<B>> memoryAllocatorSupplier) {
        var allocator = memoryAllocatorSupplier.get();
        Memory<B> memory = allocator.allocateMemory(DataType.F32, Shape.scalar());
        assertEquals(DataType.F32.byteSize(), memory.byteSize());
    }

//    @ParameterizedTest
//    @MethodSource("memoryAllocatorProvider")
//    <B> void testScalar(Supplier<ScopedMemoryAllocator<B>> memoryAllocatorSupplier) {
//        var allocator = memoryAllocatorSupplier.get();
//        Memory<B> memory = allocator.allocateMemory(DataType.F32, Shape.scalar());
//        assertEquals(DataType.F32.byteSize(), memory.byteSize());
//    }
}
