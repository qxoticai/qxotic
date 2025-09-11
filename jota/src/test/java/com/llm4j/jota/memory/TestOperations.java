package com.llm4j.jota.memory;

import com.llm4j.jota.DataType;
import com.llm4j.jota.Shape;
import com.llm4j.jota.impl.ShapeFactory;
import com.llm4j.jota.memory.impl.MemoryViewFactory;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TestOperations extends AbstractMemoryTest {

    @ParameterizedTest
    @MethodSource("contextProvider")
    <B> void testScalar(Context<B> context) {
        var allocator = context.memoryAllocator();
        MemoryView<B> view = MemoryViewFactory.allocate(Shape.scalar(), DataType.F32, allocator);
        assertEquals(DataType.F32.byteSize(), view.memory().byteSize());
    }

    @ParameterizedTest
    @MethodSource("contextProvider")
    <B> void testMatrix(Context<B> context) {
        var allocator = context.memoryAllocator();
        Shape shape = ShapeFactory.of(3, 5);
        MemoryView<B> view = MemoryViewFactory.allocate(shape, DataType.F32, allocator);
        assertEquals(DataType.F32.byteSizeFor(shape), view.memory().byteSize());
    }
}
