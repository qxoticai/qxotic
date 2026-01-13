package com.qxotic.jota.memory;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.impl.ShapeFactory;
import com.qxotic.jota.memory.impl.MemoryViewFactory;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TestOperations extends AbstractMemoryTest {

    @ParameterizedTest
    @MethodSource("contextProvider")
    <B> void testScalar(MemoryContext<B> context) {
        var allocator = context.memoryAllocator();
        MemoryView<B> view = MemoryViewFactory.allocate(allocator, DataType.F32, Shape.scalar());
        assertEquals(DataType.F32.byteSize(), view.memory().byteSize());
    }

    @ParameterizedTest
    @MethodSource("contextProvider")
    <B> void testMatrix(MemoryContext<B> context) {
        var allocator = context.memoryAllocator();
        Shape shape = ShapeFactory.of(3, 5);
        MemoryView<B> view = MemoryViewFactory.allocate(allocator, DataType.F32, shape);
        assertEquals(DataType.F32.byteSizeFor(shape), view.memory().byteSize());
    }
}
