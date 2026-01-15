package ai.qxotic.jota.memory;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.impl.ShapeFactory;
import ai.qxotic.jota.memory.impl.MemoryViewFactory;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TestOperations extends AbstractMemoryTest {

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void testScalar(MemoryContext<B> context) {
        var allocator = context.memoryAllocator();
        MemoryView<B> view = MemoryViewFactory.allocate(allocator, DataType.FP32, Shape.scalar());
        assertEquals(DataType.FP32.byteSize(), view.memory().byteSize());
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void testMatrix(MemoryContext<B> context) {
        var allocator = context.memoryAllocator();
        Shape shape = ShapeFactory.of(3, 5);
        MemoryView<B> view = MemoryViewFactory.allocate(allocator, DataType.FP32, shape);
        assertEquals(DataType.FP32.byteSizeFor(shape), view.memory().byteSize());
    }
}
