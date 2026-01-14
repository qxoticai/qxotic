
package com.qxotic.jota.memory;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ContextHelpersTest extends AbstractMemoryTest {

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void fullFillsWithValue(MemoryContext<B> context) {
        MemoryAccess<B> memoryAccess = context.memoryAccess();
        if (memoryAccess == null) {
            return;
        }
        Shape shape = Shape.of(2, 3);
        MemoryView<B> view = MemoryHelpers.full(context, DataType.F32, shape, 2.5);
        long byteStride = DataType.F32.byteSize();
        for (long i = 0; i < shape.size(); i++) {
            float actual = memoryAccess.readFloat(view.memory(), view.byteOffset() + i * byteStride);
            assertEquals(2.5f, actual);
        }

        MemoryView<B> flat = MemoryHelpers.full(context, DataType.F32, 6, 3.5);
        for (long i = 0; i < flat.shape().size(); i++) {
            float actual = memoryAccess.readFloat(flat.memory(), flat.byteOffset() + i * byteStride);
            assertEquals(3.5f, actual);
        }
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void onesAndZerosFill(MemoryContext<B> context) {
        MemoryAccess<B> memoryAccess = context.memoryAccess();
        if (memoryAccess == null) {
            return;
        }
        Shape shape = Shape.of(2, 2);
        MemoryView<B> ones = MemoryHelpers.ones(context, DataType.F32, shape);
        MemoryView<B> zeros = MemoryHelpers.zeros(context, DataType.F32, shape);
        long byteStride = DataType.F32.byteSize();
        for (long i = 0; i < shape.size(); i++) {
            long offset = i * byteStride;
            assertEquals(1.0f, memoryAccess.readFloat(ones.memory(), ones.byteOffset() + offset));
            assertEquals(0.0f, memoryAccess.readFloat(zeros.memory(), zeros.byteOffset() + offset));
        }

        MemoryView<B> flatOnes = MemoryHelpers.ones(context, DataType.F32, 4);
        MemoryView<B> flatZeros = MemoryHelpers.zeros(context, DataType.F32, 4);
        for (long i = 0; i < flatOnes.shape().size(); i++) {
            long offset = i * byteStride;
            assertEquals(1.0f, memoryAccess.readFloat(flatOnes.memory(), flatOnes.byteOffset() + offset));
            assertEquals(0.0f, memoryAccess.readFloat(flatZeros.memory(), flatZeros.byteOffset() + offset));
        }
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void arangeBuildsSequence(MemoryContext<B> context) {
        MemoryAccess<B> memoryAccess = context.memoryAccess();
        if (memoryAccess == null) {
            return;
        }
        MemoryView<B> view = MemoryHelpers.arange(context, DataType.F32, 2, 8, 2);
        long byteStride = DataType.F32.byteSize();
        for (long i = 0; i < view.shape().size(); i++) {
            float actual = memoryAccess.readFloat(view.memory(), view.byteOffset() + i * byteStride);
            assertEquals(2.0f + i * 2.0f, actual);
        }

        MemoryView<B> simple = MemoryHelpers.arange(context, DataType.F32, 10);
        for (long i = 0; i < simple.shape().size(); i++) {
            float actual = memoryAccess.readFloat(simple.memory(), simple.byteOffset() + i * byteStride);
            assertEquals((float) i, actual);
        }
    }
}
