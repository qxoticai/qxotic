
package com.qxotic.jota.memory;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MemoryViewIndexOrderTest extends AbstractMemoryTest {

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void rowMajorCoordinatesMatchLinearOrder(MemoryContext<B> context) {
        MemoryAccess<B> memoryAccess = context.memoryAccess();
        if (memoryAccess == null) {
            return;
        }

        Shape shape = Shape.of(2, 3);
        MemoryView<B> base = MemoryHelpers.arange(context, DataType.FP32, shape.size());
        MemoryView<B> view = MemoryView.of(base.memory(), 0L, DataType.FP32, Layout.rowMajor(shape));

        for (long i = 0; i < shape.size(0); i++) {
            for (long j = 0; j < shape.size(1); j++) {
                long expected = i * shape.size(1) + j;
                long offset = Indexing.coordToOffset(view, i, j);
                float actual = memoryAccess.readFloat(view.memory(), offset);
                assertEquals((float) expected, actual);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void columnMajorCoordinatesMatchColumnOrder(MemoryContext<B> context) {
        MemoryAccess<B> memoryAccess = context.memoryAccess();
        if (memoryAccess == null) {
            return;
        }

        Shape shape = Shape.of(2, 3);
        MemoryView<B> base = MemoryHelpers.arange(context, DataType.FP32, shape.size());
        MemoryView<B> view = MemoryView.of(base.memory(), DataType.FP32, Layout.columnMajor(shape));

        for (long i = 0; i < shape.size(0); i++) {
            for (long j = 0; j < shape.size(1); j++) {
                long expected = i + j * shape.size(0);
                long offset = Indexing.coordToOffset(view, i, j);
                float actual = memoryAccess.readFloat(view.memory(), offset);
                assertEquals((float) expected, actual);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void rowMajorCoordinatesMatchLinearOrder3d(MemoryContext<B> context) {
        MemoryAccess<B> memoryAccess = context.memoryAccess();
        if (memoryAccess == null) {
            return;
        }

        Shape shape = Shape.of(2, 3, 4);
        MemoryView<B> base = MemoryHelpers.arange(context, DataType.FP32, shape.size());
        MemoryView<B> view = MemoryView.of(base.memory(), DataType.FP32, Layout.rowMajor(shape));

        for (long i = 0; i < shape.size(0); i++) {
            for (long j = 0; j < shape.size(1); j++) {
                for (long k = 0; k < shape.size(2); k++) {
                    long expected = (i * shape.size(1) + j) * shape.size(2) + k;
                    long offset = Indexing.coordToOffset(view, i, j, k);
                    float actual = memoryAccess.readFloat(view.memory(), offset);
                    assertEquals((float) expected, actual);
                }
            }
        }
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void columnMajorCoordinatesMatchColumnOrder3d(MemoryContext<B> context) {
        MemoryAccess<B> memoryAccess = context.memoryAccess();
        if (memoryAccess == null) {
            return;
        }

        Shape shape = Shape.of(2, 3, 4);
        MemoryView<B> base = MemoryHelpers.arange(context, DataType.FP32, shape.size());
        MemoryView<B> view = MemoryView.of(base.memory(), 0L, DataType.FP32, Layout.columnMajor(shape));

        for (long i = 0; i < shape.size(0); i++) {
            for (long j = 0; j < shape.size(1); j++) {
                for (long k = 0; k < shape.size(2); k++) {
                    long expected = i + j * shape.size(0) + k * shape.size(0) * shape.size(1);
                    long offset = Indexing.coordToOffset(view, i, j, k);
                    float actual = memoryAccess.readFloat(view.memory(), offset);
                    assertEquals((float) expected, actual);
                }
            }
        }
    }
}
