package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.AbstractMemoryTest;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import java.util.List;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class MinReductionOpTest extends AbstractMemoryTest {

    private static final List<DataType> PRIMITIVE_TYPES =
            List.of(
                    DataType.BOOL,
                    DataType.I8,
                    DataType.I16,
                    DataType.I32,
                    DataType.I64,
                    DataType.FP16,
                    DataType.BF16,
                    DataType.FP32,
                    DataType.FP64);

    @Test
    void reducesMinAlongAxisForAllTypes() {
        Shape shape = Shape.of(2, 3);
        for (DataType dataType : PRIMITIVE_TYPES) {
            Tensor input = range(dataType, shape);
            Tensor reduced = Tracer.trace(input, t -> t.min(1));
            MemoryView<?> output = reduced.materialize();
            assertEquals(Shape.of(2), output.shape());
            assertValueEquals(
                    dataType,
                    expectedMin(dataType, 0, 3),
                    TensorTestReads.readValue(reduced, 0, dataType));
            assertValueEquals(
                    dataType,
                    expectedMin(dataType, 3, 3),
                    TensorTestReads.readValue(reduced, 1, dataType));
        }
    }

    @Test
    void reducesMultipleAxes() {
        Tensor input = range(DataType.FP32, Shape.of(2, 2, 2));
        Tensor reduced = Tracer.trace(input, t -> t.min(1, 2));
        MemoryView<?> output = reduced.materialize();
        assertEquals(Shape.of(2), output.shape());
        assertValueEquals(
                DataType.FP32, 0.0f, TensorTestReads.readValue(reduced, 0, DataType.FP32));
        assertValueEquals(
                DataType.FP32, 4.0f, TensorTestReads.readValue(reduced, 1, DataType.FP32));
    }

    @Test
    void reducesFullShapeToScalarForMin() {
        Tensor input = range(DataType.FP32, Shape.of(2, 3));
        Tensor reduced = input.min();
        MemoryView<?> output = reduced.materialize();
        assertEquals(Shape.scalar(), output.shape());
        assertValueEquals(
                DataType.FP32, 0.0f, TensorTestReads.readValue(reduced, 0, DataType.FP32));
    }

    @Test
    void wrapsNegativeAxisForMin() {
        Tensor input = range(DataType.FP32, Shape.of(2, 2, 2));
        Tensor reduced = input.min(-1);
        MemoryView<?> output = reduced.materialize();
        assertEquals(Shape.of(2, 2), output.shape());
        assertValueEquals(
                DataType.FP32, 0.0f, TensorTestReads.readValue(reduced, 0, DataType.FP32));
        assertValueEquals(
                DataType.FP32, 2.0f, TensorTestReads.readValue(reduced, 1, DataType.FP32));
        assertValueEquals(
                DataType.FP32, 4.0f, TensorTestReads.readValue(reduced, 2, DataType.FP32));
        assertValueEquals(
                DataType.FP32, 6.0f, TensorTestReads.readValue(reduced, 3, DataType.FP32));
    }

    private Tensor range(DataType dataType, Shape shape) {
        if (dataType == DataType.BOOL) {
            return Tensor.full(1L, shape).cast(DataType.BOOL);
        }
        return Tensor.iota(shape.size(), DataType.I64).cast(dataType).view(shape);
    }

    private void assertValueEquals(DataType dataType, Object expected, Object actual) {
        if (dataType.isFloatingPoint()) {
            double expectedDouble = ((Number) expected).doubleValue();
            double actualDouble = ((Number) actual).doubleValue();
            assertEquals(expectedDouble, actualDouble, 1e-4, "Mismatch for " + dataType);
        } else {
            assertEquals(expected, actual, "Mismatch for " + dataType);
        }
    }

    private Object expectedMin(DataType dataType, int start, int length) {
        int value = start;
        if (dataType.isFloatingPoint()) {
            return (double) value;
        }
        if (dataType == DataType.BOOL) {
            return (byte) 1;
        }
        if (dataType == DataType.I16 || dataType == DataType.FP16 || dataType == DataType.BF16) {
            return (short) value;
        }
        if (dataType == DataType.I32) {
            return value;
        }
        if (dataType == DataType.I64) {
            return (long) value;
        }
        return (byte) value;
    }
}
