package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import java.util.List;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class SumReductionOpTest {

    private record SumCase(DataType inputType, DataType accumulatorType) {}

    private static final List<SumCase> SUM_CASES =
            List.of(
                    new SumCase(DataType.BOOL, DataType.I32),
                    new SumCase(DataType.I8, DataType.I32),
                    new SumCase(DataType.I16, DataType.FP32),
                    new SumCase(DataType.I32, DataType.I64),
                    new SumCase(DataType.FP16, DataType.FP32),
                    new SumCase(DataType.BF16, DataType.FP32),
                    new SumCase(DataType.FP64, DataType.FP64));

    @Test
    void reducesSumAlongAxisWithAccumulators() {
        Shape shape = Shape.of(2, 3);
        for (SumCase sumCase : SUM_CASES) {
            Tensor input = range(sumCase.inputType(), shape);
            Tensor reduced = Tracer.trace(input, t -> t.sum(sumCase.accumulatorType(), 1));
            MemoryView<?> output = reduced.materialize();
            assertEquals(Shape.of(2), output.shape());
            assertValueEquals(
                    sumCase.accumulatorType(),
                    expectedSum(sumCase.inputType(), sumCase.accumulatorType(), 0, 3),
                    TensorTestReads.readValue(reduced, 0, sumCase.accumulatorType()));
            assertValueEquals(
                    sumCase.accumulatorType(),
                    expectedSum(sumCase.inputType(), sumCase.accumulatorType(), 3, 3),
                    TensorTestReads.readValue(reduced, 1, sumCase.accumulatorType()));
        }
    }

    @Test
    void reducesMultipleAxes() {
        Shape shape = Shape.of(2, 2, 2);
        Tensor input = range(DataType.I32, shape);
        Tensor reduced = Tracer.trace(input, t -> t.sum(DataType.I64, 1, 2));
        MemoryView<?> output = reduced.materialize();
        assertEquals(Shape.of(2), output.shape());
        assertValueEquals(DataType.I64, 6L, TensorTestReads.readValue(reduced, 0, DataType.I64));
        assertValueEquals(DataType.I64, 22L, TensorTestReads.readValue(reduced, 1, DataType.I64));
    }

    @Test
    void keepsDimsWhenRequested() {
        Tensor input = range(DataType.FP32, Shape.of(2, 3));
        Tensor reduced = Tracer.trace(input, t -> t.sum(DataType.FP32, true, 1));
        MemoryView<?> output = reduced.materialize();
        assertEquals(Shape.of(2, 1), output.shape());
    }

    @Test
    void reducesExpressionInputs() {
        Tensor input = range(DataType.FP32, Shape.of(2, 3));
        Tensor reduced = Tracer.trace(input, t -> t.add(1f).sum(DataType.FP32, 1));
        assertValueEquals(
                DataType.FP32, 6.0f, TensorTestReads.readValue(reduced, 0, DataType.FP32));
        assertValueEquals(
                DataType.FP32, 15.0f, TensorTestReads.readValue(reduced, 1, DataType.FP32));
    }

    @Test
    void appliesPostReductionOps() {
        Tensor input = range(DataType.I32, Shape.of(2, 3));
        Tensor reduced =
                Tracer.trace(input, t -> t.sum(DataType.I32, 1).cast(DataType.FP32).add(1f));
        MemoryView<?> output = reduced.materialize();
        assertEquals(Shape.of(2), output.shape());
        assertValueEquals(
                DataType.FP32, 4.0f, TensorTestReads.readValue(reduced, 0, DataType.FP32));
        assertValueEquals(
                DataType.FP32, 13.0f, TensorTestReads.readValue(reduced, 1, DataType.FP32));
    }

    @Test
    void wrapsNegativeAxis() {
        Tensor input = range(DataType.I32, Shape.of(2, 3));
        Tensor reduced = Tracer.trace(input, t -> t.sum(DataType.I64, -1));
        MemoryView<?> output = reduced.materialize();
        assertEquals(Shape.of(2), output.shape());
        assertValueEquals(DataType.I64, 3L, TensorTestReads.readValue(reduced, 0, DataType.I64));
        assertValueEquals(DataType.I64, 12L, TensorTestReads.readValue(reduced, 1, DataType.I64));
    }

    @Test
    void sumsBoolWithIntAccumulators() {
        Shape shape = Shape.of(2, 3);
        Tensor input = Tensor.of(new boolean[] {true, false, true, false, true, false}, shape);
        Tensor reducedI32 = Tracer.trace(input, t -> t.sum(DataType.I32, 1));
        MemoryView<?> outputI32 = reducedI32.materialize();
        assertEquals(Shape.of(2), outputI32.shape());
        assertValueEquals(DataType.I32, 2, TensorTestReads.readValue(reducedI32, 0, DataType.I32));
        assertValueEquals(DataType.I32, 1, TensorTestReads.readValue(reducedI32, 1, DataType.I32));

        Tensor reducedI64 = Tracer.trace(input, t -> t.sum(DataType.I64, 1));
        MemoryView<?> outputI64 = reducedI64.materialize();
        assertEquals(Shape.of(2), outputI64.shape());
        assertValueEquals(DataType.I64, 2L, TensorTestReads.readValue(reducedI64, 0, DataType.I64));
        assertValueEquals(DataType.I64, 1L, TensorTestReads.readValue(reducedI64, 1, DataType.I64));
    }

    @Test
    void reduces3dLastAxis() {
        Tensor input = range(DataType.I32, Shape.of(2, 2, 3));
        Tensor reduced = Tracer.trace(input, t -> t.sum(DataType.I64, -1));
        MemoryView<?> output = reduced.materialize();
        assertEquals(Shape.of(2, 2), output.shape());
        assertValueEquals(DataType.I64, 3L, TensorTestReads.readValue(reduced, 0, DataType.I64));
        assertValueEquals(DataType.I64, 12L, TensorTestReads.readValue(reduced, 1, DataType.I64));
        assertValueEquals(DataType.I64, 21L, TensorTestReads.readValue(reduced, 2, DataType.I64));
        assertValueEquals(DataType.I64, 30L, TensorTestReads.readValue(reduced, 3, DataType.I64));
    }

    @Test
    void reduces3dFirstAxis() {
        Tensor input = range(DataType.I32, Shape.of(2, 2, 3));
        Tensor reduced = Tracer.trace(input, t -> t.sum(DataType.I64, 0));
        MemoryView<?> output = reduced.materialize();
        assertEquals(Shape.of(2, 3), output.shape());
        assertValueEquals(DataType.I64, 6L, TensorTestReads.readValue(reduced, 0, DataType.I64));
        assertValueEquals(DataType.I64, 8L, TensorTestReads.readValue(reduced, 1, DataType.I64));
        assertValueEquals(DataType.I64, 10L, TensorTestReads.readValue(reduced, 2, DataType.I64));
        assertValueEquals(DataType.I64, 12L, TensorTestReads.readValue(reduced, 3, DataType.I64));
        assertValueEquals(DataType.I64, 14L, TensorTestReads.readValue(reduced, 4, DataType.I64));
        assertValueEquals(DataType.I64, 16L, TensorTestReads.readValue(reduced, 5, DataType.I64));
    }

    @Test
    void reducesFullShapeToScalar() {
        Tensor input = range(DataType.I32, Shape.of(2, 2));
        Tensor reduced = input.sum(DataType.I64);
        MemoryView<?> output = reduced.materialize();

        assertEquals(Shape.scalar(), output.shape());
        assertValueEquals(DataType.I64, 6L, TensorTestReads.readValue(reduced, 0, DataType.I64));
    }

    @Test
    void usesZeroForEmptyReduction() {
        Tensor input = range(DataType.I32, Shape.of(2, 0));
        Tensor reduced = input.sum(DataType.I64, 1);
        MemoryView<?> output = reduced.materialize();

        assertEquals(Shape.of(2), output.shape());
        assertValueEquals(DataType.I64, 0L, TensorTestReads.readValue(reduced, 0, DataType.I64));
        assertValueEquals(DataType.I64, 0L, TensorTestReads.readValue(reduced, 1, DataType.I64));
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

    private Object expectedSum(
            DataType inputType, DataType accumulatorType, int start, int length) {
        long sum =
                inputType == DataType.BOOL
                        ? length
                        : (long) length * (2L * start + length - 1) / 2L;
        if (accumulatorType == DataType.I32) {
            return (int) sum;
        }
        if (accumulatorType == DataType.I64) {
            return sum;
        }
        if (accumulatorType == DataType.FP32) {
            return (float) sum;
        }
        if (accumulatorType == DataType.FP64) {
            return (double) sum;
        }
        throw new IllegalStateException("Unsupported accumulator type: " + accumulatorType);
    }
}
