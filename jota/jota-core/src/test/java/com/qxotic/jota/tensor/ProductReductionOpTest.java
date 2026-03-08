package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.AbstractMemoryTest;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryHelpers;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import java.lang.foreign.MemorySegment;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class ProductReductionOpTest extends AbstractMemoryTest {

    private record ProductCase(DataType inputType, DataType accumulatorType) {}

    private static final List<ProductCase> PRODUCT_CASES =
            List.of(
                    new ProductCase(DataType.BOOL, DataType.I32),
                    new ProductCase(DataType.I8, DataType.I32),
                    new ProductCase(DataType.I16, DataType.FP32),
                    new ProductCase(DataType.I32, DataType.I64),
                    new ProductCase(DataType.FP16, DataType.FP32),
                    new ProductCase(DataType.BF16, DataType.FP32),
                    new ProductCase(DataType.FP64, DataType.FP64));

    private static MemoryDomain<MemorySegment> domain;

    @BeforeAll
    static void setUpDomain() {
        domain = Environment.current().nativeMemoryDomain();
    }

    @Test
    void reducesProductAlongAxisWithAccumulators() {
        Shape shape = Shape.of(2, 3);
        for (ProductCase productCase : PRODUCT_CASES) {
            MemoryView<MemorySegment> view = range(productCase.inputType(), shape);
            Tensor input = Tensor.of(view);
            Tensor reduced = Tracer.trace(input, t -> t.product(productCase.accumulatorType(), 1));
            MemoryView<?> output = reduced.materialize();
            assertEquals(Shape.of(2), output.shape());
            assertValueEquals(
                    productCase.accumulatorType(),
                    expectedProduct(productCase.inputType(), productCase.accumulatorType(), 0, 3),
                    readValue(reduced, 0, productCase.accumulatorType()));
            assertValueEquals(
                    productCase.accumulatorType(),
                    expectedProduct(productCase.inputType(), productCase.accumulatorType(), 3, 3),
                    readValue(reduced, 1, productCase.accumulatorType()));
        }
    }

    @Test
    void reducesMultipleAxes() {
        Shape shape = Shape.of(2, 2, 2);
        MemoryView<MemorySegment> view = range(DataType.I32, shape);
        Tensor input = Tensor.of(view);
        Tensor reduced = Tracer.trace(input, t -> t.product(DataType.I64, 1, 2));
        MemoryView<?> output = reduced.materialize();
        assertEquals(Shape.of(2), output.shape());
        assertValueEquals(DataType.I64, 0L, readValue(reduced, 0, DataType.I64));
        assertValueEquals(DataType.I64, 840L, readValue(reduced, 1, DataType.I64));
    }

    @Test
    void keepsDimsWhenRequested() {
        MemoryView<MemorySegment> view = range(DataType.FP32, Shape.of(2, 3));
        Tensor input = Tensor.of(view);
        Tensor reduced = Tracer.trace(input, t -> t.product(DataType.FP32, true, 1));
        MemoryView<?> output = reduced.materialize();
        assertEquals(Shape.of(2, 1), output.shape());
    }

    @Test
    void reducesExpressionInputs() {
        MemoryView<MemorySegment> view = range(DataType.FP32, Shape.of(2, 3));
        Tensor input = Tensor.of(view);
        Tensor reduced = Tracer.trace(input, t -> t.add(1f).product(DataType.FP32, 1));
        assertValueEquals(DataType.FP32, 6.0f, readValue(reduced, 0, DataType.FP32));
        assertValueEquals(DataType.FP32, 120.0f, readValue(reduced, 1, DataType.FP32));
    }

    @Test
    void appliesPostReductionOps() {
        MemoryView<MemorySegment> view = range(DataType.I32, Shape.of(2, 3));
        Tensor input = Tensor.of(view);
        Tensor reduced =
                Tracer.trace(input, t -> t.product(DataType.I32, 1).cast(DataType.FP32).add(1f));
        MemoryView<?> output = reduced.materialize();
        assertEquals(Shape.of(2), output.shape());
        assertValueEquals(DataType.FP32, 1.0f, readValue(reduced, 0, DataType.FP32));
        assertValueEquals(DataType.FP32, 61.0f, readValue(reduced, 1, DataType.FP32));
    }

    @Test
    void wrapsNegativeAxis() {
        MemoryView<MemorySegment> view = range(DataType.I32, Shape.of(2, 3));
        Tensor input = Tensor.of(view);
        Tensor reduced = Tracer.trace(input, t -> t.product(DataType.I64, -1));
        MemoryView<?> output = reduced.materialize();
        assertEquals(Shape.of(2), output.shape());
        assertValueEquals(DataType.I64, 0L, readValue(reduced, 0, DataType.I64));
        assertValueEquals(DataType.I64, 60L, readValue(reduced, 1, DataType.I64));
    }

    @Test
    void reducesFullShapeToScalar() {
        MemoryView<MemorySegment> view = range(DataType.I32, Shape.of(2, 2));
        Tensor input = Tensor.of(view);
        Tensor reduced = Tracer.trace(input, t -> t.product(DataType.I64));
        MemoryView<?> output = reduced.materialize();
        assertEquals(Shape.scalar(), output.shape());
        assertValueEquals(DataType.I64, 0L, readValue(reduced, 0, DataType.I64));
    }

    @Test
    void usesOneForEmptyReduction() {
        MemoryView<MemorySegment> view = range(DataType.I32, Shape.of(2, 0));
        Tensor input = Tensor.of(view);
        Tensor reduced = Tracer.trace(input, t -> t.product(DataType.I64, 1));
        MemoryView<?> output = reduced.materialize();
        assertEquals(Shape.of(2), output.shape());
        assertValueEquals(DataType.I64, 1L, readValue(reduced, 0, DataType.I64));
        assertValueEquals(DataType.I64, 1L, readValue(reduced, 1, DataType.I64));
    }

    private MemoryView<MemorySegment> range(DataType dataType, Shape shape) {
        if (dataType == DataType.BOOL) {
            return MemoryHelpers.full(domain, dataType, shape.size(), 1).view(shape);
        }
        return MemoryHelpers.arange(domain, dataType, shape.size()).view(shape);
    }

    private Object readValue(Tensor tensor, long linearIndex, DataType dataType) {
        return TensorTestReads.readValue(tensor, linearIndex, dataType);
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

    private Object expectedProduct(
            DataType inputType, DataType accumulatorType, int start, int length) {
        double product = 1.0;
        if (inputType == DataType.BOOL) {
            product = length == 0 ? 1.0 : 1.0;
        } else {
            for (int i = 0; i < length; i++) {
                product *= (start + i);
            }
        }
        if (accumulatorType == DataType.I32) {
            return (int) product;
        }
        if (accumulatorType == DataType.I64) {
            return (long) product;
        }
        if (accumulatorType == DataType.FP32) {
            return (float) product;
        }
        if (accumulatorType == DataType.FP64) {
            return product;
        }
        throw new IllegalStateException("Unsupported accumulator type: " + accumulatorType);
    }
}
