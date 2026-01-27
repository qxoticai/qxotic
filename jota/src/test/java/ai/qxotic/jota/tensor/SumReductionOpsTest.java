package ai.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.qxotic.jota.BFloat16;
import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.AbstractMemoryTest;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryHelpers;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.memory.impl.ContextFactory;
import ai.qxotic.jota.panama.JavaComputeEngine;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class SumReductionOpsTest extends AbstractMemoryTest {

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

    private static MemoryContext<MemorySegment> context;

    @BeforeAll
    static void setUpContext() {
        context = ContextFactory.ofMemorySegment();
    }

    @Test
    void reducesSumAlongAxisWithAccumulators() {
        Shape shape = Shape.of(2, 3);
        for (SumCase sumCase : SUM_CASES) {
            MemoryView<MemorySegment> view = range(sumCase.inputType(), shape);
            Tensor input = Tensor.of(view);
            Tensor reduced = Tracer.trace(input, t -> t.sum(sumCase.accumulatorType(), 1));
            MemoryView<?> output =
                    ComputeEngineContext.with(new JavaComputeEngine(context), reduced::materialize);
            assertEquals(Shape.of(2), output.shape());
            assertValueEquals(
                    sumCase.accumulatorType(),
                    expectedSum(sumCase.inputType(), sumCase.accumulatorType(), 0, 3),
                    readValue(output, 0, sumCase.accumulatorType()));
            assertValueEquals(
                    sumCase.accumulatorType(),
                    expectedSum(sumCase.inputType(), sumCase.accumulatorType(), 3, 3),
                    readValue(output, 1, sumCase.accumulatorType()));
        }
    }

    @Test
    void reducesMultipleAxes() {
        Shape shape = Shape.of(2, 2, 2);
        MemoryView<MemorySegment> view = range(DataType.I32, shape);
        Tensor input = Tensor.of(view);
        Tensor reduced = Tracer.trace(input, t -> t.sum(DataType.I64, 1, 2));
        MemoryView<?> output =
                ComputeEngineContext.with(new JavaComputeEngine(context), reduced::materialize);
        assertEquals(Shape.of(2), output.shape());
        assertValueEquals(DataType.I64, 6L, readValue(output, 0, DataType.I64));
        assertValueEquals(DataType.I64, 22L, readValue(output, 1, DataType.I64));
    }

    @Test
    void keepsDimsWhenRequested() {
        MemoryView<MemorySegment> view = range(DataType.FP32, Shape.of(2, 3));
        Tensor input = Tensor.of(view);
        Tensor reduced = Tracer.trace(input, t -> t.sum(DataType.FP32, true, 1));
        MemoryView<?> output =
                ComputeEngineContext.with(new JavaComputeEngine(context), reduced::materialize);
        assertEquals(Shape.of(2, 1), output.shape());
    }

    @Test
    void reducesExpressionInputs() {
        MemoryView<MemorySegment> view = range(DataType.FP32, Shape.of(2, 3));
        Tensor input = Tensor.of(view);
        Tensor reduced = Tracer.trace(input, t -> t.add(1f).sum(DataType.FP32, 1));
        MemoryView<?> output =
                ComputeEngineContext.with(new JavaComputeEngine(context), reduced::materialize);
        assertValueEquals(DataType.FP32, 6.0f, readValue(output, 0, DataType.FP32));
        assertValueEquals(DataType.FP32, 15.0f, readValue(output, 1, DataType.FP32));
    }

    @Test
    void appliesPostReductionOps() {
        MemoryView<MemorySegment> view = range(DataType.I32, Shape.of(2, 3));
        Tensor input = Tensor.of(view);
        Tensor reduced =
                Tracer.trace(input, t -> t.sum(DataType.I32, 1).cast(DataType.FP32).add(1f));
        MemoryView<?> output =
                ComputeEngineContext.with(new JavaComputeEngine(context), reduced::materialize);
        assertEquals(Shape.of(2), output.shape());
        assertValueEquals(DataType.FP32, 4.0f, readValue(output, 0, DataType.FP32));
        assertValueEquals(DataType.FP32, 13.0f, readValue(output, 1, DataType.FP32));
    }

    @Test
    void wrapsNegativeAxis() {
        MemoryView<MemorySegment> view = range(DataType.I32, Shape.of(2, 3));
        Tensor input = Tensor.of(view);
        Tensor reduced = Tracer.trace(input, t -> t.sum(DataType.I64, -1));
        MemoryView<?> output =
                ComputeEngineContext.with(new JavaComputeEngine(context), reduced::materialize);
        assertEquals(Shape.of(2), output.shape());
        assertValueEquals(DataType.I64, 3L, readValue(output, 0, DataType.I64));
        assertValueEquals(DataType.I64, 12L, readValue(output, 1, DataType.I64));
    }

    @Test
    void sumsBoolWithIntAccumulators() {
        Shape shape = Shape.of(2, 3);
        MemoryView<MemorySegment> view = boolPattern(shape, new byte[] {1, 0, 1, 0, 1, 0});
        Tensor input = Tensor.of(view);
        Tensor reducedI32 = Tracer.trace(input, t -> t.sum(DataType.I32, 1));
        MemoryView<?> outputI32 =
                ComputeEngineContext.with(new JavaComputeEngine(context), reducedI32::materialize);
        assertEquals(Shape.of(2), outputI32.shape());
        assertValueEquals(DataType.I32, 2, readValue(outputI32, 0, DataType.I32));
        assertValueEquals(DataType.I32, 1, readValue(outputI32, 1, DataType.I32));

        Tensor reducedI64 = Tracer.trace(input, t -> t.sum(DataType.I64, 1));
        MemoryView<?> outputI64 =
                ComputeEngineContext.with(new JavaComputeEngine(context), reducedI64::materialize);
        assertEquals(Shape.of(2), outputI64.shape());
        assertValueEquals(DataType.I64, 2L, readValue(outputI64, 0, DataType.I64));
        assertValueEquals(DataType.I64, 1L, readValue(outputI64, 1, DataType.I64));
    }

    @Test
    void reduces3dLastAxis() {
        MemoryView<MemorySegment> view = range(DataType.I32, Shape.of(2, 2, 3));
        Tensor input = Tensor.of(view);
        Tensor reduced = Tracer.trace(input, t -> t.sum(DataType.I64, -1));
        MemoryView<?> output =
                ComputeEngineContext.with(new JavaComputeEngine(context), reduced::materialize);
        assertEquals(Shape.of(2, 2), output.shape());
        assertValueEquals(DataType.I64, 3L, readValue(output, 0, DataType.I64));
        assertValueEquals(DataType.I64, 12L, readValue(output, 1, DataType.I64));
        assertValueEquals(DataType.I64, 21L, readValue(output, 2, DataType.I64));
        assertValueEquals(DataType.I64, 30L, readValue(output, 3, DataType.I64));
    }

    @Test
    void reduces3dFirstAxis() {
        MemoryView<MemorySegment> view = range(DataType.I32, Shape.of(2, 2, 3));
        Tensor input = Tensor.of(view);
        Tensor reduced = Tracer.trace(input, t -> t.sum(DataType.I64, 0));
        MemoryView<?> output =
                ComputeEngineContext.with(new JavaComputeEngine(context), reduced::materialize);
        assertEquals(Shape.of(2, 3), output.shape());
        assertValueEquals(DataType.I64, 6L, readValue(output, 0, DataType.I64));
        assertValueEquals(DataType.I64, 8L, readValue(output, 1, DataType.I64));
        assertValueEquals(DataType.I64, 10L, readValue(output, 2, DataType.I64));
        assertValueEquals(DataType.I64, 12L, readValue(output, 3, DataType.I64));
        assertValueEquals(DataType.I64, 14L, readValue(output, 4, DataType.I64));
        assertValueEquals(DataType.I64, 16L, readValue(output, 5, DataType.I64));
    }

    private MemoryView<MemorySegment> range(DataType dataType, Shape shape) {
        if (dataType == DataType.BOOL) {
            return MemoryHelpers.full(context, dataType, shape.size(), 1).view(shape);
        }
        return MemoryHelpers.arange(context, dataType, shape.size()).view(shape);
    }

    private MemoryView<MemorySegment> boolPattern(Shape shape, byte[] values) {
        MemoryView<MemorySegment> view =
                MemoryHelpers.full(context, DataType.BOOL, shape.size(), 0).view(shape);
        MemoryAccess<MemorySegment> access = context.memoryAccess();
        for (int i = 0; i < values.length; i++) {
            long offset = view.byteOffset() + (long) i * DataType.BOOL.byteSize();
            access.writeByte(view.memory(), offset, values[i]);
        }
        return view;
    }

    private Object readValue(MemoryView<?> view, long linearIndex, DataType dataType) {
        long offset = Indexing.linearToOffset(view, linearIndex);
        MemorySegment segment = (MemorySegment) view.memory().base();
        if (dataType == DataType.BOOL || dataType == DataType.I8) {
            return segment.get(ValueLayout.JAVA_BYTE, offset);
        }
        if (dataType == DataType.I16) {
            return segment.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset);
        }
        if (dataType == DataType.I32) {
            return segment.get(ValueLayout.JAVA_INT_UNALIGNED, offset);
        }
        if (dataType == DataType.I64) {
            return segment.get(ValueLayout.JAVA_LONG_UNALIGNED, offset);
        }
        if (dataType == DataType.FP16) {
            return Float.float16ToFloat(segment.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset));
        }
        if (dataType == DataType.BF16) {
            return BFloat16.toFloat(segment.get(ValueLayout.JAVA_SHORT_UNALIGNED, offset));
        }
        if (dataType == DataType.FP32) {
            return segment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, offset);
        }
        if (dataType == DataType.FP64) {
            return segment.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, offset);
        }
        throw new IllegalStateException("Unsupported data type: " + dataType);
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
