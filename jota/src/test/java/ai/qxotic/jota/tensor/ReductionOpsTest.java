package ai.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.qxotic.jota.BFloat16;
import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.AbstractMemoryTest;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryHelpers;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.memory.impl.ContextFactory;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class ReductionOpsTest extends AbstractMemoryTest {

    private static final java.util.List<DataType> PRIMITIVE_TYPES =
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

    private static MemoryContext<MemorySegment> context;

    @BeforeAll
    static void setUpContext() {
        context = ContextFactory.ofMemorySegment();
    }

    @Test
    void reducesMaxAlongAxisForAllTypes() {
        Shape shape = Shape.of(2, 3);
        for (DataType dataType : PRIMITIVE_TYPES) {
            MemoryView<MemorySegment> view = range(dataType, shape);

            Tensor input = Tensor.of(view);
            Tensor reduced = Tracer.trace(input, t -> t.max(1));
            MemoryView<?> output = reduced.materialize();
            assertEquals(Shape.of(2), output.shape());
            assertValueEquals(
                    dataType, expectedMax(dataType, 0, 3), readValue(output, 0, dataType));
            assertValueEquals(
                    dataType, expectedMax(dataType, 3, 3), readValue(output, 1, dataType));
        }
    }

    @Test
    void reducesMinAlongAxisForAllTypes() {
        Shape shape = Shape.of(2, 3);
        for (DataType dataType : PRIMITIVE_TYPES) {
            MemoryView<MemorySegment> view = range(dataType, shape);
            Tensor input = Tensor.of(view);
            Tensor reduced = Tracer.trace(input, t -> t.min(1));
            MemoryView<?> output = reduced.materialize();
            assertEquals(Shape.of(2), output.shape());
            assertValueEquals(
                    dataType, expectedMin(dataType, 0, 3), readValue(output, 0, dataType));
            assertValueEquals(
                    dataType, expectedMin(dataType, 3, 3), readValue(output, 1, dataType));
        }
    }

    @Test
    void keepsDimsWhenRequested() {
        MemoryView<MemorySegment> view = range(DataType.FP32, Shape.of(2, 3));
        Tensor input = Tensor.of(view);
        Tensor reduced = Tracer.trace(input, t -> t.max(true, 1));
        MemoryView<?> output = reduced.materialize();
        assertEquals(Shape.of(2, 1), output.shape());
    }

    @Test
    void reducesMultipleAxes() {
        MemoryView<MemorySegment> view = range(DataType.FP32, Shape.of(2, 2, 2));
        Tensor input = Tensor.of(view);
        Tensor reduced = Tracer.trace(input, t -> t.min(1, 2));
        MemoryView<?> output = reduced.materialize();
        assertEquals(Shape.of(2), output.shape());
        assertValueEquals(DataType.FP32, 0.0f, readValue(output, 0, DataType.FP32));
        assertValueEquals(DataType.FP32, 4.0f, readValue(output, 1, DataType.FP32));
    }

    @Test
    void reducesExpressionInputs() {
        MemoryView<MemorySegment> view = range(DataType.FP32, Shape.of(2, 3));
        Tensor input = Tensor.of(view);
        Tensor reduced = Tracer.trace(input, t -> t.add(1f).max(1));
        MemoryView<?> output = reduced.materialize();
        assertValueEquals(DataType.FP32, 3.0f, readValue(output, 0, DataType.FP32));
        assertValueEquals(DataType.FP32, 6.0f, readValue(output, 1, DataType.FP32));
    }

    private MemoryView<MemorySegment> range(DataType dataType, Shape shape) {
        if (dataType == DataType.BOOL) {
            return MemoryHelpers.full(context, dataType, shape.size(), 1).view(shape);
        }
        return MemoryHelpers.arange(context, dataType, shape.size()).view(shape);
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

    private Object expectedMax(DataType dataType, int start, int length) {
        int value = start + length - 1;
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
