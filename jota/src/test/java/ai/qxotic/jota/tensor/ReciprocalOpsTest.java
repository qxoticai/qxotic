package ai.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.AbstractMemoryTest;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.memory.impl.ContextFactory;
import ai.qxotic.jota.memory.impl.MemoryFactory;
import ai.qxotic.jota.memory.impl.MemoryViewFactory;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class ReciprocalOpsTest extends AbstractMemoryTest {

    private static MemoryContext<MemorySegment> context;

    @BeforeAll
    static void setUpContext() {
        context = ContextFactory.ofMemorySegment();
    }

    @Test
    void reciprocalPreservesDataType() {
        Shape shape = Shape.of(2, 2);
        List<DataType> floatingTypes = List.of(DataType.FP32, DataType.FP64);

        for (DataType dataType : floatingTypes) {
            if (dataType == DataType.FP32) {
                Tensor input = Tensor.of(new float[] {1.0f, 2.0f, 4.0f, 8.0f}, shape);
                Tensor result = Tracer.trace(input, Tensor::reciprocal);
                MemoryView<?> output = result.materialize();

                assertEquals(DataType.FP32, output.dataType());
                assertEquals(shape, output.shape());
                assertEquals(1.0f, readFloat(output, 0), 0.0001f);
                assertEquals(0.5f, readFloat(output, 1), 0.0001f);
                assertEquals(0.25f, readFloat(output, 2), 0.0001f);
                assertEquals(0.125f, readFloat(output, 3), 0.0001f);
            } else if (dataType == DataType.FP64) {
                Tensor input = Tensor.of(new double[] {1.0, 2.0, 4.0, 8.0}, shape);
                Tensor result = Tracer.trace(input, Tensor::reciprocal);
                MemoryView<?> output = result.materialize();

                assertEquals(DataType.FP64, output.dataType());
                assertEquals(shape, output.shape());
                assertEquals(1.0, readDouble(output, 0), 0.0001);
                assertEquals(0.5, readDouble(output, 1), 0.0001);
                assertEquals(0.25, readDouble(output, 2), 0.0001);
                assertEquals(0.125, readDouble(output, 3), 0.0001);
            }
        }
    }

    @Test
    void reciprocalWithFloatInput() {
        Tensor input = Tensor.of(new float[] {1.0f, 2.0f, 0.5f, 4.0f});
        Tensor result = Tracer.trace(input, Tensor::reciprocal);
        MemoryView<?> output = result.materialize();

        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.of(4), output.shape());
        assertEquals(1.0f, readFloat(output, 0), 0.0001f);
        assertEquals(0.5f, readFloat(output, 1), 0.0001f);
        assertEquals(2.0f, readFloat(output, 2), 0.0001f);
        assertEquals(0.25f, readFloat(output, 3), 0.0001f);
    }

    @Test
    void reciprocalWithDoubleInput() {
        Tensor input = Tensor.of(new double[] {1.0, 2.0, 4.0});
        Tensor result = Tracer.trace(input, Tensor::reciprocal);
        MemoryView<?> output = result.materialize();

        assertEquals(DataType.FP64, output.dataType());
        assertEquals(Shape.of(3), output.shape());
        assertEquals(1.0, readDouble(output, 0), 0.0001);
        assertEquals(0.5, readDouble(output, 1), 0.0001);
        assertEquals(0.25, readDouble(output, 2), 0.0001);
    }

    @Test
    void reciprocalWithNegativeValues() {
        Tensor input = Tensor.of(new float[] {-1.0f, -2.0f, -4.0f});
        Tensor result = Tracer.trace(input, Tensor::reciprocal);
        MemoryView<?> output = result.materialize();

        assertEquals(-1.0f, readFloat(output, 0), 0.0001f);
        assertEquals(-0.5f, readFloat(output, 1), 0.0001f);
        assertEquals(-0.25f, readFloat(output, 2), 0.0001f);
    }

    @Test
    void reciprocalWithSmallValues() {
        Shape shape = Shape.of(2, 2);
        Tensor input = Tensor.of(new float[] {0.1f, 0.01f, 0.25f, 0.5f}, shape);
        Tensor result = Tracer.trace(input, Tensor::reciprocal);
        MemoryView<?> output = result.materialize();

        assertEquals(10.0f, readFloat(output, 0), 0.001f);
        assertEquals(100.0f, readFloat(output, 1), 0.01f);
        assertEquals(4.0f, readFloat(output, 2), 0.0001f);
        assertEquals(2.0f, readFloat(output, 3), 0.0001f);
    }

    @Test
    void reciprocalWithScalar() {
        Tensor input = Tensor.of(new float[] {5.0f});
        Tensor result = Tracer.trace(input, Tensor::reciprocal);
        MemoryView<?> output = result.materialize();

        assertEquals(Shape.of(1), output.shape());
        assertEquals(0.2f, readFloat(output, 0), 0.0001f);
    }

    @Test
    void reciprocalWithScalar2() {
        Tensor input = Tensor.of(new float[] {5.0f}).view(Shape.scalar());
        Tensor result = Tracer.trace(input, Tensor::reciprocal);
        MemoryView<?> output = result.materialize();

        assertEquals(Shape.scalar(), output.shape());
        assertEquals(0.2f, readFloat(output, 0), 0.0001f);
    }

    @Test
    void reciprocalWithScalar3() {
        Tensor input = Tensor.scalar(5.0f);
        Tensor result = Tracer.trace(input, Tensor::reciprocal);
        MemoryView<?> output = result.materialize();

        assertEquals(Shape.scalar(), output.shape());
        assertEquals(0.2f, readFloat(output, 0), 0.0001f);
    }

    @Test
    void reciprocalThrowsForIntegerType() {
        Tensor intTensor = Tensor.of(new int[] {1, 2, 4, 8});
        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, intTensor::reciprocal);
        assertTrue(ex.getMessage().contains("reciprocal requires floating-point tensor"));
        assertTrue(ex.getMessage().contains(DataType.I32.toString()));
    }

    @Test
    void reciprocalThrowsForBoolType() {
        Shape shape = Shape.of(2);
        MemoryView<MemorySegment> view =
                MemoryViewFactory.rowMajor(
                        DataType.BOOL,
                        MemoryFactory.ofMemorySegment(MemorySegment.ofArray(new byte[] {1, 0})),
                        shape);
        Tensor boolTensor = Tensor.of(view);

        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, boolTensor::reciprocal);
        assertTrue(ex.getMessage().contains("reciprocal requires floating-point tensor"));
        assertTrue(ex.getMessage().contains(DataType.BOOL.toString()));
    }

    @Test
    void reciprocalWithTargetDataType() {
        Shape shape = Shape.of(2, 2);
        Tensor input = Tensor.of(new float[] {1.0f, 2.0f, 4.0f, 8.0f}, shape);
        Tensor result = Tracer.trace(input, t -> t.reciprocal(DataType.FP64));
        MemoryView<?> output = result.materialize();

        assertEquals(DataType.FP64, output.dataType());
        assertEquals(shape, output.shape());
        assertEquals(1.0, readDouble(output, 0), 0.0001);
        assertEquals(0.5, readDouble(output, 1), 0.0001);
        assertEquals(0.25, readDouble(output, 2), 0.0001);
        assertEquals(0.125, readDouble(output, 3), 0.0001);
    }

    @Test
    void reciprocalWithTargetDataTypeFromFP32ToFP64() {
        Tensor input = Tensor.of(new float[] {2.0f, 4.0f});
        Tensor result = Tracer.trace(input, t -> t.reciprocal(DataType.FP64));
        MemoryView<?> output = result.materialize();

        assertEquals(DataType.FP64, output.dataType());
        assertEquals(0.5, readDouble(output, 0), 0.001);
        assertEquals(0.25, readDouble(output, 1), 0.001);
    }

    @Test
    void reciprocalWithTargetDataTypeThrowsForNonFloat() {
        Tensor input = Tensor.of(new float[] {1.0f, 2.0f});
        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> input.reciprocal(DataType.I32));
        assertTrue(ex.getMessage().contains("reciprocal target type must be floating-point"));
        assertTrue(ex.getMessage().contains(DataType.I32.toString()));
    }

    @Test
    void reciprocalWithTargetDataTypeThrowsForBool() {
        Tensor input = Tensor.of(new float[] {2.0f});
        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> input.reciprocal(DataType.BOOL));
        assertTrue(ex.getMessage().contains("reciprocal target type must be floating-point"));
        assertTrue(ex.getMessage().contains(DataType.BOOL.toString()));
    }

    @Test
    void reciprocalInComplexExpression() {
        Shape shape = Shape.of(2, 2);
        Tensor input = Tensor.of(new float[] {1.0f, 2.0f, 4.0f, 8.0f}, shape);
        Tensor result = Tracer.trace(input, t -> t.reciprocal().add(1.0f));
        MemoryView<?> output = result.materialize();

        assertEquals(2.0f, readFloat(output, 0), 0.0001f);
        assertEquals(1.5f, readFloat(output, 1), 0.0001f);
        assertEquals(1.25f, readFloat(output, 2), 0.0001f);
        assertEquals(1.125f, readFloat(output, 3), 0.0001f);
    }

    @Test
    void reciprocalChainedWithOtherOperations() {
        Tensor input = Tensor.of(new float[] {2.0f, 4.0f, 8.0f});
        Tensor result = Tracer.trace(input, t -> t.reciprocal().multiply(2.0f).add(1.0f));
        MemoryView<?> output = result.materialize();

        assertEquals(2.0f, readFloat(output, 0), 0.0001f);
        assertEquals(1.5f, readFloat(output, 1), 0.0001f);
        assertEquals(1.25f, readFloat(output, 2), 0.0001f);
    }

    private float readFloat(MemoryView<?> view, long linearIndex) {
        long offset = Indexing.linearToOffset(view, linearIndex);
        MemorySegment segment = (MemorySegment) view.memory().base();
        return segment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, offset);
    }

    private double readDouble(MemoryView<?> view, long linearIndex) {
        long offset = Indexing.linearToOffset(view, linearIndex);
        MemorySegment segment = (MemorySegment) view.memory().base();
        return segment.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, offset);
    }
}
