package ai.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.*;
import ai.qxotic.jota.memory.impl.ContextFactory;
import ai.qxotic.jota.panama.JavaComputeEngine;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class BooleanOpsTest extends AbstractMemoryTest {

    private static MemoryContext<MemorySegment> context;

    @BeforeAll
    static void setUpContext() {
        context = ContextFactory.ofMemorySegment();
    }

    @Test
    void logicalOpsWorkOnBool() {
        Shape shape = Shape.of(4);
        MemoryView<MemorySegment> aView = boolPattern(shape, new byte[] {1, 0, 1, 0});
        MemoryView<MemorySegment> bView = boolPattern(shape, new byte[] {1, 1, 0, 0});
        Tensor a = Tensor.of(aView);
        Tensor b = Tensor.of(bView);

        Tensor notA = Tracer.trace(a, Tensor::logicalNot);
        Tensor andTensor = Tracer.trace(a, b, Tensor::logicalAnd);
        Tensor orTensor = Tracer.trace(a, b, Tensor::logicalOr);
        Tensor xorTensor = Tracer.trace(a, b, Tensor::logicalXor);

        MemoryView<?> notOut =
                ComputeEngineContext.with(new JavaComputeEngine(context), notA::materialize);
        MemoryView<?> andOut =
                ComputeEngineContext.with(new JavaComputeEngine(context), andTensor::materialize);
        MemoryView<?> orOut =
                ComputeEngineContext.with(new JavaComputeEngine(context), orTensor::materialize);
        MemoryView<?> xorOut =
                ComputeEngineContext.with(new JavaComputeEngine(context), xorTensor::materialize);

        assertEquals(0, readBool(notOut, 0));
        assertEquals(1, readBool(notOut, 1));
        assertEquals(0, readBool(notOut, 2));
        assertEquals(1, readBool(notOut, 3));

        assertEquals(1, readBool(andOut, 0));
        assertEquals(0, readBool(andOut, 1));
        assertEquals(0, readBool(andOut, 2));
        assertEquals(0, readBool(andOut, 3));

        assertEquals(1, readBool(orOut, 0));
        assertEquals(1, readBool(orOut, 1));
        assertEquals(1, readBool(orOut, 2));
        assertEquals(0, readBool(orOut, 3));

        assertEquals(0, readBool(xorOut, 0));
        assertEquals(1, readBool(xorOut, 1));
        assertEquals(1, readBool(xorOut, 2));
        assertEquals(0, readBool(xorOut, 3));
    }

    @Test
    void comparisonsSupportAllTypes() {
        Shape shape = Shape.of(4);
        for (DataType dataType : PRIMITIVE_DATA_TYPES) {
            MemoryView<MemorySegment> left = range(dataType, shape);
            MemoryView<MemorySegment> right = range(dataType, shape);
            Tensor leftTensor = Tensor.of(left);
            Tensor rightTensor = Tensor.of(right);

            Tensor equalTensor = Tracer.trace(leftTensor, rightTensor, Tensor::equal);
            MemoryView<?> equalOut =
                    ComputeEngineContext.with(
                            new JavaComputeEngine(context), equalTensor::materialize);
            for (int i = 0; i < shape.size(); i++) {
                assertEquals(1, readBool(equalOut, i));
            }

            MemoryView<MemorySegment> threshold =
                    dataType == DataType.BOOL
                            ? boolPattern(shape, new byte[] {1, 1, 0, 0})
                            : MemoryHelpers.full(context, dataType, shape.size(), 2).view(shape);
            Tensor thresholdTensor = Tensor.of(threshold);
            Tensor lessThanTensor = Tracer.trace(leftTensor, thresholdTensor, Tensor::lessThan);
            MemoryView<?> lessOut =
                    ComputeEngineContext.with(
                            new JavaComputeEngine(context), lessThanTensor::materialize);

            if (dataType == DataType.BOOL) {
                assertEquals(1, readBool(lessOut, 0));
                assertEquals(0, readBool(lessOut, 1));
                assertEquals(0, readBool(lessOut, 2));
                assertEquals(0, readBool(lessOut, 3));
            } else {
                assertEquals(1, readBool(lessOut, 0));
                assertEquals(1, readBool(lessOut, 1));
                assertEquals(0, readBool(lessOut, 2));
                assertEquals(0, readBool(lessOut, 3));
            }
        }
    }

    @Test
    void fusedWhere() {
        Shape shape = Shape.of(4);
        MemoryView<MemorySegment> trueView =
                MemoryHelpers.arange(context, DataType.I32, shape.size()).view(shape);
        MemoryView<MemorySegment> falseView =
                MemoryHelpers.full(context, DataType.I32, shape.size(), 2).view(shape);

        Tensor trueTensor = Tensor.of(trueView);
        Tensor falseTensor = Tensor.of(falseView);

        Tensor min =
                Tracer.trace(
                        trueTensor,
                        falseTensor,
                        // Odd min function.
                        (t, f) -> Tensor.where(t.lessThan(f), t, f));

        MemoryView<?> output =
                ComputeEngineContext.with(new JavaComputeEngine(context), min::materialize);

        assertEquals(0, readInt(output, 0));
        assertEquals(1, readInt(output, 1));
        assertEquals(2, readInt(output, 2));
        assertEquals(2, readInt(output, 3));
    }

    @Test
    void whereSelectsBetweenValues() {
        Shape shape = Shape.of(4);
        MemoryView<MemorySegment> conditionView = boolPattern(shape, new byte[] {1, 0, 1, 0});
        MemoryView<MemorySegment> trueView =
                MemoryHelpers.arange(context, DataType.I32, shape.size()).view(shape);
        MemoryView<MemorySegment> falseView =
                MemoryHelpers.full(context, DataType.I32, shape.size(), -1).view(shape);

        Tensor condition = Tensor.of(conditionView);
        Tensor trueTensor = Tensor.of(trueView);
        Tensor falseTensor = Tensor.of(falseView);

        Tensor selected = Tracer.trace(condition, trueTensor, falseTensor, Tensor::where);
        MemoryView<?> output =
                ComputeEngineContext.with(new JavaComputeEngine(context), selected::materialize);

        assertEquals(0, readInt(output, 0));
        assertEquals(-1, readInt(output, 1));
        assertEquals(2, readInt(output, 2));
        assertEquals(-1, readInt(output, 3));
    }

    private MemoryView<MemorySegment> range(DataType dataType, Shape shape) {
        if (dataType == DataType.BOOL) {
            return boolPattern(shape, new byte[] {0, 1, 0, 1});
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

    private byte readBool(MemoryView<?> view, long linearIndex) {
        long offset = Indexing.linearToOffset(view, linearIndex);
        MemorySegment segment = (MemorySegment) view.memory().base();
        return segment.get(ValueLayout.JAVA_BYTE, offset);
    }

    private int readInt(MemoryView<?> view, long linearIndex) {
        long offset = Indexing.linearToOffset(view, linearIndex);
        MemorySegment segment = (MemorySegment) view.memory().base();
        return segment.get(ValueLayout.JAVA_INT_UNALIGNED, offset);
    }
}
