package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import java.util.List;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class ArgmaxOpTest {

    private static final List<DataType> EXACT_VALUE_TYPES =
            List.of(
                    DataType.I16,
                    DataType.I32,
                    DataType.I64,
                    DataType.FP16,
                    DataType.BF16,
                    DataType.FP32,
                    DataType.FP64);

    @Test
    void argmaxAlongAxisOnI32() {
        Tensor input = Tensor.of(new int[] {1, 9, 3, 7, 2, 5}).view(Shape.of(2, 3));

        Tensor argmaxTensor = input.argmax(1);
        MemoryView<?> argmax = argmaxTensor.materialize();

        assertEquals(DataType.I64, argmax.dataType());
        assertEquals(Shape.of(2), argmax.shape());
        assertEquals(1L, readLong(argmaxTensor, 0));
        assertEquals(0L, readLong(argmaxTensor, 1));
    }

    @Test
    void keepDimsPreservesReducedAxisAsOne() {
        Tensor input = Tensor.of(new float[] {1f, 3f, 2f, 4f, -2f, 0f}).view(Shape.of(2, 3));

        Tensor argmaxTensor = input.argmax(1, true);
        MemoryView<?> argmax = argmaxTensor.materialize();

        assertEquals(Shape.of(2, 1), argmax.shape());
        assertEquals(1L, readLong(argmaxTensor, 0));
        assertEquals(0L, readLong(argmaxTensor, 1));
    }

    @Test
    void globalArgmaxReturnsScalarFlattenedIndex() {
        Tensor input = Tensor.of(new int[] {2, 5, 1, 9, 3, 4}).view(Shape.of(2, 3));
        Tensor argmaxTensor = input.argmax();
        MemoryView<?> argmax = argmaxTensor.materialize();

        assertEquals(Shape.scalar(), argmax.shape());
        assertEquals(3L, readLong(argmaxTensor, 0));
    }

    @Test
    void tieBreakUsesFirstOccurrence() {
        Tensor input = Tensor.of(new int[] {5, 1, 5, 5, 1, 1}).view(Shape.of(2, 3));

        Tensor argmaxTensor = input.argmax(1);

        assertEquals(0L, readLong(argmaxTensor, 0));
        assertEquals(0L, readLong(argmaxTensor, 1));
    }

    @Test
    void negativeAxisWrapAroundWorks() {
        Tensor input = Tensor.of(new int[] {1, 3, 2, 4, 0, 9}).view(Shape.of(2, 3));
        Tensor argmaxTensor = input.argmax(-1);
        MemoryView<?> argmax = argmaxTensor.materialize();
        assertEquals(Shape.of(2), argmax.shape());
        assertEquals(1L, readLong(argmaxTensor, 0));
        assertEquals(2L, readLong(argmaxTensor, 1));
    }

    @Test
    void scalarInputReturnsZeroForGlobalArgmax() {
        Tensor scalar = Tensor.scalar(7.0, DataType.FP64);
        assertEquals(0L, readLong(scalar.argmax(), 0));
    }

    @Test
    void argmaxRejectsBoolInput() {
        Tensor boolTensor = Tensor.scalar(1L, DataType.BOOL);
        assertThrows(IllegalArgumentException.class, boolTensor::argmax);
    }

    @Test
    void argmaxRejectsInvalidAxis() {
        Tensor input = Tensor.of(new int[] {1, 2, 3, 4}).view(Shape.of(2, 2));
        assertThrows(IllegalArgumentException.class, () -> input.argmax(2));
    }

    @Test
    void argmaxWorksForNonContiguousInputs() {
        Tensor base = Tensor.iota(12, DataType.I32).view(Shape.of(3, 4));
        Tensor nonContiguous = base.transpose(0, 1);
        Tensor outTensor = nonContiguous.argmax(1);
        MemoryView<?> out = outTensor.materialize();

        assertEquals(Shape.of(4), out.shape());
        assertEquals(2L, readLong(outTensor, 0));
        assertEquals(2L, readLong(outTensor, 1));
        assertEquals(2L, readLong(outTensor, 2));
        assertEquals(2L, readLong(outTensor, 3));
    }

    @Test
    void tracedAndEagerArgmaxMatch() {
        Tensor input = Tensor.of(new float[] {1f, 5f, 2f, 9f, 3f, 4f}).view(Shape.of(2, 3));

        Tensor eager = input.argmax(1);
        Tensor traced = Tracer.trace(input, t -> t.argmax(1));

        eager.materialize();
        traced.materialize();
        assertEquals(readLong(eager, 0), readLong(traced, 0));
        assertEquals(readLong(eager, 1), readLong(traced, 1));
    }

    @Test
    void argmaxSupportsAllNumericNonBoolInputTypes() {
        for (DataType type : EXACT_VALUE_TYPES) {
            Tensor input =
                    Tensor.of(new long[] {2L, 7L, 1L, 9L, 3L, 4L}).cast(type).view(Shape.of(2, 3));

            Tensor axisTensor = input.argmax(1);
            MemoryView<?> axisOut = axisTensor.materialize();

            assertEquals(DataType.I64, axisOut.dataType());
            assertEquals(Shape.of(2), axisOut.shape());
            assertEquals(1L, readLong(axisTensor, 0));
            assertEquals(0L, readLong(axisTensor, 1));
        }
    }

    @Test
    void argmaxAlongAxisZeroHasExpectedShapeAndValues() {
        Tensor input = Tensor.of(new int[] {1, 9, 3, 7, 2, 5}).view(Shape.of(2, 3));

        Tensor argmaxTensor = input.argmax(0);
        MemoryView<?> argmax = argmaxTensor.materialize();

        assertEquals(Shape.of(3), argmax.shape());
        assertEquals(1L, readLong(argmaxTensor, 0));
        assertEquals(0L, readLong(argmaxTensor, 1));
        assertEquals(1L, readLong(argmaxTensor, 2));
    }

    @Test
    void argmaxOnSingletonAxisAlwaysReturnsZeroIndex() {
        Tensor input = Tensor.of(new int[] {9, 7, 5, 3}).view(Shape.of(4, 1));

        Tensor argmaxTensor = input.argmax(1);
        MemoryView<?> argmax = argmaxTensor.materialize();

        assertEquals(Shape.of(4), argmax.shape());
        for (int i = 0; i < 4; i++) {
            assertEquals(0L, readLong(argmaxTensor, i));
        }
    }

    @Test
    void axisOverloadsRejectScalarAxisUsage() {
        Tensor scalar = Tensor.scalar(5L, DataType.I64);
        assertThrows(IllegalArgumentException.class, () -> scalar.argmax(0));
    }

    private long readLong(Tensor tensor, long linearIndex) {
        return TensorTestReads.readLong(tensor, linearIndex);
    }
}
