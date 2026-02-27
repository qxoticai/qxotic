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
class ArgminOpTest {

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
    void argminAlongAxisOnI32() {
        Tensor input = Tensor.of(new int[] {1, 9, 3, 7, 2, 5}).view(Shape.of(2, 3));

        Tensor argminTensor = input.argmin(1);
        MemoryView<?> argmin = argminTensor.materialize();

        assertEquals(DataType.I64, argmin.dataType());
        assertEquals(Shape.of(2), argmin.shape());
        assertEquals(0L, readLong(argminTensor, 0));
        assertEquals(1L, readLong(argminTensor, 1));
    }

    @Test
    void keepDimsPreservesReducedAxisAsOne() {
        Tensor input = Tensor.of(new float[] {1f, 3f, 2f, 4f, -2f, 0f}).view(Shape.of(2, 3));

        Tensor argminTensor = input.argmin(1, true);
        MemoryView<?> argmin = argminTensor.materialize();

        assertEquals(Shape.of(2, 1), argmin.shape());
        assertEquals(0L, readLong(argminTensor, 0));
        assertEquals(1L, readLong(argminTensor, 1));
    }

    @Test
    void globalArgminReturnsScalarFlattenedIndex() {
        Tensor input = Tensor.of(new int[] {2, 5, 1, 9, 3, 4}).view(Shape.of(2, 3));
        Tensor argminTensor = input.argmin();
        MemoryView<?> argmin = argminTensor.materialize();

        assertEquals(Shape.scalar(), argmin.shape());
        assertEquals(2L, readLong(argminTensor, 0));
    }

    @Test
    void tieBreakUsesFirstOccurrence() {
        Tensor input = Tensor.of(new int[] {5, 1, 5, 5, 1, 1}).view(Shape.of(2, 3));

        Tensor argminTensor = input.argmin(1);

        assertEquals(1L, readLong(argminTensor, 0));
        assertEquals(1L, readLong(argminTensor, 1));
    }

    @Test
    void scalarInputReturnsZeroForGlobalArgmin() {
        Tensor scalar = Tensor.scalar(7.0, DataType.FP64);
        assertEquals(0L, readLong(scalar.argmin(), 0));
    }

    @Test
    void argminRejectsBoolInput() {
        Tensor boolTensor = Tensor.scalar(1L, DataType.BOOL);
        assertThrows(IllegalArgumentException.class, boolTensor::argmin);
    }

    @Test
    void argminRejectsInvalidAxis() {
        Tensor input = Tensor.of(new int[] {1, 2, 3, 4}).view(Shape.of(2, 2));
        assertThrows(IllegalArgumentException.class, () -> input.argmin(-3));
    }

    @Test
    void tracedAndEagerArgminMatch() {
        Tensor input = Tensor.of(new float[] {4f, 1f, 2f, -3f, 5f, 0f}).view(Shape.of(2, 3));

        Tensor eager = input.argmin(1);
        Tensor traced = Tracer.trace(input, t -> t.argmin(1));

        eager.materialize();
        traced.materialize();
        assertEquals(readLong(eager, 0), readLong(traced, 0));
        assertEquals(readLong(eager, 1), readLong(traced, 1));
    }

    @Test
    void argminSupportsAllNumericNonBoolInputTypes() {
        for (DataType type : EXACT_VALUE_TYPES) {
            Tensor input =
                    Tensor.of(new long[] {2L, 7L, 1L, 9L, 3L, 4L}).cast(type).view(Shape.of(2, 3));

            Tensor globalTensor = input.argmin();
            MemoryView<?> globalOut = globalTensor.materialize();

            assertEquals(DataType.I64, globalOut.dataType());
            assertEquals(Shape.scalar(), globalOut.shape());
            assertEquals(2L, readLong(globalTensor, 0));
        }
    }

    @Test
    void argminAlongAxisZeroHasExpectedShapeAndValues() {
        Tensor input = Tensor.of(new int[] {1, 9, 3, 7, 2, 5}).view(Shape.of(2, 3));

        Tensor argminKeepDimsTensor = input.argmin(0, true);
        MemoryView<?> argminKeepDims = argminKeepDimsTensor.materialize();

        assertEquals(Shape.of(1, 3), argminKeepDims.shape());
        assertEquals(0L, readLong(argminKeepDimsTensor, 0));
        assertEquals(1L, readLong(argminKeepDimsTensor, 1));
        assertEquals(0L, readLong(argminKeepDimsTensor, 2));
    }

    @Test
    void argminOnSingletonAxisAlwaysReturnsZeroIndex() {
        Tensor input = Tensor.of(new int[] {9, 7, 5, 3}).view(Shape.of(4, 1));

        Tensor argminTensor = input.argmin(1);
        MemoryView<?> argmin = argminTensor.materialize();

        assertEquals(Shape.of(4), argmin.shape());
        for (int i = 0; i < 4; i++) {
            assertEquals(0L, readLong(argminTensor, i));
        }
    }

    @Test
    void axisOverloadsRejectScalarAxisUsage() {
        Tensor scalar = Tensor.scalar(5L, DataType.I64);
        assertThrows(IllegalArgumentException.class, () -> scalar.argmin(0, true));
    }

    private long readLong(Tensor tensor, long linearIndex) {
        return TensorTestReads.readLong(tensor, linearIndex);
    }
}
