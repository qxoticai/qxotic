package com.qxotic.jota.tensor.contracts;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class ReductionSemanticsContractTest {

    @Test
    void lazyReductionsHandleKeepDimsAndMean() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));

        Tensor sumKeepDims = input.sum(DataType.FP32, true, 1);
        Tensor mean = input.mean(1);

        MemoryView<?> sumView = sumKeepDims.materialize();
        MemoryView<?> meanView = mean.materialize();

        assertEquals(Shape.of(2, 1), sumView.shape());
        assertEquals(3.0f, readFloat(sumKeepDims, 0), 1e-4f);
        assertEquals(12.0f, readFloat(sumKeepDims, 1), 1e-4f);

        assertEquals(Shape.of(2), meanView.shape());
        assertEquals(1.0f, readFloat(mean, 0), 1e-4f);
        assertEquals(4.0f, readFloat(mean, 1), 1e-4f);
    }

    @Test
    void lazyReductionsNormalizeDuplicateAxes() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));

        Tensor reduced = input.sum(DataType.FP32, true, 1, -1);
        MemoryView<?> output = reduced.materialize();

        assertEquals(Shape.of(2, 1), output.shape());
        assertEquals(3.0f, readFloat(reduced, 0), 1e-4f);
        assertEquals(12.0f, readFloat(reduced, 1), 1e-4f);
    }

    @Test
    void lazyReductionsForwardAxisAndKeepDims() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));
        Tensor reduced = input.max(true, 1);
        MemoryView<?> output = reduced.materialize();

        assertEquals(Shape.of(2, 1), output.shape());
        assertEquals(2.0f, readFloat(reduced, 0), 1e-4f);
        assertEquals(5.0f, readFloat(reduced, 1), 1e-4f);
    }

    @Test
    void lazyMeanSupportsFloatingInputs() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));
        Tensor mean = input.mean(true, 1);
        MemoryView<?> output = mean.materialize();

        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.of(2, 1), output.shape());
        assertEquals(1.0f, readFloat(mean, 0), 1e-4f);
        assertEquals(4.0f, readFloat(mean, 1), 1e-4f);
    }

    @Test
    void lazyMeanReducesAllAxesToScalar() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));
        Tensor mean = input.mean();
        MemoryView<?> output = mean.materialize();

        assertEquals(Shape.scalar(), output.shape());
        assertEquals(2.5f, readFloat(mean, 0), 1e-4f);
    }

    @Test
    void lazyMeanSupportsMultiAxisWithWrapAround() {
        Tensor input = Tensor.iota(8, DataType.FP32).view(Shape.of(2, 2, 2));
        Tensor mean = input.mean(false, 1, -1);
        MemoryView<?> output = mean.materialize();

        assertEquals(Shape.of(2), output.shape());
        assertEquals(1.5f, readFloat(mean, 0), 1e-4f);
        assertEquals(5.5f, readFloat(mean, 1), 1e-4f);
    }

    @Test
    void lazyMeanRejectsNonFloatingInputs() {
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Tensor input = Tensor.iota(6, DataType.I32).view(Shape.of(2, 3));
                    input.mean(1).materialize();
                });
    }

    @Test
    void reductionsValidateAccumulatorTypeStrictly() {
        Tensor boolInput = Tensor.full(1L, DataType.BOOL, Shape.of(2, 2));
        Tensor i32Input = Tensor.iota(4, DataType.I32).view(Shape.of(2, 2));

        assertThrows(
                IllegalArgumentException.class,
                () -> boolInput.sum(DataType.BOOL, 1).materialize());

        assertThrows(
                IllegalArgumentException.class, () -> i32Input.sum(DataType.FP32, 1).materialize());

        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Tensor lazyInput = Tensor.iota(4, DataType.I32).view(Shape.of(2, 2));
                    lazyInput.sum(DataType.FP32, 1).materialize();
                });
    }

    private static float readFloat(Tensor tensor, long linearIndex) {
        return TensorTestReads.readFloat(tensor, linearIndex);
    }
}
