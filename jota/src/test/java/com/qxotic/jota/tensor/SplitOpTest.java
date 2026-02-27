package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class SplitOpTest {

    @Test
    void splitAlongAxisWithExactSizes() {
        Tensor input = Tensor.iota(12, DataType.FP32).view(Shape.of(2, 6));

        Tensor[] parts = Tensor.split(1, input, 2, 1, 3);

        assertEquals(3, parts.length);
        assertEquals(Shape.of(2, 2), parts[0].shape());
        assertEquals(Shape.of(2, 1), parts[1].shape());
        assertEquals(Shape.of(2, 3), parts[2].shape());
    }

    @Test
    void splitSupportsSingleInferenceMinusOne() {
        Tensor input = Tensor.iota(20, DataType.FP32).view(Shape.of(2, 10));

        Tensor[] parts = Tensor.split(1, input, 3, -1, 2);

        assertEquals(3, parts.length);
        assertEquals(Shape.of(2, 3), parts[0].shape());
        assertEquals(Shape.of(2, 5), parts[1].shape());
        assertEquals(Shape.of(2, 2), parts[2].shape());
    }

    @Test
    void splitSupportsNegativeAxisWrapAround() {
        Tensor input = Tensor.iota(24, DataType.FP32).view(Shape.of(2, 3, 4));

        Tensor[] parts = Tensor.split(-1, input, 1, 3);

        assertEquals(2, parts.length);
        assertEquals(Shape.of(2, 3, 1), parts[0].shape());
        assertEquals(Shape.of(2, 3, 3), parts[1].shape());
    }

    @Test
    void splitProducesExpectedValues() {
        Tensor input = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, Shape.of(1, 6));

        Tensor[] parts = Tensor.split(1, input, 2, 2, 2);

        assertEquals(1f, TensorTestReads.readFloat(parts[0], 0), 1e-4f);
        assertEquals(2f, TensorTestReads.readFloat(parts[0], 1), 1e-4f);
        assertEquals(3f, TensorTestReads.readFloat(parts[1], 0), 1e-4f);
        assertEquals(4f, TensorTestReads.readFloat(parts[1], 1), 1e-4f);
        assertEquals(5f, TensorTestReads.readFloat(parts[2], 0), 1e-4f);
        assertEquals(6f, TensorTestReads.readFloat(parts[2], 1), 1e-4f);
    }

    @Test
    void splitWorksWithNonContiguousInput() {
        Tensor base = Tensor.iota(12, DataType.FP32).view(Shape.of(3, 4));
        Tensor transposed = base.transpose(0, 1);

        Tensor[] parts = Tensor.split(1, transposed, 1, 2);

        assertEquals(Shape.of(4, 1), parts[0].shape());
        assertEquals(Shape.of(4, 2), parts[1].shape());
    }

    @Test
    void splitPreservesNestedModesOutsideSplitAxis() {
        Tensor input = Tensor.iota(30, DataType.FP32).view(Shape.of(2, Shape.of(3, 5)));

        Tensor[] parts = Tensor.split(0, input, 1, 1);

        assertEquals(Shape.of(1, Shape.of(3, 5)), parts[0].shape());
        assertEquals(Shape.of(1, Shape.of(3, 5)), parts[1].shape());
    }

    @Test
    void splitRejectsNestedSplitAxis() {
        Tensor input = Tensor.iota(30, DataType.FP32).view(Shape.of(2, Shape.of(3, 5)));

        assertThrows(IllegalArgumentException.class, () -> Tensor.split(1, input, 5, 10));
    }

    @Test
    void splitRejectsMultipleMinusOneSizes() {
        Tensor input = Tensor.iota(20, DataType.FP32).view(Shape.of(2, 10));

        assertThrows(IllegalArgumentException.class, () -> Tensor.split(1, input, -1, -1));
    }

    @Test
    void splitRejectsZeroAndNegativeSizes() {
        Tensor input = Tensor.iota(20, DataType.FP32).view(Shape.of(2, 10));

        assertThrows(IllegalArgumentException.class, () -> Tensor.split(1, input, 0, 10));
        assertThrows(IllegalArgumentException.class, () -> Tensor.split(1, input, -2, 12));
    }

    @Test
    void splitRejectsSumMismatchWithoutInference() {
        Tensor input = Tensor.iota(20, DataType.FP32).view(Shape.of(2, 10));

        assertThrows(IllegalArgumentException.class, () -> Tensor.split(1, input, 3, 3));
        assertThrows(IllegalArgumentException.class, () -> Tensor.split(1, input, 8, 5));
    }

    @Test
    void splitRejectsInvalidInferredSize() {
        Tensor input = Tensor.iota(20, DataType.FP32).view(Shape.of(2, 10));

        assertThrows(IllegalArgumentException.class, () -> Tensor.split(1, input, 9, -1, 1));
        assertThrows(IllegalArgumentException.class, () -> Tensor.split(1, input, 11, -1));
    }

    @Test
    void splitRejectsOutOfRangeAxis() {
        Tensor input = Tensor.iota(20, DataType.FP32).view(Shape.of(2, 10));

        assertThrows(IllegalArgumentException.class, () -> Tensor.split(2, input, 3, 7));
        assertThrows(IllegalArgumentException.class, () -> Tensor.split(-3, input, 3, 7));
    }

    @Test
    void splitRejectsNullInputs() {
        Tensor input = Tensor.iota(20, DataType.FP32).view(Shape.of(2, 10));

        assertThrows(NullPointerException.class, () -> Tensor.split(1, null, 3, 7));
        assertThrows(NullPointerException.class, () -> Tensor.split(1, input, 3, 7, (long[]) null));
    }

    @Test
    void splitWorksThroughTracing() {
        Tensor input = Tensor.iota(12, DataType.FP32).view(Shape.of(2, 6));

        Tensor traced = Tracer.trace(input, t -> Tensor.split(1, t, 2, 4)[1].add(1f));
        MemoryView<?> materialized = traced.materialize();

        assertEquals(Shape.of(2, 4), materialized.shape());
        assertEquals(3f, TensorTestReads.readFloat(traced, 0), 1e-4f);
    }
}
