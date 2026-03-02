package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryHelpers;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class ApiExampleTest {

    @Test
    void basicFunctionRunsAndMaterializes() {
        MemoryView<?> view =
                MemoryHelpers.arange(DomainFactory.ofMemorySegment(), DataType.FP32, 6)
                        .view(Shape.of(2, 3));
        Tensor x = Tensor.of(view);
        Tensor y = x.add(x).sqrt();

        assertTrue(TensorTestInternals.isLazy(y));
        assertFalse(TensorTestInternals.isMaterialized(y));

        MemoryView<?> output = y.materialize();

        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.of(2, 3), output.shape());
        assertEquals(0.0f, TensorTestReads.readFloat(y, 0), 0.0001f);
        assertEquals((float) Math.sqrt(10.0), TensorTestReads.readFloat(y, 5), 0.0001f);
    }

    @Test
    void clipClipsValuesToRange() {
        Tensor x = Tensor.of(new float[] {-2.0f, 0.0f, 3.0f, 7.0f, 10.0f});
        Tensor cliped = x.clip(0.0, 5.0);

        assertTrue(TensorTestInternals.isLazy(cliped));

        MemoryView<?> output = cliped.materialize();
        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.of(5), output.shape());

        // Values below min are cliped to min
        assertEquals(0.0f, TensorTestReads.readFloat(cliped, 0), 0.0001f);
        // Values within range are unchanged
        assertEquals(0.0f, TensorTestReads.readFloat(cliped, 1), 0.0001f);
        assertEquals(3.0f, TensorTestReads.readFloat(cliped, 2), 0.0001f);
        // Values above max are cliped to max
        assertEquals(5.0f, TensorTestReads.readFloat(cliped, 3), 0.0001f);
        assertEquals(5.0f, TensorTestReads.readFloat(cliped, 4), 0.0001f);
    }

    @Test
    void clipWorksWithTensorBounds() {
        Tensor x = Tensor.of(new float[] {1.0f, 5.0f, 10.0f});
        Tensor min = Tensor.scalar(2.0f);
        Tensor max = Tensor.scalar(8.0f);
        Tensor cliped = x.clip(min, max);

        MemoryView<?> output = cliped.materialize();
        assertEquals(2.0f, TensorTestReads.readFloat(cliped, 0), 0.0001f);
        assertEquals(5.0f, TensorTestReads.readFloat(cliped, 1), 0.0001f);
        assertEquals(8.0f, TensorTestReads.readFloat(cliped, 2), 0.0001f);
    }

    @Test
    void clipWorksWithDifferentDtypes() {
        Tensor x = Tensor.of(new double[] {-1.0, 0.5, 2.0});
        Tensor cliped = x.clip(0.0, 1.0);

        MemoryView<?> output = cliped.materialize();
        assertEquals(DataType.FP64, output.dataType());
        assertEquals(
                0.0,
                ((Number) TensorTestReads.readValue(cliped, 0, DataType.FP64)).doubleValue(),
                0.0001);
        assertEquals(
                0.5,
                ((Number) TensorTestReads.readValue(cliped, 1, DataType.FP64)).doubleValue(),
                0.0001);
        assertEquals(
                1.0,
                ((Number) TensorTestReads.readValue(cliped, 2, DataType.FP64)).doubleValue(),
                0.0001);
    }

    @Test
    void clipWithSameMinMaxReturnsConstant() {
        Tensor x = Tensor.of(new float[] {1.0f, 5.0f, 10.0f});
        Tensor cliped = x.clip(5.0, 5.0);

        MemoryView<?> output = cliped.materialize();
        assertEquals(5.0f, TensorTestReads.readFloat(cliped, 0), 0.0001f);
        assertEquals(5.0f, TensorTestReads.readFloat(cliped, 1), 0.0001f);
        assertEquals(5.0f, TensorTestReads.readFloat(cliped, 2), 0.0001f);
    }

    @Test
    void clipWorksWithIntegerTensors() {
        // Test with int tensor
        Tensor x = Tensor.of(new int[] {-10, 0, 50, 100});
        Tensor cliped = x.clip(0L, 50L);

        MemoryView<?> output = cliped.materialize();
        assertEquals(DataType.I32, output.dataType());
        assertEquals(0, TensorTestReads.readValue(cliped, 0, DataType.I32));
        assertEquals(0, TensorTestReads.readValue(cliped, 1, DataType.I32));
        assertEquals(50, TensorTestReads.readValue(cliped, 2, DataType.I32));
        assertEquals(50, TensorTestReads.readValue(cliped, 3, DataType.I32));

        // Test with long tensor
        Tensor y = Tensor.of(new long[] {-100L, 25L, 200L});
        Tensor cliped2 = y.clip(0L, 100L);

        MemoryView<?> output2 = cliped2.materialize();
        assertEquals(DataType.I64, output2.dataType());
        assertEquals(0L, TensorTestReads.readValue(cliped2, 0, DataType.I64));
        assertEquals(25L, TensorTestReads.readValue(cliped2, 1, DataType.I64));
        assertEquals(100L, TensorTestReads.readValue(cliped2, 2, DataType.I64));
    }
}
