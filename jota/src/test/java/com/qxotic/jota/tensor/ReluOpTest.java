package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.AbstractMemoryTest;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

@Disabled
@RunOnAllAvailableBackends
class ReluOpTest extends AbstractMemoryTest {

    @Test
    void reluWithFloatInput() {
        Tensor input = Tensor.of(new float[] {-2.0f, -1.0f, 0.0f, 1.0f});
        Tensor result = input.relu();
        MemoryView<?> output = result.materialize();

        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.of(4), output.shape());
        assertEquals(0.0f, readFloat(result, 0), 0.0001f);
        assertEquals(0.0f, readFloat(result, 1), 0.0001f);
        assertEquals(0.0f, readFloat(result, 2), 0.0001f);
        assertEquals(1.0f, readFloat(result, 3), 0.0001f);
    }

    @Test
    void reluWithNegativeValues() {
        Tensor input = Tensor.of(new float[] {-10.0f, -5.0f, -2.0f, -1.0f});
        Tensor result = input.relu();
        MemoryView<?> output = result.materialize();

        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.of(4), output.shape());
        assertEquals(0.0f, readFloat(result, 0), 0.0001f);
        assertEquals(0.0f, readFloat(result, 1), 0.0001f);
        assertEquals(0.0f, readFloat(result, 2), 0.0001f);
        assertEquals(0.0f, readFloat(result, 3), 0.0001f);
    }

    @Test
    void reluWithScalar() {
        Tensor input = Tensor.of(new float[] {-5.0f});
        Tensor result = input.relu();
        MemoryView<?> output = result.materialize();

        assertEquals(Shape.of(1), output.shape());
        assertEquals(0.0f, readFloat(result, 0), 0.0001f);
    }

    @Test
    void reluWithLargeValues() {
        Tensor input = Tensor.of(new float[] {100.0f, 200.0f, 300.0f});
        Tensor result = input.relu();
        MemoryView<?> output = result.materialize();

        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.of(3), output.shape());
        assertEquals(100.0f, readFloat(result, 0), 0.0001f);
        assertEquals(200.0f, readFloat(result, 1), 0.0001f);
        assertEquals(300.0f, readFloat(result, 2), 0.0001f);
    }

    private static float readFloat(Tensor tensor, long linearIndex) {
        return TensorTestReads.readFloat(tensor, linearIndex);
    }
}
