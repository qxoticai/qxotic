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
class SiluOpTest extends AbstractMemoryTest {

    @Test
    void siluWithFloatInput() {
        Tensor input = Tensor.of(new float[] {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
        Tensor result = input.silu();
        MemoryView<?> output = result.materialize();

        float delta = 0.0001f;
        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.of(5), output.shape());
        assertEquals(-0.2689f, readFloat(result, 0), delta);
        assertEquals(-0.2689f, readFloat(result, 1), delta);
        assertEquals(-0.1192f, readFloat(result, 2), delta);
        assertEquals(0.0000f, readFloat(result, 3), delta);
        assertEquals(0.4621f, readFloat(result, 4), delta);
    }

    @Test
    void siluWithNegativeValues() {
        Tensor input = Tensor.of(new float[] {-10.0f, -5.0f, -2.0f, -1.0f});
        Tensor result = input.silu();
        result.materialize();

        float delta = 0.0001f;
        assertEquals(-0.0000500f, readFloat(result, 0), delta);
        assertEquals(-0.0067f, readFloat(result, 1), delta);
        assertEquals(-0.1192f, readFloat(result, 2), delta);
        assertEquals(-0.2689f, readFloat(result, 3), delta);
        assertEquals(-0.4621f, readFloat(result, 4), 0.0001f);
    }

    @Test
    void siluWithScalar() {
        Tensor input = Tensor.of(new float[] {-5.0f});
        Tensor result = input.silu();
        MemoryView<?> output = result.materialize();

        assertEquals(Shape.of(1), output.shape());
        assertEquals(-0.00669f, readFloat(result, 0), 0.0001f);
    }

    @Test
    void siluWithLargeValues() {
        Tensor input = Tensor.of(new float[] {10.0f, 50.0f, 100.0f});
        Tensor result = input.silu();
        MemoryView<?> output = result.materialize();

        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.of(3), output.shape());
        assertEquals(0.0f, readFloat(result, 0), 0.001f);
        assertEquals(1.0f, readFloat(result, 1), 0.001f);
        assertEquals(0.9999085f, readFloat(result, 2), 0.001f);
    }

    private static float readFloat(Tensor tensor, long linearIndex) {
        return TensorTestReads.readFloat(tensor, linearIndex);
    }
}
