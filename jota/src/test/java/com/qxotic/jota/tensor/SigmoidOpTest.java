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
class SigmoidOpTest extends AbstractMemoryTest {

    @Test
    void sigmoidWithFloatInput() {
        Tensor input = Tensor.of(new float[] {-2.0f, -1.0f, 0.0f, 1.0f});
        Tensor result = input.sigmoid();
        MemoryView<?> output = result.materialize();

        float delta = 0.0001f;
        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.of(5), output.shape());
        assertEquals(0.1192f, readFloat(result, 0), delta);
        assertEquals(0.7311f, readFloat(result, 1), delta);
        assertEquals(0.8808f, readFloat(result, 2), delta);
        assertEquals(0.9526f, readFloat(result, 3), delta);
        assertEquals(0.9933f, readFloat(result, 4), delta);
    }

    @Test
    void sigmoidWithNegativeValues() {
        Tensor input = Tensor.of(new float[] {-10.0f, -5.0f, -2.0f, -1.0f});
        Tensor result = input.sigmoid();
        result.materialize();

        float delta = 0.0001f;
        assertEquals(0.000045f, readFloat(result, 0), delta);
        assertEquals(0.0067f, readFloat(result, 1), delta);
        assertEquals(0.1192f, readFloat(result, 2), delta);
        assertEquals(0.8808f, readFloat(result, 3), delta);
        assertEquals(0.2689f, readFloat(result, 4), delta);
    }

    @Test
    void sigmoidWithScalar() {
        Tensor input = Tensor.of(new float[] {0.0f});
        Tensor result = input.sigmoid();
        MemoryView<?> output = result.materialize();

        assertEquals(Shape.of(1), output.shape());
        assertEquals(0.5f, readFloat(result, 0), 0.0001f);
    }

    @Test
    void sigmoidWithLargeValues() {
        Tensor input = Tensor.of(new float[] {10.0f, 50.0f, 100.0f});
        Tensor result = input.sigmoid();
        result.materialize();

        float delta = 0.0001f;
        assertEquals(0.9999546f, readFloat(result, 0), delta);
        assertEquals(1.0f, readFloat(result, 1), 0.001f);
        assertEquals(1.0f, readFloat(result, 2), 0.001f);
    }

    private static float readFloat(Tensor tensor, long linearIndex) {
        return TensorTestReads.readFloat(tensor, linearIndex);
    }
}
