package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.random.RandomKey;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class RandomTensorTest {

    @Test
    void manualSeedMakesRandDeterministic() {
        Tensor.manualSeed(1234L);
        Tensor first = Tensor.rand(Shape.of(16), DataType.FP32);
        Tensor second = Tensor.rand(Shape.of(16), DataType.FP32);

        Tensor.manualSeed(1234L);
        Tensor firstAgain = Tensor.rand(Shape.of(16), DataType.FP32);
        Tensor secondAgain = Tensor.rand(Shape.of(16), DataType.FP32);

        assertArrayEquals(toFloatArray(first), toFloatArray(firstAgain));
        assertArrayEquals(toFloatArray(second), toFloatArray(secondAgain));
    }

    @Test
    void explicitKeySplitIsDeterministic() {
        RandomKey key = RandomKey.of(9876L);

        Tensor a = Tensor.rand(Shape.of(8), DataType.FP32, key.split(0L));
        Tensor b = Tensor.rand(Shape.of(8), DataType.FP32, key.split(1L));

        Tensor aAgain = Tensor.rand(Shape.of(8), DataType.FP32, key.split(0L));
        Tensor bAgain = Tensor.rand(Shape.of(8), DataType.FP32, key.split(1L));

        assertArrayEquals(toFloatArray(a), toFloatArray(aAgain));
        assertArrayEquals(toFloatArray(b), toFloatArray(bAgain));
    }

    @Test
    void randAndRandnRejectNonFloatingOutputTypes() {
        RandomKey key = RandomKey.of(1L);
        assertThrows(
                IllegalArgumentException.class, () -> Tensor.rand(Shape.of(4), DataType.I32, key));
        assertThrows(
                IllegalArgumentException.class, () -> Tensor.randn(Shape.of(4), DataType.I64, key));
    }

    @Test
    void tracedRandUsesDeterministicRandomNode() {
        Tensor.manualSeed(999L);
        Tensor a = Tracer.trace(Tensor.rand(Shape.of(8), DataType.FP32), t -> t.add(1.0f));

        Tensor.manualSeed(999L);
        Tensor b = Tracer.trace(Tensor.rand(Shape.of(8), DataType.FP32), t -> t.add(1.0f));

        assertArrayEquals(toFloatArray(a), toFloatArray(b));
    }

    private static float[] toFloatArray(Tensor tensor) {
        float[] out = new float[Math.toIntExact(tensor.shape().size())];
        for (int i = 0; i < out.length; i++) {
            out[i] = TensorTestReads.readFloat(tensor, i);
        }
        return out;
    }
}
