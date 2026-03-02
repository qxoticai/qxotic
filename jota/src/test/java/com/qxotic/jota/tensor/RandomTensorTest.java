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
    void explicitKeyStreamsAreDeterministic() {
        RandomKey key = Tensor.randomKey(1234L);
        Tensor first = Tensor.rand(key.split(0L), Shape.of(16), DataType.FP32);
        Tensor second = Tensor.rand(key.split(1L), Shape.of(16), DataType.FP32);

        RandomKey keyAgain = Tensor.randomKey(1234L);
        Tensor firstAgain = Tensor.rand(keyAgain.split(0L), Shape.of(16), DataType.FP32);
        Tensor secondAgain = Tensor.rand(keyAgain.split(1L), Shape.of(16), DataType.FP32);

        assertArrayEquals(toFloatArray(first), toFloatArray(firstAgain));
        assertArrayEquals(toFloatArray(second), toFloatArray(secondAgain));
    }

    @Test
    void explicitKeySplitIsDeterministic() {
        RandomKey key = RandomKey.of(9876L);

        Tensor a = Tensor.rand(key.split(0L), Shape.of(8), DataType.FP32);
        Tensor b = Tensor.rand(key.split(1L), Shape.of(8), DataType.FP32);

        Tensor aAgain = Tensor.rand(key.split(0L), Shape.of(8), DataType.FP32);
        Tensor bAgain = Tensor.rand(key.split(1L), Shape.of(8), DataType.FP32);

        assertArrayEquals(toFloatArray(a), toFloatArray(aAgain));
        assertArrayEquals(toFloatArray(b), toFloatArray(bAgain));
    }

    @Test
    void randAndRandnRejectNonFloatingOutputTypes() {
        RandomKey key = RandomKey.of(1L);
        assertThrows(
                IllegalArgumentException.class, () -> Tensor.rand(key, Shape.of(4), DataType.I32));
        assertThrows(
                IllegalArgumentException.class, () -> Tensor.randn(key, Shape.of(4), DataType.I64));
    }

    @Test
    void tracedRandUsesDeterministicRandomNode() {
        RandomKey key = Tensor.randomKey(999L);
        Tensor a = Tracer.trace(Tensor.rand(key, Shape.of(8), DataType.FP32), t -> t.add(1.0f));
        Tensor b = Tracer.trace(Tensor.rand(key, Shape.of(8), DataType.FP32), t -> t.add(1.0f));

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
