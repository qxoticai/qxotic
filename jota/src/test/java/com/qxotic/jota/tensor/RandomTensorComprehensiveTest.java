package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.random.RandomKey;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class RandomTensorComprehensiveTest {

    @Test
    void sizeAndShapeRandomApisMatch() {
        RandomKey key = RandomKey.of(7L);
        Tensor fromSize = Tensor.rand(6, DataType.FP32, key);
        Tensor fromShape = Tensor.rand(Shape.of(2, 3), DataType.FP32, key);
        assertArrayEquals(toFloatArray(fromSize), toFloatArray(fromShape));
    }

    @Test
    void randFp32ValuesStayWithinUnitInterval() {
        Tensor t = Tensor.rand(Shape.of(2048), DataType.FP32, RandomKey.of(10L));
        float[] values = toFloatArray(t);

        for (float v : values) {
            assertTrue(v >= 0.0f);
            assertTrue(v < 1.0f);
            assertTrue(Float.isFinite(v));
        }
    }

    @Test
    void randFp64ValuesStayWithinUnitInterval() {
        Tensor t = Tensor.rand(Shape.of(2048), DataType.FP64, RandomKey.of(11L));
        double[] values = toDoubleArray(t);

        for (double v : values) {
            assertTrue(v >= 0.0);
            assertTrue(v < 1.0);
            assertTrue(Double.isFinite(v));
        }
    }

    @Test
    void randnWithManualSeedIsDeterministic() {
        Tensor.manualSeed(444L);
        Tensor first = Tensor.randn(Shape.of(64), DataType.FP32);

        Tensor.manualSeed(444L);
        Tensor second = Tensor.randn(Shape.of(64), DataType.FP32);

        assertArrayEquals(toFloatArray(first), toFloatArray(second));
    }

    @Test
    void randnWithExplicitKeyIsDeterministicForFp64() {
        RandomKey key = RandomKey.of(123L);
        Tensor first = Tensor.randn(Shape.of(64), DataType.FP64, key);
        Tensor second = Tensor.randn(Shape.of(64), DataType.FP64, key);

        assertArrayEquals(toDoubleArray(first), toDoubleArray(second));
    }

    @Test
    void randnProducesFiniteValuesWithBothSigns() {
        Tensor t = Tensor.randn(Shape.of(8192), DataType.FP64, RandomKey.of(555L));
        double[] values = toDoubleArray(t);

        boolean hasPositive = false;
        boolean hasNegative = false;
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < values.length; i++) {
            double v = values[i];
            assertTrue(Double.isFinite(v));
            hasPositive |= v > 0.0;
            hasNegative |= v < 0.0;
            min = Math.min(min, v);
            max = Math.max(max, v);
        }

        assertTrue(hasPositive);
        assertTrue(hasNegative);
        assertTrue(max - min > 1.0);
    }

    @Test
    void randomApisRejectNullArguments() {
        RandomKey key = RandomKey.of(1L);

        assertThrows(NullPointerException.class, () -> Tensor.rand(null, DataType.FP32, key));
        assertThrows(NullPointerException.class, () -> Tensor.rand(Shape.of(1), null, key));
        assertThrows(
                NullPointerException.class, () -> Tensor.rand(Shape.of(1), DataType.FP32, null));

        assertThrows(NullPointerException.class, () -> Tensor.randn(null, DataType.FP32, key));
        assertThrows(NullPointerException.class, () -> Tensor.randn(Shape.of(1), null, key));
        assertThrows(
                NullPointerException.class, () -> Tensor.randn(Shape.of(1), DataType.FP32, null));

        assertThrows(
                NullPointerException.class, () -> Tensor.randInt(0, 10, Shape.of(1), null, key));
        assertThrows(
                NullPointerException.class,
                () -> Tensor.randInt(0, 10, Shape.of(1), DataType.I32, null));
    }

    @Test
    void randomSupportsZeroSizedShapes() {
        Tensor u = Tensor.rand(Shape.of(0, 3), DataType.FP32, RandomKey.of(9L));
        Tensor n = Tensor.randn(Shape.of(0, 3), DataType.FP64, RandomKey.of(9L));

        assertEquals(0L, u.shape().size());
        assertEquals(0L, n.shape().size());
        assertEquals(0, toFloatArray(u).length);
        assertEquals(0, toDoubleArray(n).length);
    }

    @Test
    void tracedRandnIsDeterministicWithManualSeed() {
        Tensor.manualSeed(808L);
        Tensor first = Tracer.trace(Tensor.randn(Shape.of(32), DataType.FP32), t -> t.add(0.5f));

        Tensor.manualSeed(808L);
        Tensor second = Tracer.trace(Tensor.randn(Shape.of(32), DataType.FP32), t -> t.add(0.5f));

        assertArrayEquals(toFloatArray(first), toFloatArray(second));
    }

    @Test
    void randIntProducesValuesWithinBounds() {
        Tensor ints = Tensor.randInt(-5, 13, Shape.of(4096), DataType.I32, RandomKey.of(41L));
        float[] values = toFloatArray(ints.cast(DataType.FP32));
        for (float v : values) {
            assertTrue(v >= -5.0f);
            assertTrue(v < 13.0f);
        }
    }

    @Test
    void randIntIsDeterministicForSameKey() {
        RandomKey key = RandomKey.of(123L);
        Tensor a = Tensor.randInt(10, 20, 256, DataType.I64, key);
        Tensor b = Tensor.randInt(10, 20, 256, DataType.I64, key);
        assertArrayEquals(
                toDoubleArray(a.cast(DataType.FP64)), toDoubleArray(b.cast(DataType.FP64)));
    }

    @Test
    void randIntValidatesDtypeAndBounds() {
        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.randInt(0, 10, 4, DataType.FP32, RandomKey.of(1L)));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.randInt(0, 10, 4, DataType.BOOL, RandomKey.of(1L)));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.randInt(5, 5, 4, DataType.I32, RandomKey.of(1L)));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.randInt(0, 1000, 4, DataType.I8, RandomKey.of(1L)));
    }

    @Test
    void uniformSupportsCustomFloatRange() {
        Tensor t = Tensor.uniform(2048, -3.5, 2.25, DataType.FP64, RandomKey.of(8L));
        double[] values = toDoubleArray(t);
        for (double v : values) {
            assertTrue(v >= -3.5);
            assertTrue(v < 2.25);
        }
    }

    @Test
    void normalIntIsDeterministicAndIntegralTyped() {
        RandomKey key = RandomKey.of(77L);
        Tensor a = Tensor.normalInt(512, 5.0, 2.0, DataType.I16, key);
        Tensor b = Tensor.normalInt(512, 5.0, 2.0, DataType.I16, key);
        assertEquals(DataType.I16, a.dataType());
        assertArrayEquals(toFloatArray(a.cast(DataType.FP32)), toFloatArray(b.cast(DataType.FP32)));
    }

    @Test
    void normalAndUniformRejectInvalidParameters() {
        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.uniform(4, 1.0, 1.0, DataType.FP32, RandomKey.of(1L)));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.normal(4, 0.0, 0.0, DataType.FP32, RandomKey.of(1L)));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.normalInt(4, 0.0, -1.0, DataType.I32, RandomKey.of(1L)));
    }

    @Test
    void manualSeedIsThreadLocal() throws ExecutionException, InterruptedException {
        Tensor.manualSeed(123L);
        float[] mainFirst = toFloatArray(Tensor.rand(Shape.of(16), DataType.FP32));

        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<float[]> workerFuture =
                    executor.submit(
                            () -> {
                                Tensor.manualSeed(999L);
                                return toFloatArray(Tensor.rand(Shape.of(16), DataType.FP32));
                            });

            float[] workerValues = workerFuture.get();

            float[] mainSecond = toFloatArray(Tensor.rand(Shape.of(16), DataType.FP32));

            Tensor.manualSeed(123L);
            float[] expectedMainFirst = toFloatArray(Tensor.rand(Shape.of(16), DataType.FP32));
            float[] expectedMainSecond = toFloatArray(Tensor.rand(Shape.of(16), DataType.FP32));

            assertArrayEquals(expectedMainFirst, mainFirst);
            assertArrayEquals(expectedMainSecond, mainSecond);
            assertFalse(arraysEqual(workerValues, mainFirst));
        } finally {
            executor.shutdownNow();
        }
    }

    private static boolean arraysEqual(float[] a, float[] b) {
        if (a.length != b.length) {
            return false;
        }
        for (int i = 0; i < a.length; i++) {
            if (Float.floatToIntBits(a[i]) != Float.floatToIntBits(b[i])) {
                return false;
            }
        }
        return true;
    }

    private static float[] toFloatArray(Tensor tensor) {
        float[] out = new float[Math.toIntExact(tensor.shape().size())];
        for (int i = 0; i < out.length; i++) {
            out[i] = TensorTestReads.readFloat(tensor, i);
        }
        return out;
    }

    private static double[] toDoubleArray(Tensor tensor) {
        double[] out = new double[Math.toIntExact(tensor.shape().size())];
        for (int i = 0; i < out.length; i++) {
            out[i] = ((Number) TensorTestReads.readValue(tensor, i, DataType.FP64)).doubleValue();
        }
        return out;
    }
}
