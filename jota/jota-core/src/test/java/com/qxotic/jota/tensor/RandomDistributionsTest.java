package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.random.RandomKey;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import com.qxotic.jota.testutil.TensorTestReads;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class RandomDistributionsTest {

    @Test
    void uniformMatchesRandAliasForSameKey() {
        RandomKey key = RandomKey.of(101L);
        Tensor a = Tensor.uniform(key, 256, 0.0, 1.0, DataType.FP32);
        Tensor b = Tensor.rand(key, 256, DataType.FP32);
        assertArrayEquals(toFloatArray(a), toFloatArray(b));
    }

    @Test
    void uniformMatchesRandAliasForSameKeyFp64() {
        RandomKey key = RandomKey.of(111L);
        Tensor a = Tensor.uniform(key, 256, 0.0, 1.0, DataType.FP64);
        Tensor b = Tensor.rand(key, 256, DataType.FP64);
        assertArrayEquals(toDoubleArray(a), toDoubleArray(b));
    }

    @Test
    void normalMatchesRandnAliasForSameKey() {
        RandomKey key = RandomKey.of(202L);
        Tensor a = Tensor.normal(key, 256, 0.0, 1.0, DataType.FP64);
        Tensor b = Tensor.randn(key, 256, DataType.FP64);
        assertArrayEquals(toDoubleArray(a), toDoubleArray(b));
    }

    @Test
    void normalMatchesRandnAliasForSameKeyFp32() {
        RandomKey key = RandomKey.of(212L);
        Tensor a = Tensor.normal(key, 256, 0.0, 1.0, DataType.FP32);
        Tensor b = Tensor.randn(key, 256, DataType.FP32);
        assertArrayEquals(toFloatArray(a), toFloatArray(b));
    }

    @Test
    void uniformIntMatchesRandIntAliasForSameKey() {
        RandomKey key = RandomKey.of(303L);
        Tensor a = Tensor.uniformInt(key, -7, 19, 512, DataType.I32);
        Tensor b = Tensor.randInt(key, -7, 19, 512, DataType.I32);
        assertArrayEquals(toLongArray(a.cast(DataType.I64)), toLongArray(b.cast(DataType.I64)));
    }

    @Test
    void shapeOverloadsMatchSizeOverloads() {
        RandomKey key = RandomKey.of(404L);

        Tensor uSize = Tensor.uniform(key, 12, -2.0, 5.0, DataType.FP64);
        Tensor uShape = Tensor.uniform(key, Shape.of(3, 4), -2.0, 5.0, DataType.FP64);
        assertArrayEquals(toDoubleArray(uSize), toDoubleArray(uShape));

        Tensor nSize = Tensor.normalInt(key, 12, 3.0, 1.5, DataType.I16);
        Tensor nShape = Tensor.normalInt(key, Shape.of(3, 4), 3.0, 1.5, DataType.I16);
        assertArrayEquals(
                toLongArray(nSize.cast(DataType.I64)), toLongArray(nShape.cast(DataType.I64)));

        Tensor uiSize = Tensor.uniformInt(key, -3, 9, 12, DataType.I32);
        Tensor uiShape = Tensor.uniformInt(key, -3, 9, Shape.of(3, 4), DataType.I32);
        assertArrayEquals(
                toLongArray(uiSize.cast(DataType.I64)), toLongArray(uiShape.cast(DataType.I64)));

        Tensor nFloatSize = Tensor.normal(key, 12, 2.0, 0.25, DataType.FP64);
        Tensor nFloatShape = Tensor.normal(key, Shape.of(3, 4), 2.0, 0.25, DataType.FP64);
        assertArrayEquals(toDoubleArray(nFloatSize), toDoubleArray(nFloatShape));
    }

    @Test
    void uniformIntRespectsEndExclusive() {
        Tensor t = Tensor.uniformInt(RandomKey.of(505L), 0, 4, 4096, DataType.I32);
        long[] values = toLongArray(t.cast(DataType.I64));
        for (long v : values) {
            assertTrue(v >= 0);
            assertTrue(v < 4);
        }
    }

    @Test
    void normalIntClampsToDtypeRange() {
        Tensor t = Tensor.normalInt(RandomKey.of(606L), 2048, 10_000.0, 1_000.0, DataType.I8);
        long[] values = toLongArray(t.cast(DataType.I64));
        for (long v : values) {
            assertTrue(v >= Byte.MIN_VALUE);
            assertTrue(v <= Byte.MAX_VALUE);
        }
    }

    @Test
    void distributionApisValidateDtypeFamilies() {
        RandomKey key = RandomKey.of(707L);
        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.uniform(key, 8, 0.0, 1.0, DataType.I32));
        assertThrows(
                IllegalArgumentException.class, () -> Tensor.uniform(key, 2, 0f, 1f, DataType.I8));
        assertThrows(
                IllegalArgumentException.class, () -> Tensor.normal(key, 2, 0f, 1f, DataType.I8));
        assertThrows(IllegalArgumentException.class, () -> Tensor.rand(key, 2, DataType.I8));
        assertThrows(IllegalArgumentException.class, () -> Tensor.randn(key, 2, DataType.I8));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.normal(key, 8, 0.0, 1.0, DataType.I64));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.uniformInt(key, 0, 10, 8, DataType.FP32));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.normalInt(key, 8, 0.0, 1.0, DataType.BOOL));
    }

    @Test
    void distributionApisValidateRangeAndStdParameters() {
        RandomKey key = RandomKey.of(808L);
        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.uniform(key, 8, 1.0, 1.0, DataType.FP32));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.uniform(key, 8, 2.0, -1.0, DataType.FP64));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.uniform(key, 8, Double.NaN, 1.0, DataType.FP32));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.uniform(key, 8, 0.0, Double.POSITIVE_INFINITY, DataType.FP64));

        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.normal(key, 8, 0.0, 0.0, DataType.FP32));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.normal(key, 8, 0.0, -1.0, DataType.FP64));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.normal(key, 8, Double.NaN, 1.0, DataType.FP32));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.normalInt(key, 8, 0.0, Double.POSITIVE_INFINITY, DataType.I32));
    }

    @Test
    void normalAndNormalIntUseConfiguredDtypes() {
        RandomKey key = RandomKey.of(909L);
        assertEquals(DataType.FP32, Tensor.normal(key, 8, 0.0, 1.0, DataType.FP32).dataType());
        assertEquals(DataType.I16, Tensor.normalInt(key, 8, 0.0, 1.0, DataType.I16).dataType());
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

    private static long[] toLongArray(Tensor tensor) {
        long[] out = new long[Math.toIntExact(tensor.shape().size())];
        for (int i = 0; i < out.length; i++) {
            out[i] = TensorTestReads.readLong(tensor, i);
        }
        return out;
    }
}
