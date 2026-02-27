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
        Tensor a = Tensor.uniform(256, 0.0, 1.0, DataType.FP32, key);
        Tensor b = Tensor.rand(256, DataType.FP32, key);
        assertArrayEquals(toFloatArray(a), toFloatArray(b));
    }

    @Test
    void uniformMatchesRandAliasForSameKeyFp64() {
        RandomKey key = RandomKey.of(111L);
        Tensor a = Tensor.uniform(256, 0.0, 1.0, DataType.FP64, key);
        Tensor b = Tensor.rand(256, DataType.FP64, key);
        assertArrayEquals(toDoubleArray(a), toDoubleArray(b));
    }

    @Test
    void normalMatchesRandnAliasForSameKey() {
        RandomKey key = RandomKey.of(202L);
        Tensor a = Tensor.normal(256, 0.0, 1.0, DataType.FP64, key);
        Tensor b = Tensor.randn(256, DataType.FP64, key);
        assertArrayEquals(toDoubleArray(a), toDoubleArray(b));
    }

    @Test
    void normalMatchesRandnAliasForSameKeyFp32() {
        RandomKey key = RandomKey.of(212L);
        Tensor a = Tensor.normal(256, 0.0, 1.0, DataType.FP32, key);
        Tensor b = Tensor.randn(256, DataType.FP32, key);
        assertArrayEquals(toFloatArray(a), toFloatArray(b));
    }

    @Test
    void uniformIntMatchesRandIntAliasForSameKey() {
        RandomKey key = RandomKey.of(303L);
        Tensor a = Tensor.uniformInt(-7, 19, 512, DataType.I32, key);
        Tensor b = Tensor.randInt(-7, 19, 512, DataType.I32, key);
        assertArrayEquals(toLongArray(a.cast(DataType.I64)), toLongArray(b.cast(DataType.I64)));
    }

    @Test
    void shapeOverloadsMatchSizeOverloads() {
        RandomKey key = RandomKey.of(404L);

        Tensor uSize = Tensor.uniform(12, -2.0, 5.0, DataType.FP64, key);
        Tensor uShape = Tensor.uniform(Shape.of(3, 4), -2.0, 5.0, DataType.FP64, key);
        assertArrayEquals(toDoubleArray(uSize), toDoubleArray(uShape));

        Tensor nSize = Tensor.normalInt(12, 3.0, 1.5, DataType.I16, key);
        Tensor nShape = Tensor.normalInt(Shape.of(3, 4), 3.0, 1.5, DataType.I16, key);
        assertArrayEquals(
                toLongArray(nSize.cast(DataType.I64)), toLongArray(nShape.cast(DataType.I64)));

        Tensor uiSize = Tensor.uniformInt(-3, 9, 12, DataType.I32, key);
        Tensor uiShape = Tensor.uniformInt(-3, 9, Shape.of(3, 4), DataType.I32, key);
        assertArrayEquals(
                toLongArray(uiSize.cast(DataType.I64)), toLongArray(uiShape.cast(DataType.I64)));

        Tensor nFloatSize = Tensor.normal(12, 2.0, 0.25, DataType.FP64, key);
        Tensor nFloatShape = Tensor.normal(Shape.of(3, 4), 2.0, 0.25, DataType.FP64, key);
        assertArrayEquals(toDoubleArray(nFloatSize), toDoubleArray(nFloatShape));
    }

    @Test
    void uniformAndRandConsumeThreadLocalKeyConsistently() {
        Tensor.manualSeed(9001L);
        Tensor uniformFirst = Tensor.uniform(64, 0.0, 1.0, DataType.FP32);

        Tensor.manualSeed(9001L);
        Tensor randFirst = Tensor.rand(64, DataType.FP32);

        assertArrayEquals(toFloatArray(uniformFirst), toFloatArray(randFirst));
    }

    @Test
    void normalAndRandnConsumeThreadLocalKeyConsistently() {
        Tensor.manualSeed(7777L);
        Tensor normalFirst = Tensor.normal(64, 0.0, 1.0, DataType.FP64);

        Tensor.manualSeed(7777L);
        Tensor randnFirst = Tensor.randn(64, DataType.FP64);

        assertArrayEquals(toDoubleArray(normalFirst), toDoubleArray(randnFirst));
    }

    @Test
    void uniformIntRespectsEndExclusive() {
        Tensor t = Tensor.uniformInt(0, 4, 4096, DataType.I32, RandomKey.of(505L));
        long[] values = toLongArray(t.cast(DataType.I64));
        for (long v : values) {
            assertTrue(v >= 0);
            assertTrue(v < 4);
        }
    }

    @Test
    void normalIntClampsToDtypeRange() {
        Tensor t = Tensor.normalInt(2048, 10_000.0, 1_000.0, DataType.I8, RandomKey.of(606L));
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
                () -> Tensor.uniform(8, 0.0, 1.0, DataType.I32, key));
        assertThrows(IllegalArgumentException.class, () -> Tensor.uniform(2, 0f, 1f, DataType.I8));
        assertThrows(IllegalArgumentException.class, () -> Tensor.normal(2, 0f, 1f, DataType.I8));
        assertThrows(IllegalArgumentException.class, () -> Tensor.rand(2, DataType.I8));
        assertThrows(IllegalArgumentException.class, () -> Tensor.randn(2, DataType.I8));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.normal(8, 0.0, 1.0, DataType.I64, key));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.uniformInt(0, 10, 8, DataType.FP32, key));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.normalInt(8, 0.0, 1.0, DataType.BOOL, key));
    }

    @Test
    void distributionApisValidateRangeAndStdParameters() {
        RandomKey key = RandomKey.of(808L);
        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.uniform(8, 1.0, 1.0, DataType.FP32, key));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.uniform(8, 2.0, -1.0, DataType.FP64, key));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.uniform(8, Double.NaN, 1.0, DataType.FP32, key));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.uniform(8, 0.0, Double.POSITIVE_INFINITY, DataType.FP64, key));

        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.normal(8, 0.0, 0.0, DataType.FP32, key));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.normal(8, 0.0, -1.0, DataType.FP64, key));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.normal(8, Double.NaN, 1.0, DataType.FP32, key));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tensor.normalInt(8, 0.0, Double.POSITIVE_INFINITY, DataType.I32, key));
    }

    @Test
    void normalAndNormalIntUseConfiguredDtypes() {
        assertEquals(DataType.FP32, Tensor.normal(8, 0.0, 1.0, DataType.FP32).dataType());
        assertEquals(DataType.I16, Tensor.normalInt(8, 0.0, 1.0, DataType.I16).dataType());
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
