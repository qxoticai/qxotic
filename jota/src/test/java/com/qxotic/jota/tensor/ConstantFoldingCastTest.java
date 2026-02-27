package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class ConstantFoldingCastTest {

    @Test
    void foldsCastFloatToDouble() {
        Tensor result = Tensor.scalar(3.14f).cast(DataType.FP64);
        assertTrue(result.isLazy());
        assertEquals(DataType.FP64, result.dataType());
        assertEquals(3.14, getConstant(result).value().doubleValue(), 0.001);
    }

    @Test
    void foldsCastDoubleToFloat() {
        Tensor result = Tensor.scalar(2.718).cast(DataType.FP32);
        assertTrue(result.isLazy());
        assertEquals(DataType.FP32, result.dataType());
        assertEquals(2.718f, getConstant(result).value().floatValue(), 0.001f);
    }

    @Test
    void foldsCastIntToFloat() {
        Tensor result = Tensor.scalar(42L, DataType.I32).cast(DataType.FP32);
        assertTrue(result.isLazy());
        assertEquals(DataType.FP32, result.dataType());
        assertEquals(42.0, getConstant(result).value().doubleValue(), 0.0001);
    }

    @Test
    void foldsCastFloatToInt() {
        Tensor result = Tensor.scalar(7.9f).cast(DataType.I32);
        assertTrue(result.isLazy());
        assertEquals(DataType.I32, result.dataType());
        assertEquals(7L, getConstant(result).value().longValue());
    }

    @Test
    void foldsCastI32ToI64() {
        Tensor result = Tensor.scalar(100L, DataType.I32).cast(DataType.I64);
        assertTrue(result.isLazy());
        assertEquals(DataType.I64, result.dataType());
        assertEquals(100L, getConstant(result).value().longValue());
    }

    @Test
    void castSameTypeReturnsThis() {
        Tensor original = Tensor.scalar(5.0f);
        Tensor result = original.cast(DataType.FP32);
        assertSame(original, result);
    }

    @Test
    void foldsCastChain() {
        Tensor result =
                Tensor.scalar(3.7f).cast(DataType.FP64).cast(DataType.I64).cast(DataType.I32);
        assertTrue(result.isLazy());
        assertEquals(DataType.I32, result.dataType());
        assertEquals(3L, getConstant(result).value().longValue());
    }

    private static ConstantComputation getConstant(Tensor tensor) {
        return tensor.computation()
                .filter(ConstantComputation.class::isInstance)
                .map(ConstantComputation.class::cast)
                .orElseThrow(() -> new AssertionError("Expected ConstantComputation"));
    }
}
