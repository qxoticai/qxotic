package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class ConstantFoldingBitwiseTest {

    @Test
    void foldsBitwiseNot() {
        Tensor result = Tensor.scalar(0x0FL).bitwiseNot();
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(~0x0FL, getConstant(result).value().longValue());
    }

    @Test
    void foldsBitwiseNotAllOnes() {
        Tensor result = Tensor.scalar(-1L).bitwiseNot();
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(0L, getConstant(result).value().longValue());
    }

    @Test
    void foldsBitwiseAnd() {
        Tensor result = Tensor.scalar(0xFFL).bitwiseAnd(Tensor.scalar(0x0FL));
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(0x0FL, getConstant(result).value().longValue());
    }

    @Test
    void foldsBitwiseAndWithZero() {
        Tensor result = Tensor.scalar(0x12345678L).bitwiseAnd(Tensor.scalar(0L));
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(0L, getConstant(result).value().longValue());
    }

    @Test
    void foldsBitwiseAndWithAllOnes() {
        Tensor result = Tensor.scalar(0xABCDL).bitwiseAnd(Tensor.scalar(-1L));
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(0xABCDL, getConstant(result).value().longValue());
    }

    @Test
    void foldsBitwiseOr() {
        Tensor result = Tensor.scalar(0xF0L).bitwiseOr(Tensor.scalar(0x0FL));
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(0xFFL, getConstant(result).value().longValue());
    }

    @Test
    void foldsBitwiseOrWithZero() {
        Tensor result = Tensor.scalar(0x12345678L).bitwiseOr(Tensor.scalar(0L));
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(0x12345678L, getConstant(result).value().longValue());
    }

    @Test
    void foldsBitwiseOrWithAllOnes() {
        Tensor result = Tensor.scalar(0xABCDL).bitwiseOr(Tensor.scalar(-1L));
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(-1L, getConstant(result).value().longValue());
    }

    @Test
    void foldsBitwiseXor() {
        Tensor result = Tensor.scalar(0xFFL).bitwiseXor(Tensor.scalar(0x0FL));
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(0xF0L, getConstant(result).value().longValue());
    }

    @Test
    void foldsBitwiseXorSameValue() {
        Tensor result = Tensor.scalar(0x12345678L).bitwiseXor(Tensor.scalar(0x12345678L));
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(0L, getConstant(result).value().longValue());
    }

    @Test
    void foldsBitwiseXorWithZero() {
        Tensor result = Tensor.scalar(0xABCDL).bitwiseXor(Tensor.scalar(0L));
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(0xABCDL, getConstant(result).value().longValue());
    }

    @Test
    void foldsBitwiseXorWithAllOnes() {
        Tensor result = Tensor.scalar(0x0FL).bitwiseXor(Tensor.scalar(-1L));
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(~0x0FL, getConstant(result).value().longValue());
    }

    @Test
    void foldsBitwiseI32() {
        Tensor result =
                Tensor.scalar(0xFFL, DataType.I32).bitwiseAnd(Tensor.scalar(0x0FL, DataType.I32));
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(DataType.I32, result.dataType());
        assertEquals(0x0FL, getConstant(result).value().longValue());
    }

    @Test
    void foldsBitwiseChained() {
        Tensor result =
                Tensor.scalar(0xFFL)
                        .bitwiseAnd(Tensor.scalar(0xF0L))
                        .bitwiseOr(Tensor.scalar(0x0FL));
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(0xFFL, getConstant(result).value().longValue());
    }

    @Test
    void foldsBitwiseNotTwice() {
        Tensor result = Tensor.scalar(0x12345678L).bitwiseNot().bitwiseNot();
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(0x12345678L, getConstant(result).value().longValue());
    }

    @Test
    void foldsDeMorgansLaw() {
        long a = 0xF0L;
        long b = 0x0FL;
        Tensor leftSide = Tensor.scalar(a).bitwiseAnd(Tensor.scalar(b)).bitwiseNot();
        Tensor rightSide = Tensor.scalar(a).bitwiseNot().bitwiseOr(Tensor.scalar(b).bitwiseNot());
        assertEquals(
                getConstant(leftSide).value().longValue(),
                getConstant(rightSide).value().longValue());
    }

    @Test
    void preservesDataTypeInBitwiseOps() {
        Tensor result =
                Tensor.scalar(0xFFL, DataType.I16).bitwiseOr(Tensor.scalar(0x100L, DataType.I16));
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(DataType.I16, result.dataType());
    }

    private static ConstantComputation getConstant(Tensor tensor) {
        return TensorTestInternals.computation(tensor)
                .filter(ConstantComputation.class::isInstance)
                .map(ConstantComputation.class::cast)
                .orElseThrow(() -> new AssertionError("Expected ConstantComputation"));
    }
}
