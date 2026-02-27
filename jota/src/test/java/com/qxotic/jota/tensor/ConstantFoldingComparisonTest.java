package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import org.junit.jupiter.api.Test;

@RunOnAllAvailableBackends
class ConstantFoldingComparisonTest {

    @Test
    void foldsEqualTrue() {
        Tensor result = Tensor.scalar(5.0f).equal(Tensor.scalar(5.0f));
        assertTrue(result.isLazy());
        assertEquals(DataType.BOOL, result.dataType());
        assertEquals(1L, getConstant(result).value().longValue());
    }

    @Test
    void foldsEqualFalse() {
        Tensor result = Tensor.scalar(5.0f).equal(Tensor.scalar(3.0f));
        assertTrue(result.isLazy());
        assertEquals(DataType.BOOL, result.dataType());
        assertEquals(0L, getConstant(result).value().longValue());
    }

    @Test
    void foldsLessThanTrue() {
        Tensor result = Tensor.scalar(3.0f).lessThan(Tensor.scalar(5.0f));
        assertTrue(result.isLazy());
        assertEquals(DataType.BOOL, result.dataType());
        assertEquals(1L, getConstant(result).value().longValue());
    }

    @Test
    void foldsLessThanFalse() {
        Tensor result = Tensor.scalar(7.0f).lessThan(Tensor.scalar(5.0f));
        assertTrue(result.isLazy());
        assertEquals(DataType.BOOL, result.dataType());
        assertEquals(0L, getConstant(result).value().longValue());
    }

    @Test
    void foldsNotEqual() {
        Tensor result = Tensor.scalar(5.0f).notEqual(Tensor.scalar(3.0f));
        assertTrue(result.isLazy());
        assertEquals(DataType.BOOL, result.dataType());
        assertEquals(1L, getConstant(result).value().longValue());
    }

    @Test
    void foldsGreaterThan() {
        Tensor result = Tensor.scalar(7.0f).greaterThan(Tensor.scalar(3.0f));
        assertTrue(result.isLazy());
        assertEquals(DataType.BOOL, result.dataType());
        assertEquals(1L, getConstant(result).value().longValue());
    }

    @Test
    void foldsLessThanOrEqual() {
        Tensor result = Tensor.scalar(3.0f).lessThanOrEqual(Tensor.scalar(5.0f));
        assertTrue(result.isLazy());
        assertEquals(DataType.BOOL, result.dataType());
        assertEquals(1L, getConstant(result).value().longValue());
    }

    @Test
    void foldsGreaterThanOrEqual() {
        Tensor result = Tensor.scalar(5.0f).greaterThanOrEqual(Tensor.scalar(5.0f));
        assertTrue(result.isLazy());
        assertEquals(DataType.BOOL, result.dataType());
        assertEquals(1L, getConstant(result).value().longValue());
    }

    @Test
    void foldsIntegerComparison() {
        Tensor result = Tensor.scalar(10L).lessThan(Tensor.scalar(20L));
        assertTrue(result.isLazy());
        assertEquals(DataType.BOOL, result.dataType());
        assertEquals(1L, getConstant(result).value().longValue());
    }

    @Test
    void foldsLogicalNot() {
        Tensor boolTrue = Tensor.scalar(5.0f).equal(Tensor.scalar(5.0f));
        Tensor result = boolTrue.logicalNot();
        assertTrue(result.isLazy());
        assertEquals(DataType.BOOL, result.dataType());
        assertEquals(0L, getConstant(result).value().longValue());
    }

    @Test
    void foldsComparisonChain() {
        Tensor aLtB = Tensor.scalar(1.0f).lessThan(Tensor.scalar(2.0f));
        Tensor bLtC = Tensor.scalar(2.0f).lessThan(Tensor.scalar(3.0f));
        assertEquals(1L, getConstant(aLtB).value().longValue());
        assertEquals(1L, getConstant(bLtC).value().longValue());
    }

    @Test
    void foldsMixedIntAndFloatComparison() {
        Tensor result = Tensor.scalar(5L, DataType.I8).lessThan(Tensor.scalar(5.5f));
        assertTrue(result.isLazy());
        assertEquals(DataType.BOOL, result.dataType());
        assertEquals(1L, getConstant(result).value().longValue());
    }

    private static ConstantComputation getConstant(Tensor tensor) {
        return tensor.computation()
                .filter(ConstantComputation.class::isInstance)
                .map(ConstantComputation.class::cast)
                .orElseThrow(() -> new AssertionError("Expected ConstantComputation"));
    }
}
