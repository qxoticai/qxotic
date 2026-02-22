package com.qxotic.jota;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class TypeRulesTest {

    @Test
    void promotesIntegralTypes() {
        assertEquals(DataType.I64, TypeRules.promote(DataType.I8, DataType.I64));
        assertEquals(DataType.I32, TypeRules.promote(DataType.I16, DataType.I32));
    }

    @Test
    void promotesFloatTypes() {
        assertEquals(DataType.FP64, TypeRules.promote(DataType.FP64, DataType.FP32));
        assertEquals(DataType.FP32, TypeRules.promote(DataType.FP16, DataType.FP32));
        assertEquals(DataType.FP32, TypeRules.promote(DataType.BF16, DataType.FP32));
        assertEquals(DataType.FP64, TypeRules.promote(DataType.FP16, DataType.FP64));
        assertEquals(DataType.FP64, TypeRules.promote(DataType.BF16, DataType.FP64));
    }

    @Test
    void rejectsIncompatible16BitFloats() {
        assertThrows(
                IllegalArgumentException.class,
                () -> TypeRules.promote(DataType.FP16, DataType.BF16));
    }

    @Test
    void promotesIntegralToFloatLosslessly() {
        // I8 can promote to any float
        assertEquals(DataType.FP16, TypeRules.promote(DataType.I8, DataType.FP16));
        assertEquals(DataType.BF16, TypeRules.promote(DataType.I8, DataType.BF16));
        assertEquals(DataType.FP32, TypeRules.promote(DataType.I8, DataType.FP32));
        assertEquals(DataType.FP64, TypeRules.promote(DataType.I8, DataType.FP64));

        // I16 can promote to FP32 or FP64
        assertEquals(DataType.FP32, TypeRules.promote(DataType.I16, DataType.FP32));
        assertEquals(DataType.FP64, TypeRules.promote(DataType.I16, DataType.FP64));

        // I32 can only promote to FP64
        assertEquals(DataType.FP64, TypeRules.promote(DataType.I32, DataType.FP64));
    }

    @Test
    void rejectsLossyIntegralToFloat() {
        // I16 cannot promote to FP16 or BF16
        assertThrows(
                IllegalArgumentException.class,
                () -> TypeRules.promote(DataType.I16, DataType.FP16));
        assertThrows(
                IllegalArgumentException.class,
                () -> TypeRules.promote(DataType.I16, DataType.BF16));

        // I32 cannot promote to FP16, BF16, or FP32
        assertThrows(
                IllegalArgumentException.class,
                () -> TypeRules.promote(DataType.I32, DataType.FP16));
        assertThrows(
                IllegalArgumentException.class,
                () -> TypeRules.promote(DataType.I32, DataType.FP32));

        // I64 cannot promote to any float
        assertThrows(
                IllegalArgumentException.class,
                () -> TypeRules.promote(DataType.I64, DataType.FP32));
        assertThrows(
                IllegalArgumentException.class,
                () -> TypeRules.promote(DataType.I64, DataType.FP64));
    }

    @Test
    void rejectsQuantizedTypes() {
        assertThrows(
                IllegalArgumentException.class,
                () -> TypeRules.promote(DataType.Q8_0, DataType.FP32));
        assertThrows(
                IllegalArgumentException.class,
                () -> TypeRules.promote(DataType.I32, DataType.Q4_0));
    }

    @Test
    void rejectsBoolNumericPromotion() {
        assertThrows(
                IllegalArgumentException.class,
                () -> TypeRules.promote(DataType.BOOL, DataType.I8));
        assertThrows(
                IllegalArgumentException.class,
                () -> TypeRules.promote(DataType.BOOL, DataType.FP32));
        assertThrows(
                IllegalArgumentException.class,
                () -> TypeRules.promote(DataType.BOOL, DataType.BOOL));
    }

    @Test
    void comparisonPromotionAllowsBoolBoolOnly() {
        assertEquals(DataType.BOOL, TypeRules.promoteForComparison(DataType.BOOL, DataType.BOOL));
        assertThrows(
                IllegalArgumentException.class,
                () -> TypeRules.promoteForComparison(DataType.BOOL, DataType.I32));
    }
}
