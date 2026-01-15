package com.qxotic.jota;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class TypeRulesTest {

    @Test
    void promotesIntegralTypes() {
        assertEquals(DataType.I64, TypeRules.promote(DataType.I8, DataType.I64));
        assertEquals(DataType.I32, TypeRules.promote(DataType.I16, DataType.I32));
    }

    @Test
    void promotesFloatTypes() {
        assertEquals(DataType.FP64, TypeRules.promote(DataType.FP64, DataType.FP32));
        assertEquals(DataType.FP32, TypeRules.promote(DataType.FP16, DataType.BF16));
    }

    @Test
    void promotesIntegralToFloat() {
        assertEquals(DataType.FP32, TypeRules.promote(DataType.I32, DataType.FP16));
    }

    @Test
    void rejectsQuantizedTypes() {
        assertThrows(IllegalArgumentException.class, () -> TypeRules.promote(DataType.Q8_0, DataType.FP32));
        assertThrows(IllegalArgumentException.class, () -> TypeRules.promote(DataType.I32, DataType.Q4_0));
    }
}
