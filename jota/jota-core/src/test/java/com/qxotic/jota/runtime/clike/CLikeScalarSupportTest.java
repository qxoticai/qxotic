package com.qxotic.jota.runtime.clike;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.DataType;
import org.junit.jupiter.api.Test;

class CLikeScalarSupportTest {

    @Test
    void emitsTypedTernaryCondition() {
        assertEquals(
                "(cond ? t : f)",
                CLikeScalarSupport.ternaryExpr("cond", DataType.BOOL, "t", "f"));
        assertEquals(
                "(cond != 0 ? t : f)",
                CLikeScalarSupport.ternaryExpr("cond", DataType.I32, "t", "f"));
    }

    @Test
    void normalizesBoolStoreValue() {
        assertEquals("(v ? 1 : 0)", CLikeScalarSupport.boolStoreValue("v", DataType.BOOL));
        assertEquals("(v != 0 ? 1 : 0)", CLikeScalarSupport.boolStoreValue("v", DataType.FP32));
    }

    @Test
    void buildsComparisonWithExpectedCoercions() {
        String expr =
                CLikeScalarSupport.comparisonExpr(
                        "<",
                        "left",
                        DataType.BOOL,
                        "right",
                        DataType.FP16,
                        DataType.I32,
                        (source, target, value) -> "((" + target + ")" + value + ")",
                        (source, value) -> "((float)" + value + ")");

        assertEquals("((((fp32)left) < ((float)right)) ? 1 : 0)", expr);
    }

    @Test
    void keepsBoolComparisonsInWiderNumericDomain() {
        String expr =
                CLikeScalarSupport.comparisonExpr(
                        "<",
                        "left",
                        DataType.BOOL,
                        "right",
                        DataType.FP64,
                        DataType.BOOL,
                        (source, target, value) -> "cast(" + source + "->" + target + "," + value + ")",
                        (source, value) -> "toFloat(" + source + "," + value + ")");

        assertEquals("(cast(bool->fp64,left) < right)", expr);
    }
}
