package com.qxotic.jota.runtime.clike;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.DataType;
import org.junit.jupiter.api.Test;

class CLikeExprSupportTest {

    @Test
    void normalizesShiftByTypeWidth() {
        assertEquals("((int)(rhs) & 7)", CLikeExprSupport.normalizedShift(DataType.I8, "rhs"));
        assertEquals("((int)(rhs) & 15)", CLikeExprSupport.normalizedShift(DataType.I16, "rhs"));
        assertEquals("((int)(rhs) & 31)", CLikeExprSupport.normalizedShift(DataType.I32, "rhs"));
        assertEquals("((int)(rhs) & 63)", CLikeExprSupport.normalizedShift(DataType.I64, "rhs"));
    }

    @Test
    void formatsFloatAndDoubleLiteralsWithStableSuffixes() {
        assertEquals("1.0f", CLikeExprSupport.formatFloatLiteral(1.0f));
        assertEquals("1.5f", CLikeExprSupport.formatFloatLiteral(1.5f));
        assertEquals("1.0", CLikeExprSupport.formatDoubleLiteral(1.0));
        assertEquals("1.5", CLikeExprSupport.formatDoubleLiteral(1.5));
    }
}
