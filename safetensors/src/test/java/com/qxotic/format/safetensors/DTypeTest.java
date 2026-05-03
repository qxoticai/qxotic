package com.qxotic.format.safetensors;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

public class DTypeTest {

    @Test
    public void testJavaTypes() {
        assertEquals(boolean.class, DType.BOOL.javaType());
        assertEquals(byte.class, DType.U8.javaType());
        assertEquals(byte.class, DType.I8.javaType());
        assertEquals(short.class, DType.I16.javaType());
        assertEquals(short.class, DType.U16.javaType());
        assertEquals(short.class, DType.F16.javaType());
        assertEquals(short.class, DType.BF16.javaType());
        assertEquals(int.class, DType.I32.javaType());
        assertEquals(int.class, DType.U32.javaType());
        assertEquals(float.class, DType.F32.javaType());
        assertEquals(double.class, DType.F64.javaType());
        assertEquals(long.class, DType.I64.javaType());
        assertEquals(long.class, DType.U64.javaType());
    }
}
