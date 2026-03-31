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

    @Test
    public void testSizes() {
        assertEquals(1, DType.BOOL.size());
        assertEquals(1, DType.U8.size());
        assertEquals(1, DType.I8.size());
        assertEquals(2, DType.I16.size());
        assertEquals(2, DType.U16.size());
        assertEquals(2, DType.F16.size());
        assertEquals(2, DType.BF16.size());
        assertEquals(4, DType.I32.size());
        assertEquals(4, DType.U32.size());
        assertEquals(4, DType.F32.size());
        assertEquals(8, DType.F64.size());
        assertEquals(8, DType.I64.size());
        assertEquals(8, DType.U64.size());
    }
}
