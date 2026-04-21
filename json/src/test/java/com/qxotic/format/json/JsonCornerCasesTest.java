package com.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class JsonCornerCasesTest {

    @Test
    void testDeepNesting() {
        int depth = 50;
        StringBuilder sb = new StringBuilder();
        sb.append("{\"a\":".repeat(depth));
        sb.append("1");
        sb.append("}".repeat(depth));
        Object result = Json.parse(sb.toString());
        for (int i = 0; i < depth; i++) {
            assertInstanceOf(Map.class, result);
            result = ((Map<?, ?>) result).get("a");
        }
        assertEquals(1L, result);
    }

    @Test
    void testWideNestedArrays() {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < 50; i++) {
            if (i > 0) sb.append(",");
            sb.append(i);
        }
        sb.append("]");
        List<?> result = (List<?>) Json.parse(sb.toString());
        assertEquals(50, result.size());
        for (int i = 0; i < 50; i++) {
            assertEquals((long) i, result.get(i));
        }
    }

    @Test
    void testWideNestedObjects() {
        StringBuilder sb = new StringBuilder("{");
        for (int i = 0; i < 50; i++) {
            if (i > 0) sb.append(",");
            sb.append("\"key").append(i).append("\":").append(i);
        }
        sb.append("}");
        Map<?, ?> result = (Map<?, ?>) Json.parse(sb.toString());
        assertEquals(50, result.size());
        for (int i = 0; i < 50; i++) {
            assertEquals((long) i, result.get("key" + i));
        }
    }

    @Test
    void testLargeString() {
        StringBuilder sb = new StringBuilder("\"");
        sb.append("x".repeat(1000));
        sb.append("\"");
        String result = (String) Json.parse(sb.toString());
        assertEquals(1000, result.length());
    }

    @Test
    void testVeryLargeNumber() {
        StringBuilder sb = new StringBuilder("1");
        sb.append("0".repeat(500));
        Object result = Json.parse(sb.toString());
        assertTrue(result instanceof BigInteger || result instanceof BigDecimal);
    }

    @Test
    void testManyDecimals() {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < 50; i++) {
            if (i > 0) sb.append(",");
            sb.append("0.").append(String.format("%03d", i));
        }
        sb.append("]");
        List<?> result = (List<?>) Json.parse(sb.toString());
        assertEquals(50, result.size());
    }

    @Test
    void testComplexMixedNesting() {
        String json = "{\"a\":[1,2,{\"b\":[3,4,{\"c\":5}]},6],\"d\":{\"e\":{\"f\":7}}}";
        Object result = Json.parse(json);
        assertInstanceOf(Map.class, result);
    }

    @Test
    void testStringWithEscapedQuotes() {
        String input = "\"He said, \\\"Hello World!\\\"\"";
        String result = (String) Json.parse(input);
        assertEquals("He said, \"Hello World!\"", result);
    }

    @Test
    void testStringWithMultipleEscapes() {
        String input = "\"Line1\\nLine2\\tTab\\rReturn\\\\Backslash\\/Slash\"";
        String result = (String) Json.parse(input);
        assertEquals("Line1\nLine2\tTab\rReturn\\Backslash/Slash", result);
    }

    @Test
    void testUnicodeEscapesInSequence() {
        String input = "\"\\u0041\\u0042\\u0043\\u0044\\u0045\"";
        String result = (String) Json.parse(input);
        assertEquals("ABCDE", result);
    }

    @Test
    void testArrayWithNulls() {
        String json = "[null,1,null,\"str\",null,[],null,{}]";
        List<?> result = (List<?>) Json.parse(json);
        assertEquals(8, result.size());
        assertSame(Json.NULL, result.get(0));
        assertEquals(1L, result.get(1));
        assertSame(Json.NULL, result.get(2));
        assertEquals("str", result.get(3));
        assertSame(Json.NULL, result.get(4));
        assertInstanceOf(List.class, result.get(5));
        assertSame(Json.NULL, result.get(6));
        assertInstanceOf(Map.class, result.get(7));
    }

    @Test
    void testObjectWithNulls() {
        String json = "{\"a\":null,\"b\":1,\"c\":null,\"d\":\"str\"}";
        Map<?, ?> result = (Map<?, ?>) Json.parse(json);
        assertEquals(4, result.size());
        assertSame(Json.NULL, result.get("a"));
        assertEquals(1L, result.get("b"));
        assertSame(Json.NULL, result.get("c"));
        assertEquals("str", result.get("d"));
    }

    @Test
    void testBooleanMixedWithNull() {
        String json = "[true,false,null,true,false,null]";
        List<?> result = (List<?>) Json.parse(json);
        assertEquals(6, result.size());
        assertEquals(true, result.get(0));
        assertEquals(false, result.get(1));
        assertSame(Json.NULL, result.get(2));
        assertEquals(true, result.get(3));
        assertEquals(false, result.get(4));
        assertSame(Json.NULL, result.get(5));
    }

    @Test
    void testWhitespaceVariations() {
        String json = "  \t\n  {  \t\n  \"key\"  \t\n  :  \t\n  \"value\"  \t\n  }  \t\n  ";
        Map<?, ?> result = (Map<?, ?>) Json.parse(json);
        assertEquals("value", result.get("key"));
    }

    @Test
    void testNumbersWithExponents() {
        String json = "[1e0,1E1,1e-1,1E+1,1e10,1E-10]";
        List<?> result = (List<?>) Json.parse(json);
        assertEquals(6, result.size());
        for (Object o : result) {
            assertInstanceOf(BigDecimal.class, o);
        }
    }

    @Test
    void testNumbersWithDecimals() {
        String json = "[0.0,0.1,1.0,10.0,0.01,1.23,123.456]";
        List<?> result = (List<?>) Json.parse(json);
        assertEquals(7, result.size());
        for (Object o : result) {
            assertInstanceOf(BigDecimal.class, o);
        }
    }

    @Test
    void testNegativeNumbers() {
        String json = "[-0,-1,-10,-0.5,-1e10,-123.456]";
        List<?> result = (List<?>) Json.parse(json);
        assertEquals(6, result.size());
    }

    @Test
    void testNumberZeroVariations() {
        String json = "[0,-0,0.0,-0.0,0e0,0E0]";
        List<?> result = (List<?>) Json.parse(json);
        assertEquals(6, result.size());
    }

    @Test
    void testEmptyKey() {
        String json = "{\"\": \"empty key\"}";
        Map<?, ?> result = (Map<?, ?>) Json.parse(json);
        assertEquals(1, result.size());
        assertEquals("empty key", result.get(""));
    }

    @Test
    void testSpecialCharsInKey() {
        String json = "{\"a-b_c.d/e\": 1}";
        Map<?, ?> result = (Map<?, ?>) Json.parse(json);
        assertEquals(1, result.size());
        assertEquals(1L, result.get("a-b_c.d/e"));
    }

    @Test
    void testMaxSafeInteger() {
        long maxSafe = 9007199254740991L;
        String json = String.valueOf(maxSafe);
        Object result = Json.parse(json);
        assertEquals(maxSafe, result);
    }

    @Test
    void testMinSafeInteger() {
        long minSafe = -9007199254740991L;
        String json = String.valueOf(minSafe);
        Object result = Json.parse(json);
        assertEquals(minSafe, result);
    }

    @Test
    void testStringifiedNumber() {
        String json = "12345678901234567890";
        Object result = Json.parse(json);
        assertInstanceOf(BigInteger.class, result);
        assertEquals(new BigInteger("12345678901234567890"), result);
    }

    @Test
    void testPreciseDecimal() {
        String json = "0.123456789012345678901234567890";
        Object result = Json.parse(json);
        assertInstanceOf(BigDecimal.class, result);
    }
}
