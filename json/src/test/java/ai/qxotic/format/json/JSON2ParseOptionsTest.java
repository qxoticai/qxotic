package ai.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.*;
import org.junit.jupiter.api.Test;

class JSON2ParseOptionsTest {

    @Test
    void testDefaultOptionsReturnsDoubleForFloats() {
        Object result = JSON.parse("3.14", JSON.ParseOptions.defaults().useDoubleForFloats());
        assertTrue(
                result instanceof Double, "Expected Double, got: " + result.getClass().getName());
        assertEquals(3.14, result);
    }

    @Test
    void testUseBigDecimalForFloats() {
        Object result = JSON.parse("3.14", JSON.ParseOptions.defaults().useBigDecimalForFloats());
        assertTrue(
                result instanceof BigDecimal,
                "Expected BigDecimal, got: " + result.getClass().getName());
        assertEquals(new BigDecimal("3.14"), result);
    }

    @Test
    void testIntegersAlwaysReturnLong() {
        Object result = JSON.parse("42");
        assertTrue(result instanceof Long, "Expected Long, got: " + result.getClass().getName());
        assertEquals(42L, result);
    }

    @Test
    void testIntegersReturnLongRegardlessOfOptions() {
        Object result = JSON.parse("42", JSON.ParseOptions.defaults().useBigDecimalForFloats());
        assertTrue(result instanceof Long, "Expected Long, got: " + result.getClass().getName());
    }

    @Test
    void testLargeIntegersReturnBigInteger() {
        Object result = JSON.parse("9999999999999999999999999999999999999999");
        assertTrue(
                result instanceof BigInteger,
                "Expected BigInteger, got: " + result.getClass().getName());
    }

    @Test
    void testNegativeZeroInDoubleMode() {
        Object result = JSON.parse("-0", JSON.ParseOptions.defaults().useBigDecimalForFloats());
        assertTrue(result instanceof BigDecimal);
    }

    @Test
    void testDefaultMaxParsingDepth() {
        JSON.ParseOptions options = JSON.ParseOptions.defaults();
        assertEquals(1000, options.getMaxParsingDepth());
    }

    @Test
    void testCustomMaxParsingDepth() {
        JSON.ParseOptions options = JSON.ParseOptions.defaults().maxParsingDepth(100);
        assertEquals(100, options.getMaxParsingDepth());
    }

    @Test
    void testMaxParsingDepthTooLowThrows() {
        assertThrows(
                IllegalArgumentException.class,
                () -> JSON.ParseOptions.defaults().maxParsingDepth(0));

        assertThrows(
                IllegalArgumentException.class,
                () -> JSON.ParseOptions.defaults().maxParsingDepth(-1));
    }

    @Test
    void testMaxParsingDepthExceededThrows() {
        StringBuilder sb = new StringBuilder();
        int depth = 51;
        for (int i = 0; i < depth; i++) {
            sb.append("{\"a\":");
        }
        sb.append("1");
        for (int i = 0; i < depth; i++) {
            sb.append("}");
        }

        JSON.ParseException e =
                assertThrows(
                        JSON.ParseException.class,
                        () ->
                                JSON.parse(
                                        sb.toString(),
                                        JSON.ParseOptions.defaults().maxParsingDepth(50)));
        assertTrue(e.getMessage().contains("Maximum parsing depth exceeded"));
    }

    @Test
    void testMaxParsingDepthNotExceeded() {
        StringBuilder sb = new StringBuilder();
        int depth = 10;
        for (int i = 0; i < depth; i++) {
            sb.append("{\"a\":");
        }
        sb.append("1");
        for (int i = 0; i < depth; i++) {
            sb.append("}");
        }

        Object result = JSON.parse(sb.toString(), JSON.ParseOptions.defaults().maxParsingDepth(50));
        assertNotNull(result);
    }

    @Test
    void testNestedArraysWithDepthLimit() {
        StringBuilder sb = new StringBuilder();
        int depth = 26;
        for (int i = 0; i < depth; i++) {
            sb.append("[");
        }
        sb.append("1");
        for (int i = 0; i < depth; i++) {
            sb.append("]");
        }

        JSON.ParseException e =
                assertThrows(
                        JSON.ParseException.class,
                        () ->
                                JSON.parse(
                                        sb.toString(),
                                        JSON.ParseOptions.defaults().maxParsingDepth(25)));
        assertTrue(e.getMessage().contains("Maximum parsing depth exceeded"));
    }

    @Test
    void testCombinedOptions() {
        Object result =
                JSON.parse(
                        "3.14",
                        JSON.ParseOptions.defaults().useBigDecimalForFloats().maxParsingDepth(500));

        assertTrue(result instanceof BigDecimal);
    }

    @Test
    void testDefaultParseStillUsesDefaultMaxDepth() {
        StringBuilder sb = new StringBuilder();
        int depth = 1001;
        for (int i = 0; i < depth; i++) {
            sb.append("{\"a\":");
        }
        sb.append("1");
        for (int i = 0; i < depth; i++) {
            sb.append("}");
        }

        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse(sb.toString()));
        assertTrue(e.getMessage().contains("Maximum parsing depth exceeded"));
    }

    @Test
    void testExponentWithBigDecimal() {
        Object result = JSON.parse("1.5e10", JSON.ParseOptions.defaults().useBigDecimalForFloats());
        assertTrue(result instanceof BigDecimal);
    }

    @Test
    void testExponentWithDouble() {
        Object result = JSON.parse("1.5e10", JSON.ParseOptions.defaults().useDoubleForFloats());
        assertTrue(
                result instanceof Double, "Expected Double, got: " + result.getClass().getName());
    }

    @Test
    void testZeroWithBigDecimal() {
        Object result = JSON.parse("0.0", JSON.ParseOptions.defaults().useBigDecimalForFloats());
        assertTrue(result instanceof BigDecimal);
        assertEquals(new BigDecimal("0"), result);
    }

    @Test
    void testPrecisionPreservedInBigDecimalMode() {
        String json = "0.123456789012345678901234567890";
        Object result = JSON.parse(json, JSON.ParseOptions.defaults().useBigDecimalForFloats());
        assertTrue(result instanceof BigDecimal);
        assertEquals(new BigDecimal("0.123456789012345678901234567890"), result);
    }

    @Test
    void testSimpleObjectWithOptions() {
        String json = "{\"name\":\"John\",\"value\":3.14}";
        Map<?, ?> result =
                (Map<?, ?>) JSON.parse(json, JSON.ParseOptions.defaults().useBigDecimalForFloats());

        assertEquals("John", result.get("name"));
        assertEquals(new BigDecimal("3.14"), result.get("value"));
    }

    @Test
    void testSimpleArrayWithOptions() {
        String json = "[1, 2.5, 3]";
        List<?> result =
                (List<?>) JSON.parse(json, JSON.ParseOptions.defaults().useBigDecimalForFloats());

        assertEquals(1L, result.get(0));
        assertEquals(new BigDecimal("2.5"), result.get(1));
        assertEquals(3L, result.get(2));
    }

    @Test
    void testStaticCreateMethod() {
        JSON.ParseOptions options = JSON.ParseOptions.defaults();
        assertNotNull(options);
        assertEquals(1000, options.getMaxParsingDepth());
    }

    @Test
    void testDefaultsFactoryMethod() {
        JSON.ParseOptions options = JSON.ParseOptions.defaults();
        assertNotNull(options);
        assertEquals(1000, options.getMaxParsingDepth());
        assertTrue(options.shouldUseBigDecimal());
    }

    @Test
    void testBigDecimalPreset() {
        JSON.ParseOptions options = JSON.ParseOptions.bigDecimal();
        assertTrue(options.shouldUseBigDecimal());
    }

    @Test
    void testDoublePrecisionPreset() {
        JSON.ParseOptions options = JSON.ParseOptions.doublePrecision();
        assertFalse(options.shouldUseBigDecimal());
    }

    @Test
    void testNullOptionsRejected() {
        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> JSON.parse("3.14", null));
        assertTrue(e.getMessage().contains("options must not be null"));
    }

    @Test
    void testChainedOptions() {
        JSON.ParseOptions options =
                JSON.ParseOptions.defaults()
                        .useBigDecimalForFloats()
                        .maxParsingDepth(200)
                        .useBigDecimalForFloats();

        assertTrue(options.shouldUseBigDecimal());
        assertEquals(200, options.getMaxParsingDepth());
    }
}
