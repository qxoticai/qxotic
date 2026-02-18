package ai.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.*;
import org.junit.jupiter.api.Test;

class JSON2ParseOptionsTest {

    @Test
    void testDefaultOptionsReturnsDoubleForFloats() {
        Object result =
                JSON.parse("3.14", JSON.ParseOptions.defaults().decimalsAsBigDecimal(false));
        assertTrue(
                result instanceof Double, "Expected Double, got: " + result.getClass().getName());
        assertEquals(3.14, result);
    }

    @Test
    void testUseBigDecimalForFloats() {
        Object result = JSON.parse("3.14", JSON.ParseOptions.defaults().decimalsAsBigDecimal(true));
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
        Object result = JSON.parse("42", JSON.ParseOptions.defaults().decimalsAsBigDecimal(true));
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
        Object result = JSON.parse("-0", JSON.ParseOptions.defaults().decimalsAsBigDecimal(true));
        assertTrue(result instanceof BigDecimal);
    }

    @Test
    void testDefaultMaxParsingDepth() {
        JSON.ParseOptions options = JSON.ParseOptions.defaults();
        assertEquals(1000, options.maxDepth());
    }

    @Test
    void testCustomMaxParsingDepth() {
        JSON.ParseOptions options = JSON.ParseOptions.defaults().maxDepth(100);
        assertEquals(100, options.maxDepth());
    }

    @Test
    void testMaxParsingDepthTooLowThrows() {
        assertThrows(
                IllegalArgumentException.class, () -> JSON.ParseOptions.defaults().maxDepth(0));

        assertThrows(
                IllegalArgumentException.class, () -> JSON.ParseOptions.defaults().maxDepth(-1));
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
                        () -> JSON.parse(sb.toString(), JSON.ParseOptions.defaults().maxDepth(50)));
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

        Object result = JSON.parse(sb.toString(), JSON.ParseOptions.defaults().maxDepth(50));
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
                        () -> JSON.parse(sb.toString(), JSON.ParseOptions.defaults().maxDepth(25)));
        assertTrue(e.getMessage().contains("Maximum parsing depth exceeded"));
    }

    @Test
    void testCombinedOptions() {
        Object result =
                JSON.parse(
                        "3.14",
                        JSON.ParseOptions.defaults().decimalsAsBigDecimal(true).maxDepth(500));

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
        Object result =
                JSON.parse("1.5e10", JSON.ParseOptions.defaults().decimalsAsBigDecimal(true));
        assertTrue(result instanceof BigDecimal);
    }

    @Test
    void testExponentWithDouble() {
        Object result =
                JSON.parse("1.5e10", JSON.ParseOptions.defaults().decimalsAsBigDecimal(false));
        assertTrue(
                result instanceof Double, "Expected Double, got: " + result.getClass().getName());
    }

    @Test
    void testZeroWithBigDecimal() {
        Object result = JSON.parse("0.0", JSON.ParseOptions.defaults().decimalsAsBigDecimal(true));
        assertTrue(result instanceof BigDecimal);
        assertEquals(new BigDecimal("0"), result);
    }

    @Test
    void testPrecisionPreservedInBigDecimalMode() {
        String json = "0.123456789012345678901234567890";
        Object result = JSON.parse(json, JSON.ParseOptions.defaults().decimalsAsBigDecimal(true));
        assertTrue(result instanceof BigDecimal);
        assertEquals(new BigDecimal("0.123456789012345678901234567890"), result);
    }

    @Test
    void testSimpleObjectWithOptions() {
        String json = "{\"name\":\"John\",\"value\":3.14}";
        Map<?, ?> result =
                (Map<?, ?>)
                        JSON.parse(json, JSON.ParseOptions.defaults().decimalsAsBigDecimal(true));

        assertEquals("John", result.get("name"));
        assertEquals(new BigDecimal("3.14"), result.get("value"));
    }

    @Test
    void testSimpleArrayWithOptions() {
        String json = "[1, 2.5, 3]";
        List<?> result =
                (List<?>) JSON.parse(json, JSON.ParseOptions.defaults().decimalsAsBigDecimal(true));

        assertEquals(1L, result.get(0));
        assertEquals(new BigDecimal("2.5"), result.get(1));
        assertEquals(3L, result.get(2));
    }

    @Test
    void testStaticCreateMethod() {
        JSON.ParseOptions options = JSON.ParseOptions.defaults();
        assertNotNull(options);
        assertEquals(1000, options.maxDepth());
    }

    @Test
    void testDefaultsFactoryMethod() {
        JSON.ParseOptions options = JSON.ParseOptions.defaults();
        assertNotNull(options);
        assertEquals(1000, options.maxDepth());
        assertTrue(options.decimalsAsBigDecimal());
    }

    @Test
    void testBigDecimalPreset() {
        JSON.ParseOptions options = JSON.ParseOptions.defaults().decimalsAsBigDecimal(true);
        assertTrue(options.decimalsAsBigDecimal());
    }

    @Test
    void testDoublePrecisionPreset() {
        JSON.ParseOptions options = JSON.ParseOptions.defaults().decimalsAsBigDecimal(false);
        assertFalse(options.decimalsAsBigDecimal());
    }

    @Test
    void testSimplifiedOptionsNaming() {
        JSON.ParseOptions options =
                JSON.ParseOptions.defaults()
                        .decimalsAsBigDecimal(false)
                        .maxDepth(123)
                        .failOnDuplicateKeys(true);

        assertFalse(options.decimalsAsBigDecimal());
        assertEquals(123, options.maxDepth());
        assertTrue(options.failOnDuplicateKeys());
    }

    @Test
    void testNullOptionsRejected() {
        assertThrows(NullPointerException.class, () -> JSON.parse("3.14", null));
    }

    @Test
    void testDuplicateKeysAllowedByDefaultLastWins() {
        Map<?, ?> obj = (Map<?, ?>) JSON.parse("{\"a\":1,\"a\":2}");
        assertEquals(2L, obj.get("a"));
    }

    @Test
    void testFailOnDuplicateKeysTopLevel() {
        JSON.ParseException e =
                assertThrows(
                        JSON.ParseException.class,
                        () ->
                                JSON.parse(
                                        "{\"a\":1,\"a\":2}",
                                        JSON.ParseOptions.defaults().failOnDuplicateKeys(true)));
        assertTrue(e.getMessage().contains("Duplicate key"));
    }

    @Test
    void testFailOnDuplicateKeysNestedObject() {
        JSON.ParseException e =
                assertThrows(
                        JSON.ParseException.class,
                        () ->
                                JSON.parse(
                                        "{\"outer\":{\"x\":1,\"x\":2}}",
                                        JSON.ParseOptions.defaults().failOnDuplicateKeys(true)));
        assertTrue(e.getMessage().contains("Duplicate key"));
    }

    @Test
    void testChainedOptions() {
        JSON.ParseOptions options =
                JSON.ParseOptions.defaults()
                        .decimalsAsBigDecimal(true)
                        .maxDepth(200)
                        .decimalsAsBigDecimal(true);

        assertTrue(options.decimalsAsBigDecimal());
        assertEquals(200, options.maxDepth());
    }
}
