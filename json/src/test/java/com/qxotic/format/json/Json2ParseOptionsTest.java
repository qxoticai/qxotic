package com.qxotic.format.json;

import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class Json2ParseOptionsTest {

    @Test
    void testDefaultOptionsReturnsDoubleForFloats() {
        Object result =
                Json.parse("3.14", Json.ParseOptions.defaults().decimalsAsBigDecimal(false));
        assertInstanceOf(
                Double.class, result, "Expected Double, got: " + result.getClass().getName());
        assertEquals(3.14, result);
    }

    @Test
    void testUseBigDecimalForFloats() {
        Object result = Json.parse("3.14", Json.ParseOptions.defaults().decimalsAsBigDecimal(true));
        assertInstanceOf(
                BigDecimal.class,
                result,
                "Expected BigDecimal, got: " + result.getClass().getName());
        assertEquals(new BigDecimal("3.14"), result);
    }

    @Test
    void testIntegersAlwaysReturnLong() {
        Object result = Json.parse("42");
        assertInstanceOf(Long.class, result, "Expected Long, got: " + result.getClass().getName());
        assertEquals(42L, result);
    }

    @Test
    void testIntegersReturnLongRegardlessOfOptions() {
        Object result = Json.parse("42", Json.ParseOptions.defaults().decimalsAsBigDecimal(true));
        assertInstanceOf(Long.class, result, "Expected Long, got: " + result.getClass().getName());
    }

    @Test
    void testLargeIntegersReturnBigInteger() {
        Object result = Json.parse("9999999999999999999999999999999999999999");
        assertInstanceOf(
                BigInteger.class,
                result,
                "Expected BigInteger, got: " + result.getClass().getName());
    }

    @Test
    void testNegativeZeroInDoubleMode() {
        Object result = Json.parse("-0", Json.ParseOptions.defaults().decimalsAsBigDecimal(true));
        // -0 as integer parses to 0L regardless of decimalsAsBigDecimal
        assertEquals(0L, result);
    }

    @Test
    void testDefaultMaxParsingDepth() {
        Json.ParseOptions options = Json.ParseOptions.defaults();
        assertEquals(1000, options.maxDepth());
    }

    @Test
    void testCustomMaxParsingDepth() {
        Json.ParseOptions options = Json.ParseOptions.defaults().maxDepth(100);
        assertEquals(100, options.maxDepth());
    }

    @Test
    void testMaxParsingDepthTooLowThrows() {
        assertThrows(
                IllegalArgumentException.class, () -> Json.ParseOptions.defaults().maxDepth(0));

        assertThrows(
                IllegalArgumentException.class, () -> Json.ParseOptions.defaults().maxDepth(-1));
    }

    @Test
    void testMaxParsingDepthExceededThrows() {
        StringBuilder sb = new StringBuilder();
        int depth = 51;
        sb.append("{\"a\":".repeat(depth));
        sb.append("1");
        sb.append("}".repeat(depth));

        Json.ParseException e =
                assertThrows(
                        Json.ParseException.class,
                        () -> Json.parse(sb.toString(), Json.ParseOptions.defaults().maxDepth(50)));
        assertTrue(e.getMessage().contains("Maximum parsing depth exceeded"));
    }

    @Test
    void testMaxParsingDepthNotExceeded() {
        StringBuilder sb = new StringBuilder();
        int depth = 10;
        sb.append("{\"a\":".repeat(depth));
        sb.append("1");
        sb.append("}".repeat(depth));

        Object result = Json.parse(sb.toString(), Json.ParseOptions.defaults().maxDepth(50));
        assertNotNull(result);
    }

    @Test
    void testNestedArraysWithDepthLimit() {
        StringBuilder sb = new StringBuilder();
        int depth = 26;
        sb.append("[".repeat(depth));
        sb.append("1");
        sb.append("]".repeat(depth));

        Json.ParseException e =
                assertThrows(
                        Json.ParseException.class,
                        () -> Json.parse(sb.toString(), Json.ParseOptions.defaults().maxDepth(25)));
        assertTrue(e.getMessage().contains("Maximum parsing depth exceeded"));
    }

    @Test
    void testCombinedOptions() {
        Object result =
                Json.parse(
                        "3.14",
                        Json.ParseOptions.defaults().decimalsAsBigDecimal(true).maxDepth(500));

        assertInstanceOf(BigDecimal.class, result);
    }

    @Test
    void testDefaultParseStillUsesDefaultMaxDepth() {
        StringBuilder sb = new StringBuilder();
        int depth = 1001;
        sb.append("{\"a\":".repeat(depth));
        sb.append("1");
        sb.append("}".repeat(depth));

        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse(sb.toString()));
        assertTrue(e.getMessage().contains("Maximum parsing depth exceeded"));
    }

    @Test
    void testExponentWithBigDecimal() {
        Object result =
                Json.parse("1.5e10", Json.ParseOptions.defaults().decimalsAsBigDecimal(true));
        assertInstanceOf(BigDecimal.class, result);
    }

    @Test
    void testExponentWithDouble() {
        Object result =
                Json.parse("1.5e10", Json.ParseOptions.defaults().decimalsAsBigDecimal(false));
        assertInstanceOf(
                Double.class, result, "Expected Double, got: " + result.getClass().getName());
    }

    @Test
    void testZeroWithBigDecimal() {
        Object result = Json.parse("0.0", Json.ParseOptions.defaults().decimalsAsBigDecimal(true));
        assertInstanceOf(BigDecimal.class, result);
        assertEquals(new BigDecimal("0.0"), result);
    }

    @Test
    void testPrecisionPreservedInBigDecimalMode() {
        String json = "0.123456789012345678901234567890";
        Object result = Json.parse(json, Json.ParseOptions.defaults().decimalsAsBigDecimal(true));
        assertInstanceOf(BigDecimal.class, result);
        assertEquals(new BigDecimal("0.123456789012345678901234567890"), result);
    }

    @Test
    void testSimpleObjectWithOptions() {
        String json = "{\"name\":\"John\",\"value\":3.14}";
        Map<?, ?> result =
                (Map<?, ?>)
                        Json.parse(json, Json.ParseOptions.defaults().decimalsAsBigDecimal(true));

        assertEquals("John", result.get("name"));
        assertEquals(new BigDecimal("3.14"), result.get("value"));
    }

    @Test
    void testSimpleArrayWithOptions() {
        String json = "[1, 2.5, 3]";
        List<?> result =
                (List<?>) Json.parse(json, Json.ParseOptions.defaults().decimalsAsBigDecimal(true));

        assertEquals(1L, result.get(0));
        assertEquals(new BigDecimal("2.5"), result.get(1));
        assertEquals(3L, result.get(2));
    }

    @Test
    void testStaticCreateMethod() {
        Json.ParseOptions options = Json.ParseOptions.defaults();
        assertNotNull(options);
        assertEquals(1000, options.maxDepth());
    }

    @Test
    void testDefaultsFactoryMethod() {
        Json.ParseOptions options = Json.ParseOptions.defaults();
        assertNotNull(options);
        assertEquals(1000, options.maxDepth());
        assertTrue(options.decimalsAsBigDecimal());
    }

    @Test
    void testBigDecimalPreset() {
        Json.ParseOptions options = Json.ParseOptions.defaults().decimalsAsBigDecimal(true);
        assertTrue(options.decimalsAsBigDecimal());
    }

    @Test
    void testDoublePrecisionPreset() {
        Json.ParseOptions options = Json.ParseOptions.defaults().decimalsAsBigDecimal(false);
        assertFalse(options.decimalsAsBigDecimal());
    }

    @Test
    void testSimplifiedOptionsNaming() {
        Json.ParseOptions options =
                Json.ParseOptions.defaults()
                        .decimalsAsBigDecimal(false)
                        .maxDepth(123)
                        .failOnDuplicateKeys(true);

        assertFalse(options.decimalsAsBigDecimal());
        assertEquals(123, options.maxDepth());
        assertTrue(options.failOnDuplicateKeys());
    }

    @Test
    void testNullOptionsRejected() {
        assertThrows(NullPointerException.class, () -> Json.parse("3.14", null));
    }

    @Test
    void testDuplicateKeysAllowedByDefaultLastWins() {
        Map<?, ?> obj = (Map<?, ?>) Json.parse("{\"a\":1,\"a\":2}");
        assertEquals(2L, obj.get("a"));
    }

    @Test
    void testFailOnDuplicateKeysTopLevel() {
        Json.ParseException e =
                assertThrows(
                        Json.ParseException.class,
                        () ->
                                Json.parse(
                                        "{\"a\":1,\"a\":2}",
                                        Json.ParseOptions.defaults().failOnDuplicateKeys(true)));
        assertTrue(e.getMessage().contains("Duplicate key"));
    }

    @Test
    void testFailOnDuplicateKeysNestedObject() {
        Json.ParseException e =
                assertThrows(
                        Json.ParseException.class,
                        () ->
                                Json.parse(
                                        "{\"outer\":{\"x\":1,\"x\":2}}",
                                        Json.ParseOptions.defaults().failOnDuplicateKeys(true)));
        assertTrue(e.getMessage().contains("Duplicate key"));
    }

    @Test
    void testChainedOptions() {
        Json.ParseOptions options =
                Json.ParseOptions.defaults()
                        .decimalsAsBigDecimal(true)
                        .maxDepth(200)
                        .decimalsAsBigDecimal(true);

        assertTrue(options.decimalsAsBigDecimal());
        assertEquals(200, options.maxDepth());
    }
}
