package ai.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.*;
import org.junit.jupiter.api.Test;

class JSONPrinterTest {

    @Test
    void testCompactObject() {
        Map<String, Object> obj = new LinkedHashMap<>();
        obj.put("key", "value");
        String json = JSON.stringify(obj, false);
        assertTrue(json.contains("\"key\":\"value\""));
    }

    @Test
    void testCompactArray() {
        List<Object> arr = Arrays.asList(1, 2, 3);
        String json = JSON.stringify(arr, false);
        assertEquals("[1,2,3]", json);
    }

    @Test
    void testPrettyPrintObject() {
        Map<String, Object> obj = new LinkedHashMap<>();
        obj.put("name", "John");
        obj.put("age", 30);
        String json = JSON.stringify(obj, true);
        assertTrue(json.contains("\"name\" : \"John\""));
        assertTrue(json.contains("\n"));
    }

    @Test
    void testPrettyPrintArray() {
        List<Object> arr = Arrays.asList(1, 2, 3);
        String json = JSON.stringify(arr, true);
        assertTrue(json.contains("["));
        assertTrue(json.contains("\n"));
    }

    @Test
    void testNegativeZeroPrint() {
        String json = JSON.stringify(new BigDecimal("-0"), false);
        assertEquals("0", json);
    }

    @Test
    void testNegativeZeroInObjectPrint() {
        Map<String, Object> obj = new LinkedHashMap<>();
        obj.put("zero", new BigDecimal("-0"));
        String json = JSON.stringify(obj, false);
        assertTrue(json.contains("\"zero\":0"));
    }

    @Test
    void testLongPrint() {
        assertEquals("123", JSON.stringify(123L, false));
        assertEquals("-456", JSON.stringify(-456L, false));
    }

    @Test
    void testBigIntegerPrint() {
        BigInteger big = new BigInteger("9999999999999999999999999999999999999999");
        String json = JSON.stringify(big, false);
        assertEquals("9999999999999999999999999999999999999999", json);
    }

    @Test
    void testBigDecimalPrint() {
        BigDecimal bd = new BigDecimal("123.456");
        String json = JSON.stringify(bd, false);
        assertEquals("123.456", json);
    }

    @Test
    void testBigDecimalStripTrailingZeros() {
        BigDecimal bd = new BigDecimal("1.5000");
        String json = JSON.stringify(bd, false);
        assertEquals("1.5", json);
    }

    @Test
    void testDoublePrint() {
        assertEquals("1.5", JSON.stringify(1.5, false));
        assertEquals("2", JSON.stringify(2.0, false));
    }

    @Test
    void testFloatPrint() {
        assertEquals("1.5", JSON.stringify(1.5f, false));
        assertEquals("2", JSON.stringify(2.0f, false));
    }

    @Test
    void testBooleanPrint() {
        assertEquals("true", JSON.stringify(true, false));
        assertEquals("false", JSON.stringify(false, false));
    }

    @Test
    void testNullPrint() {
        assertEquals("null", JSON.stringify(JSON.NULL, false));
    }

    @Test
    void testStringPrint() {
        assertEquals("\"hello\"", JSON.stringify("hello", false));
    }

    @Test
    void testStringEscapesQuotes() {
        String json = JSON.stringify("\"test\"", false);
        assertEquals("\"\\\"test\\\"\"", json);
    }

    @Test
    void testStringEscapesBackslash() {
        String json = JSON.stringify("\\test", false);
        assertEquals("\"\\\\test\"", json);
    }

    @Test
    void testStringEscapesNewline() {
        String json = JSON.stringify("line1\nline2", false);
        assertEquals("\"line1\\nline2\"", json);
    }

    @Test
    void testStringEscapesTab() {
        String json = JSON.stringify("a\tb", false);
        assertEquals("\"a\\tb\"", json);
    }

    @Test
    void testStringEscapesUnicode() {
        String json = JSON.stringify("中文", false);
        // JSON spec allows direct Unicode output
        assertEquals("\"中文\"", json);
    }

    @Test
    void testStringEscapesSurrogates() {
        String json = JSON.stringify("\uD83D\uDE00", false);
        // JSON spec allows direct Unicode output for supplementary characters
        assertEquals("\"😀\"", json);
    }

    @Test
    void testEmptyArrayPrint() {
        assertEquals("[]", JSON.stringify(new ArrayList<>(), false));
    }

    @Test
    void testEmptyObjectPrint() {
        assertEquals("{}", JSON.stringify(new LinkedHashMap<>(), false));
    }

    @Test
    void testComplexStructurePrint() {
        Map<String, Object> obj = new LinkedHashMap<>();
        obj.put("array", Arrays.asList(1, 2, 3));
        obj.put("nested", new LinkedHashMap<>(Map.of("a", 1)));
        String json = JSON.stringify(obj, false);
        assertTrue(json.contains("\"array\":[1,2,3]"));
        assertTrue(json.contains("\"nested\":{\"a\":1}"));
    }

    @Test
    void testEmptyStringPrint() {
        assertEquals("\"\"", JSON.stringify("", false));
    }

    @Test
    void testWhitespaceInStringPrint() {
        assertEquals("\"  spaces  \"", JSON.stringify("  spaces  ", false));
    }

    @Test
    void testNaNThrows() {
        assertThrows(IllegalArgumentException.class, () -> JSON.stringify(Double.NaN, false));
        assertThrows(IllegalArgumentException.class, () -> JSON.stringify(Float.NaN, false));
    }

    @Test
    void testInfinityThrows() {
        assertThrows(
                IllegalArgumentException.class,
                () -> JSON.stringify(Double.POSITIVE_INFINITY, false));
        assertThrows(
                IllegalArgumentException.class,
                () -> JSON.stringify(Double.NEGATIVE_INFINITY, false));
        assertThrows(
                IllegalArgumentException.class,
                () -> JSON.stringify(Float.POSITIVE_INFINITY, false));
        assertThrows(
                IllegalArgumentException.class,
                () -> JSON.stringify(Float.NEGATIVE_INFINITY, false));
    }
}
