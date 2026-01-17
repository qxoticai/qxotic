package ai.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.*;
import org.junit.jupiter.api.Test;

class JSON2EdgeCasesTest {

    @Test
    void testLargeNumbers() {
        BigInteger bigInt = new BigInteger("9999999999999999999999999999999999999999");
        Object parsed = JSON.parse("9999999999999999999999999999999999999999");
        assertEquals(bigInt, parsed);
        assertEquals("9999999999999999999999999999999999999999", JSON.stringify(parsed));

        // Test large decimal with BigDecimal mode
        String largeDec = "12345678901234567890.12345678901234567890";
        Object bdParsed = JSON.parse(largeDec, JSON.ParseOptions.create().useBigDecimalForFloats());
        assertTrue(bdParsed instanceof BigDecimal);
        assertEquals(new BigDecimal(largeDec), bdParsed);
    }

    @Test
    void testUnicode() {
        String emoji = "\"\\uD83D\\uDE00\""; // 😀
        Object parsed = JSON.parse(emoji);
        assertEquals("\uD83D\uDE00", parsed);
        String stringified = JSON.stringify(parsed);
        // JSON spec allows direct Unicode output
        assertEquals("\"😀\"", stringified);

        String chinese = "\"\\u4E2D\\u6587\""; // 中文
        parsed = JSON.parse(chinese);
        assertEquals("\u4e2d\u6587", parsed);
        // JSON spec allows direct Unicode output
        assertEquals("\"中文\"", JSON.stringify(parsed));

        String mixed = "\"Hello \\u4E16\\u754C!\""; // Hello 世界!
        parsed = JSON.parse(mixed);
        assertEquals("Hello \u4e16\u754c!", parsed);
        // JSON spec allows direct Unicode output
        assertEquals("\"Hello 世界!\"", JSON.stringify(parsed));
    }

    @Test
    void testPrettyPrinting() {
        Map<String, Object> obj = new LinkedHashMap<>();
        obj.put("name", "John");
        obj.put("age", 30);
        obj.put("active", true);

        String compact = JSON.stringify(obj, false);
        assertEquals("{\"name\":\"John\",\"age\":30,\"active\":true}", compact);

        String pretty = JSON.stringify(obj, true);
        assertEquals(
                "{\n"
                        + "  \"name\" : \"John\",\n"
                        + "  \"age\" : 30,\n"
                        + "  \"active\" : true\n"
                        + "}",
                pretty);

        List<Object> arr = new ArrayList<>();
        arr.add(1);
        arr.add(2);
        arr.add(3);

        String compactArr = JSON.stringify(arr, false);
        assertEquals("[1,2,3]", compactArr);

        String prettyArr = JSON.stringify(arr, true);
        assertEquals("[\n" + "  1,\n" + "  2,\n" + "  3\n" + "]", prettyArr);
    }

    @Test
    void testNullHandling() {
        assertSame(JSON.NULL, JSON.parse("null"));
        assertEquals("null", JSON.stringify(JSON.NULL));

        Map<String, Object> obj = new LinkedHashMap<>();
        obj.put("key1", JSON.NULL);
        obj.put("key2", "value");
        String json = JSON.stringify(obj);
        assertEquals("{\"key1\":null,\"key2\":\"value\"}", json);

        Object parsed = JSON.parse(json);
        assertEquals(obj, parsed);
    }

    @Test
    void testWhitespace() {
        String json = "  {  \"key\"  :  \n  \"value\"  \t  }  ";
        Object parsed = JSON.parse(json);
        assertEquals(Map.of("key", "value"), parsed);

        json = "  [  1  ,  2  ,  3  ]  ";
        parsed = JSON.parse(json);
        assertEquals(List.of(1L, 2L, 3L), parsed);

        json = "  \"  hello  world  \"  ";
        parsed = JSON.parse(json);
        assertEquals("  hello  world  ", parsed);
    }

    @Test
    void testDuplicateKeys() {
        String json = "{\"key\":\"value1\",\"key\":\"value2\"}";
        Map<String, Object> obj = (Map<String, Object>) JSON.parse(json);
        assertEquals(1, obj.size());
        assertEquals("value2", obj.get("key"));
    }

    @Test
    void testNegativeZero() {
        // Test with BigDecimal mode
        Object parsed = JSON.parse("-0", JSON.ParseOptions.create().useBigDecimalForFloats());
        assertTrue(parsed instanceof BigDecimal);
        assertEquals(0, ((BigDecimal) parsed).compareTo(BigDecimal.ZERO));
        String stringified = JSON.stringify(parsed);
        assertEquals("0", stringified);

        parsed = JSON.parse("-0.0", JSON.ParseOptions.create().useBigDecimalForFloats());
        assertTrue(parsed instanceof BigDecimal);
        stringified = JSON.stringify(parsed);
        // stripTrailingZeros() removes .0
        assertEquals("0", stringified);
    }
}
