package ai.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.*;
import org.junit.jupiter.api.Test;

class JSONRoundTripTest {

    private boolean deepEquals(Object o1, Object o2) {
        if (o1 == o2) return true;
        if (o1 == null || o2 == null) return false;

        if (o1 == JSON.NULL && o2 == JSON.NULL) return true;

        if (o1 instanceof Number && o2 instanceof Number) {
            Number n1 = (Number) o1;
            Number n2 = (Number) o2;
            if (n1 instanceof BigDecimal || n2 instanceof BigDecimal) {
                return new BigDecimal(n1.toString()).compareTo(new BigDecimal(n2.toString())) == 0;
            }
            if (n1 instanceof BigInteger || n2 instanceof BigInteger) {
                return new BigInteger(n1.toString()).equals(new BigInteger(n2.toString()));
            }
            return n1.doubleValue() == n2.doubleValue();
        }

        if (o1 instanceof List && o2 instanceof List) {
            List<?> l1 = (List<?>) o1;
            List<?> l2 = (List<?>) o2;
            if (l1.size() != l2.size()) return false;
            for (int i = 0; i < l1.size(); i++) {
                if (!deepEquals(l1.get(i), l2.get(i))) return false;
            }
            return true;
        }

        if (o1 instanceof Map && o2 instanceof Map) {
            Map<?, ?> m1 = (Map<?, ?>) o1;
            Map<?, ?> m2 = (Map<?, ?>) o2;
            if (m1.size() != m2.size()) return false;
            for (Map.Entry<?, ?> entry : m1.entrySet()) {
                Object value2 = m2.get(entry.getKey());
                if (!deepEquals(entry.getValue(), value2)) return false;
            }
            return true;
        }

        return o1.equals(o2);
    }

    @Test
    void testSimpleObjectRoundTrip() {
        Map<String, Object> original = new LinkedHashMap<>();
        original.put("name", "John");
        original.put("age", 30);
        original.put("active", true);

        String json = JSON.stringify(original, false);
        Object parsed = JSON.parse(json);
        assertTrue(deepEquals(original, parsed));
    }

    @Test
    void testSpecialNumbersRoundTrip() {
        List<Object> special = new ArrayList<>();
        special.add(BigDecimal.valueOf(0.5));
        special.add(BigDecimal.valueOf(-0.5));
        special.add(new BigDecimal("-0"));
        special.add(new BigInteger("123456789"));

        String json = JSON.stringify(special, false);
        List<?> parsed =
                (List<?>) JSON.parse(json, JSON.ParseOptions.create().useBigDecimalForFloats());
        assertTrue(deepEquals(special, parsed));
    }

    @Test
    void testNestedObjectRoundTrip() {
        Map<String, Object> inner = new LinkedHashMap<>();
        inner.put("value", 42);
        Map<String, Object> outer = new LinkedHashMap<>();
        outer.put("inner", inner);

        String json = JSON.stringify(outer, false);
        Object parsed = JSON.parse(json);
        assertTrue(deepEquals(outer, parsed));
    }

    @Test
    void testNegativeZeroRoundTrip() {
        String json = JSON.stringify(new BigDecimal("-0"), false);
        Object parsed = JSON.parse(json, JSON.ParseOptions.create().useBigDecimalForFloats());
        // BigDecimal("-0") prints as "0", which parses as Long(0) in default mode
        // Even with useBigDecimalForFloats(), integers return Long
        assertEquals(0L, parsed);
    }

    @Test
    void testNegativeZeroInStructure() {
        Map<String, Object> original = new LinkedHashMap<>();
        original.put("zero", new BigDecimal("-0"));

        String json = JSON.stringify(original, false);
        Object parsed = JSON.parse(json, JSON.ParseOptions.create().useBigDecimalForFloats());
        assertTrue(deepEquals(original, parsed));
    }

    @Test
    void testLargeNumberRoundTrip() {
        BigInteger large = new BigInteger("9999999999999999999999999999999999999999");
        String json = JSON.stringify(large, false);
        Object parsed = JSON.parse(json);
        assertEquals(large, parsed);
    }

    @Test
    void testDecimalRoundTrip() {
        BigDecimal bd = new BigDecimal("123.456789");
        String json = JSON.stringify(bd, false);
        Object parsed = JSON.parse(json, JSON.ParseOptions.create().useBigDecimalForFloats());
        assertEquals(bd, parsed);
    }

    @Test
    void testUnicodeRoundTrip() {
        Map<String, Object> original = new LinkedHashMap<>();
        original.put("chinese", "中文");
        original.put("emoji", "\uD83D\uDE00");

        String json = JSON.stringify(original, false);
        Object parsed = JSON.parse(json);
        assertTrue(deepEquals(original, parsed));
    }

    @Test
    void testEscapedCharactersRoundTrip() {
        Map<String, Object> original = new LinkedHashMap<>();
        original.put("quotes", "\"quoted\"");
        original.put("backslash", "\\");
        original.put("newline", "\n");

        String json = JSON.stringify(original, false);
        Object parsed = JSON.parse(json);
        assertTrue(deepEquals(original, parsed));
    }

    @Test
    void testNullValueRoundTrip() {
        Map<String, Object> original = new LinkedHashMap<>();
        original.put("null", JSON.NULL);
        original.put("string", "value");

        String json = JSON.stringify(original, false);
        Object parsed = JSON.parse(json);
        assertTrue(deepEquals(original, parsed));
    }

    @Test
    void testComplexNestedRoundTrip() {
        Map<String, Object> obj = new LinkedHashMap<>();
        obj.put("array", new ArrayList<>(Arrays.asList(1, 2, 3)));
        obj.put("nested", new LinkedHashMap<>(Map.of("a", 1, "b", 2)));

        String json = JSON.stringify(obj, false);
        Object parsed = JSON.parse(json);
        assertTrue(deepEquals(obj, parsed));
    }

    @Test
    void testPrettyPrintRoundTrip() {
        Map<String, Object> original = new LinkedHashMap<>();
        original.put("key", "value");

        String json = JSON.stringify(original, true);
        Object parsed = JSON.parse(json);
        assertTrue(deepEquals(original, parsed));
    }

    @Test
    void testPrettyPrintArrayRoundTrip() {
        List<Object> original = Arrays.asList(1, 2, 3);
        String json = JSON.stringify(original, true);
        Object parsed = JSON.parse(json);
        assertTrue(deepEquals(original, parsed));
    }

    @Test
    void testEmptyStructuresRoundTrip() {
        Map<String, Object> original = new LinkedHashMap<>();
        original.put("array", new ArrayList<>());
        original.put("object", new LinkedHashMap<>());

        String json = JSON.stringify(original, false);
        Object parsed = JSON.parse(json);
        assertTrue(deepEquals(original, parsed));
    }

    @Test
    void testDeepNestingRoundTrip() {
        Object current = 1;
        for (int i = 0; i < 10; i++) {
            Map<String, Object> next = new LinkedHashMap<>();
            next.put("level", i);
            next.put("value", current);
            current = next;
        }

        String json = JSON.stringify(current, false);
        Object parsed = JSON.parse(json);
        assertTrue(deepEquals(current, parsed));
    }

    @Test
    void testSpecialNumberValues() {
        List<Number> special = new ArrayList<>();
        special.add(0L);
        special.add(1L);
        special.add(-1L);
        special.add(BigDecimal.valueOf(0.5));
        special.add(BigDecimal.valueOf(-0.5));
        special.add(new BigDecimal("-0"));
        special.add(new BigInteger("123456789"));

        String json = JSON.stringify(special, false);
        Object parsed = JSON.parse(json);
        assertTrue(deepEquals(special, parsed));
    }

    @Test
    void testMixedTypesRoundTrip() {
        Map<String, Object> original = new LinkedHashMap<>();
        original.put("string", "hello");
        original.put("number", 42);
        original.put("decimal", new BigDecimal("3.14"));
        String json = JSON.stringify(original, false);
        Map<?, ?> parsed =
                (Map<?, ?>) JSON.parse(json, JSON.ParseOptions.create().useBigDecimalForFloats());
        assertTrue(deepEquals(original, parsed));
    }

    @Test
    void testPrettyPrintNestedRoundTrip() {
        Map<String, Object> inner = new LinkedHashMap<>();
        inner.put("value", 42);
        Map<String, Object> outer = new LinkedHashMap<>();
        outer.put("inner", inner);
        outer.put("array", new ArrayList<>(Arrays.asList(1, 2, 3)));

        String json = JSON.stringify(outer, true);
        Object parsed = JSON.parse(json);
        assertTrue(deepEquals(outer, parsed));
    }

    @Test
    void testChineseCharactersRoundTrip() {
        String chinese = "中文繁體日本語한국어";
        Map<String, Object> obj = new LinkedHashMap<>();
        obj.put("text", chinese);

        String json = JSON.stringify(obj, false);
        Object parsed = JSON.parse(json);
        assertTrue(deepEquals(obj, parsed));
    }
}
