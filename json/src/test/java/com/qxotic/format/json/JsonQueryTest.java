package com.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** Tests for query API and type check methods. */
class JsonQueryTest {

    // ============== Query String Tests ==============

    @Test
    @DisplayName("queryString: simple key access")
    void testQueryStringSimple() {
        Map<String, Object> data = Json.parseMap("{\"name\": \"Alice\"}");
        Optional<String> result = Json.queryString(data, "name");
        assertTrue(result.isPresent());
        assertEquals("Alice", result.get());
    }

    @Test
    @DisplayName("queryString: nested key access")
    void testQueryStringNested() {
        Map<String, Object> data = Json.parseMap("{\"user\": {\"profile\": {\"name\": \"Bob\"}}}");
        Optional<String> result = Json.queryString(data, "user", "profile", "name");
        assertTrue(result.isPresent());
        assertEquals("Bob", result.get());
    }

    @Test
    @DisplayName("queryString: missing key returns empty")
    void testQueryStringMissing() {
        Map<String, Object> data = Json.parseMap("{\"name\": \"Alice\"}");
        Optional<String> result = Json.queryString(data, "nonexistent");
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("queryString: type mismatch returns empty")
    void testQueryStringTypeMismatch() {
        Map<String, Object> data = Json.parseMap("{\"age\": 30}");
        Optional<String> result = Json.queryString(data, "age");
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("queryString: intermediate not map returns empty")
    void testQueryStringIntermediateNotMap() {
        Map<String, Object> data = Json.parseMap("{\"name\": \"Alice\"}");
        Optional<String> result = Json.queryString(data, "name", "invalid");
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("queryString: with default value")
    void testQueryStringWithDefault() {
        Map<String, Object> data = Json.parseMap("{\"name\": \"Alice\"}");
        String result = Json.queryString(data, "nonexistent").orElse("Default");
        assertEquals("Default", result);
    }

    @Test
    @DisplayName("queryString: empty string value")
    void testQueryStringEmpty() {
        Map<String, Object> data = Json.parseMap("{\"name\": \"\"}");
        Optional<String> result = Json.queryString(data, "name");
        assertTrue(result.isPresent());
        assertEquals("", result.get());
    }

    @Test
    @DisplayName("queryString: unicode string")
    void testQueryStringUnicode() {
        Map<String, Object> data = Json.parseMap("{\"name\": \"日本語\"}");
        Optional<String> result = Json.queryString(data, "name");
        assertTrue(result.isPresent());
        assertEquals("日本語", result.get());
    }

    // ============== Query Map Tests ==============

    @Test
    @DisplayName("queryMap: simple map access")
    void testQueryMapSimple() {
        Map<String, Object> data = Json.parseMap("{\"user\": {\"name\": \"Alice\"}}");
        Optional<Map<String, Object>> result = Json.queryMap(data, "user");
        assertTrue(result.isPresent());
        assertEquals("Alice", result.get().get("name"));
    }

    @Test
    @DisplayName("queryMap: nested map access")
    void testQueryMapNested() {
        Map<String, Object> data = Json.parseMap("{\"a\": {\"b\": {\"c\": 1}}}");
        Optional<Map<String, Object>> result = Json.queryMap(data, "a", "b");
        assertTrue(result.isPresent());
        assertEquals(1L, result.get().get("c"));
    }

    @Test
    @DisplayName("queryMap: missing key returns empty")
    void testQueryMapMissing() {
        Map<String, Object> data = Json.parseMap("{\"user\": {}}");
        Optional<Map<String, Object>> result = Json.queryMap(data, "nonexistent");
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("queryMap: type mismatch (string instead of map)")
    void testQueryMapTypeMismatch() {
        Map<String, Object> data = Json.parseMap("{\"user\": \"not a map\"}");
        Optional<Map<String, Object>> result = Json.queryMap(data, "user");
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("queryMap: empty map")
    void testQueryMapEmpty() {
        Map<String, Object> data = Json.parseMap("{\"user\": {}}");
        Optional<Map<String, Object>> result = Json.queryMap(data, "user");
        assertTrue(result.isPresent());
        assertTrue(result.get().isEmpty());
    }

    // ============== Query List Tests ==============

    @Test
    @DisplayName("queryList: simple list access")
    void testQueryListSimple() {
        Map<String, Object> data = Json.parseMap("{\"items\": [1, 2, 3]}");
        Optional<List<Object>> result = Json.queryList(data, "items");
        assertTrue(result.isPresent());
        assertEquals(3, result.get().size());
    }

    @Test
    @DisplayName("queryList: nested list access")
    void testQueryListNested() {
        Map<String, Object> data = Json.parseMap("{\"data\": {\"items\": [\"a\", \"b\"]}}");
        Optional<List<Object>> result = Json.queryList(data, "data", "items");
        assertTrue(result.isPresent());
        assertEquals(2, result.get().size());
    }

    @Test
    @DisplayName("queryList: missing key returns empty")
    void testQueryListMissing() {
        Map<String, Object> data = Json.parseMap("{\"items\": []}");
        Optional<List<Object>> result = Json.queryList(data, "nonexistent");
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("queryList: type mismatch (string instead of list)")
    void testQueryListTypeMismatch() {
        Map<String, Object> data = Json.parseMap("{\"items\": \"not a list\"}");
        Optional<List<Object>> result = Json.queryList(data, "items");
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("queryList: empty list")
    void testQueryListEmpty() {
        Map<String, Object> data = Json.parseMap("{\"items\": []}");
        Optional<List<Object>> result = Json.queryList(data, "items");
        assertTrue(result.isPresent());
        assertTrue(result.get().isEmpty());
    }

    @Test
    @DisplayName("queryList: list of objects")
    void testQueryListOfObjects() {
        Map<String, Object> data =
                Json.parseMap("{\"users\": [{\"name\": \"Alice\"}, {\"name\": \"Bob\"}]}");
        Optional<List<Object>> result = Json.queryList(data, "users");
        assertTrue(result.isPresent());
        assertEquals(2, result.get().size());
    }

    // ============== Query Boolean Tests ==============

    @Test
    @DisplayName("queryBoolean: true value")
    void testQueryBooleanTrue() {
        Map<String, Object> data = Json.parseMap("{\"active\": true}");
        Optional<Boolean> result = Json.queryBoolean(data, "active");
        assertTrue(result.isPresent());
        assertTrue(result.get());
    }

    @Test
    @DisplayName("queryBoolean: false value")
    void testQueryBooleanFalse() {
        Map<String, Object> data = Json.parseMap("{\"active\": false}");
        Optional<Boolean> result = Json.queryBoolean(data, "active");
        assertTrue(result.isPresent());
        assertFalse(result.get());
    }

    @Test
    @DisplayName("queryBoolean: missing key returns empty")
    void testQueryBooleanMissing() {
        Map<String, Object> data = Json.parseMap("{\"active\": true}");
        Optional<Boolean> result = Json.queryBoolean(data, "nonexistent");
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("queryBoolean: type mismatch (string instead of boolean)")
    void testQueryBooleanTypeMismatch() {
        Map<String, Object> data = Json.parseMap("{\"active\": \"yes\"}");
        Optional<Boolean> result = Json.queryBoolean(data, "active");
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("queryBoolean: nested access")
    void testQueryBooleanNested() {
        Map<String, Object> data =
                Json.parseMap("{\"user\": {\"settings\": {\"notifications\": true}}}");
        Optional<Boolean> result = Json.queryBoolean(data, "user", "settings", "notifications");
        assertTrue(result.isPresent());
        assertTrue(result.get());
    }

    // ============== Query Number Tests ==============

    @Test
    @DisplayName("queryNumber: integer value")
    void testQueryNumberInteger() {
        Map<String, Object> data = Json.parseMap("{\"age\": 30}");
        Optional<Number> result = Json.queryNumber(data, "age");
        assertTrue(result.isPresent());
        assertEquals(30, result.get().intValue());
    }

    @Test
    @DisplayName("queryNumber: decimal value")
    void testQueryNumberDecimal() {
        Map<String, Object> data = Json.parseMap("{\"price\": 19.99}");
        Optional<Number> result = Json.queryNumber(data, "price");
        assertTrue(result.isPresent());
        assertEquals(19.99, result.get().doubleValue(), 0.001);
    }

    @Test
    @DisplayName("queryNumber: negative value")
    void testQueryNumberNegative() {
        Map<String, Object> data = Json.parseMap("{\"temperature\": -5}");
        Optional<Number> result = Json.queryNumber(data, "temperature");
        assertTrue(result.isPresent());
        assertEquals(-5, result.get().intValue());
    }

    @Test
    @DisplayName("queryNumber: zero value")
    void testQueryNumberZero() {
        Map<String, Object> data = Json.parseMap("{\"count\": 0}");
        Optional<Number> result = Json.queryNumber(data, "count");
        assertTrue(result.isPresent());
        assertEquals(0, result.get().intValue());
    }

    @Test
    @DisplayName("queryNumber: missing key returns empty")
    void testQueryNumberMissing() {
        Map<String, Object> data = Json.parseMap("{\"age\": 30}");
        Optional<Number> result = Json.queryNumber(data, "nonexistent");
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("queryNumber: type mismatch (string instead of number)")
    void testQueryNumberTypeMismatch() {
        Map<String, Object> data = Json.parseMap("{\"age\": \"thirty\"}");
        Optional<Number> result = Json.queryNumber(data, "age");
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("queryNumber: nested access")
    void testQueryNumberNested() {
        Map<String, Object> data = Json.parseMap("{\"product\": {\"details\": {\"stock\": 100}}}");
        Optional<Number> result = Json.queryNumber(data, "product", "details", "stock");
        assertTrue(result.isPresent());
        assertEquals(100, result.get().intValue());
    }

    @Test
    @DisplayName("queryNumber: scientific notation")
    void testQueryNumberScientific() {
        Map<String, Object> data = Json.parseMap("{\"value\": 1.5e10}");
        Optional<Number> result = Json.queryNumber(data, "value");
        assertTrue(result.isPresent());
        assertEquals(1.5e10, result.get().doubleValue(), 0.001);
    }

    // ============== Query (Raw) Tests ==============

    @Test
    @DisplayName("query: returns String")
    void testQueryString() {
        Map<String, Object> data = Json.parseMap("{\"name\": \"Alice\"}");
        Optional<Object> result = Json.query(data, "name");
        assertTrue(result.isPresent());
        assertInstanceOf(String.class, result.get());
        assertEquals("Alice", result.get());
    }

    @Test
    @DisplayName("query: returns Map")
    void testQueryMap() {
        Map<String, Object> data = Json.parseMap("{\"user\": {\"name\": \"Alice\"}}");
        Optional<Object> result = Json.query(data, "user");
        assertTrue(result.isPresent());
        assertInstanceOf(Map.class, result.get());
    }

    @Test
    @DisplayName("query: returns List")
    void testQueryList() {
        Map<String, Object> data = Json.parseMap("{\"items\": [1, 2, 3]}");
        Optional<Object> result = Json.query(data, "items");
        assertTrue(result.isPresent());
        assertInstanceOf(List.class, result.get());
    }

    @Test
    @DisplayName("query: returns Json.NULL for explicit null")
    void testQueryExplicitNull() {
        Map<String, Object> data = Json.parseMap("{\"value\": null}");
        Optional<Object> result = Json.query(data, "value");
        assertTrue(result.isPresent());
        assertSame(Json.NULL, result.get());
    }

    @Test
    @DisplayName("query: returns empty for missing key")
    void testQueryMissing() {
        Map<String, Object> data = Json.parseMap("{\"name\": \"Alice\"}");
        Optional<Object> result = Json.query(data, "nonexistent");
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("query: nested access")
    void testQueryNested() {
        Map<String, Object> data = Json.parseMap("{\"a\": {\"b\": {\"c\": 1}}}");
        Optional<Object> result = Json.query(data, "a", "b", "c");
        assertTrue(result.isPresent());
        assertEquals(1L, result.get());
    }

    // ============== Zero Keys (Cast) Tests ==============

    @Test
    @DisplayName("queryString: 0 keys casts root to String")
    void testQueryStringZeroKeys() {
        Object root = "test string";
        Optional<String> result = Json.queryString(root);
        assertTrue(result.isPresent());
        assertEquals("test string", result.get());
    }

    @Test
    @DisplayName("queryMap: 0 keys casts root to Map")
    void testQueryMapZeroKeys() {
        Map<String, Object> root = Map.of("key", "value");
        Optional<Map<String, Object>> result = Json.queryMap(root);
        assertTrue(result.isPresent());
        assertEquals("value", result.get().get("key"));
    }

    @Test
    @DisplayName("queryList: 0 keys casts root to List")
    void testQueryListZeroKeys() {
        List<Object> root = List.of(1, 2, 3);
        Optional<List<Object>> result = Json.queryList(root);
        assertTrue(result.isPresent());
        assertEquals(3, result.get().size());
    }

    @Test
    @DisplayName("queryBoolean: 0 keys casts root to Boolean")
    void testQueryBooleanZeroKeys() {
        Object root = true;
        Optional<Boolean> result = Json.queryBoolean(root);
        assertTrue(result.isPresent());
        assertTrue(result.get());
    }

    @Test
    @DisplayName("queryNumber: 0 keys casts root to Number")
    void testQueryNumberZeroKeys() {
        Object root = 42;
        Optional<Number> result = Json.queryNumber(root);
        assertTrue(result.isPresent());
        assertEquals(42, result.get().intValue());
    }

    @Test
    @DisplayName("query: 0 keys returns root as-is")
    void testQueryZeroKeys() {
        Map<String, Object> root = Map.of("key", "value");
        Optional<Object> result = Json.query(root);
        assertTrue(result.isPresent());
        assertSame(root, result.get());
    }

    @Test
    @DisplayName("queryString: 0 keys with wrong type returns empty")
    void testQueryStringZeroKeysWrongType() {
        Object root = 42;
        Optional<String> result = Json.queryString(root);
        assertTrue(result.isEmpty());
    }

    // ============== Renamed Parsing Method Tests ==============

    @Test
    @DisplayName("parseMap: parses JSON object")
    void testParseMap() {
        Map<String, Object> result = Json.parseMap("{\"name\": \"Alice\", \"age\": 30}");
        assertNotNull(result);
        assertEquals("Alice", result.get("name"));
        assertEquals(30L, result.get("age"));
    }

    @Test
    @DisplayName("parseMap: parses empty object")
    void testParseMapEmpty() {
        Map<String, Object> result = Json.parseMap("{}");
        assertNotNull(result);
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("parseMap: with options")
    void testParseMapWithOptions() {
        Map<String, Object> result =
                Json.parseMap("{\"value\": 1.5}", Json.ParseOptions.defaults());
        assertNotNull(result);
        assertInstanceOf(Number.class, result.get("value"));
    }

    @Test
    @DisplayName("parseList: parses JSON array")
    void testParseList() {
        List<Object> result = Json.parseList("[1, 2, 3]");
        assertNotNull(result);
        assertEquals(3, result.size());
        assertEquals(1L, result.get(0));
        assertEquals(2L, result.get(1));
        assertEquals(3L, result.get(2));
    }

    @Test
    @DisplayName("parseList: parses empty array")
    void testParseListEmpty() {
        List<Object> result = Json.parseList("[]");
        assertNotNull(result);
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("parseList: parses array of objects")
    void testParseListOfObjects() {
        List<Object> result = Json.parseList("[{\"name\": \"Alice\"}, {\"name\": \"Bob\"}]");
        assertNotNull(result);
        assertEquals(2, result.size());
        assertInstanceOf(Map.class, result.get(0));
        assertEquals("Alice", ((Map<?, ?>) result.get(0)).get("name"));
    }

    @Test
    @DisplayName("parseList: with options")
    void testParseListWithOptions() {
        List<Object> result = Json.parseList("[1, 2, 3]", Json.ParseOptions.defaults());
        assertNotNull(result);
        assertEquals(3, result.size());
    }

    // ============== Deep Nesting Tests ==============

    @Test
    @DisplayName("query: very deep nesting (10 levels)")
    void testDeepNesting() {
        StringBuilder json = new StringBuilder("{");
        for (int i = 0; i < 10; i++) {
            json.append("\"level").append(i).append("\": {");
        }
        json.append("\"value\": \"found\"");
        json.append("}".repeat(10));
        json.append("}");

        Map<String, Object> data = Json.parseMap(json.toString());
        String[] keys = new String[11];
        for (int i = 0; i < 10; i++) {
            keys[i] = "level" + i;
        }
        keys[10] = "value";

        Optional<String> result = Json.queryString(data, keys);
        assertTrue(result.isPresent());
        assertEquals("found", result.get());
    }

    // ============== Complex JSON Tests ==============

    @Test
    @DisplayName("query: complex nested structure")
    void testComplexStructure() {
        String json =
                "{\"company\": {    \"name\": \"Acme Corp\",    \"employees\": [        {\"id\": 1,"
                    + " \"name\": \"Alice\", \"active\": true, \"salary\": 50000.50},       "
                    + " {\"id\": 2, \"name\": \"Bob\", \"active\": false, \"salary\": 60000.00}   "
                    + " ],    \"metadata\": null,    \"settings\": {\"debug\": true, \"version\":"
                    + " 1.5}}}";

        Map<String, Object> data = Json.parseMap(json);

        // Query various types
        Optional<String> companyName = Json.queryString(data, "company", "name");
        assertTrue(companyName.isPresent());
        assertEquals("Acme Corp", companyName.get());

        Optional<List<Object>> employees = Json.queryList(data, "company", "employees");
        assertTrue(employees.isPresent());
        assertEquals(2, employees.get().size());

        Optional<Map<String, Object>> settings = Json.queryMap(data, "company", "settings");
        assertTrue(settings.isPresent());

        Optional<Object> metadata = Json.query(data, "company", "metadata");
        assertTrue(metadata.isPresent());
        assertSame(Json.NULL, metadata.get());
    }

    // ============== Error Handling Tests ==============

    @Test
    @DisplayName("query: null root throws NullPointerException")
    void testNullRoot() {
        assertThrows(
                NullPointerException.class,
                () -> {
                    Json.queryString(null, "key");
                });
    }

    @Test
    @DisplayName("query: null key throws NullPointerException")
    void testNullKey() {
        Map<String, Object> data = Map.of("key", "value");
        assertThrows(
                NullPointerException.class,
                () -> {
                    Json.queryString(data, (String) null);
                });
    }

    @Test
    @DisplayName("query: null in varargs throws NullPointerException")
    void testNullInVarargs() {
        Map<String, Object> data = Json.parseMap("{\"a\": {\"b\": 1}}");
        assertThrows(
                NullPointerException.class,
                () -> {
                    Json.queryString(data, "a", null, "b");
                });
    }

    // ============== Integration with Existing API Tests ==============

    @Test
    @DisplayName("query works with parseMap")
    void testQueryWithParseMap() {
        Map<String, Object> data = Json.parseMap("{\"name\": \"Alice\"}");
        Optional<String> result = Json.queryString(data, "name");
        assertTrue(result.isPresent());
        assertEquals("Alice", result.get());
    }

    @Test
    @DisplayName("query works with parseList")
    void testQueryWithParseList() {
        List<Object> data = Json.parseList("[{\"name\": \"Alice\"}]");
        assertEquals(1, data.size());
        assertInstanceOf(Map.class, data.get(0));
    }

    // ============== Edge Case Tests ==============

    @Test
    @DisplayName("query: key with special characters in value")
    void testKeyWithSpecialCharacters() {
        Map<String, Object> data = Json.parseMap("{\"message\": \"Hello\\nWorld\\t!\"}");
        Optional<String> result = Json.queryString(data, "message");
        assertTrue(result.isPresent());
        assertEquals("Hello\nWorld\t!", result.get());
    }

    @Test
    @DisplayName("query: very long string value")
    void testVeryLongString() {
        String longValue = "a".repeat(10000);
        Map<String, Object> data = Json.parseMap("{\"data\": \"" + longValue + "\"}");
        Optional<String> result = Json.queryString(data, "data");
        assertTrue(result.isPresent());
        assertEquals(longValue, result.get());
    }

    @Test
    @DisplayName("query: large number")
    void testLargeNumber() {
        Map<String, Object> data =
                Json.parseMap("{\"big\": 9223372036854775807}"); // Long.MAX_VALUE
        Optional<Number> result = Json.queryNumber(data, "big");
        assertTrue(result.isPresent());
        assertEquals(9223372036854775807L, result.get().longValue());
    }

    @Test
    @DisplayName("query: multiple queries on same data")
    void testMultipleQueries() {
        Map<String, Object> data = Json.parseMap("{\"a\": 1, \"b\": 2, \"c\": 3}");

        Optional<Number> a = Json.queryNumber(data, "a");
        Optional<Number> b = Json.queryNumber(data, "b");
        Optional<Number> c = Json.queryNumber(data, "c");

        assertTrue(a.isPresent() && a.get().intValue() == 1);
        assertTrue(b.isPresent() && b.get().intValue() == 2);
        assertTrue(c.isPresent() && c.get().intValue() == 3);
    }

    @Test
    @DisplayName("query: chain with filter and map")
    void testQueryChain() {
        Map<String, Object> data = Json.parseMap("{\"user\": {\"age\": 25}}");

        boolean isAdult =
                Json.queryNumber(data, "user", "age")
                        .filter(age -> age.intValue() >= 18)
                        .isPresent();

        assertTrue(isAdult);
    }

    @Test
    @DisplayName("query: functional composition")
    void testFunctionalComposition() {
        Map<String, Object> data =
                Json.parseMap("{\"users\": [{\"name\": \"Alice\"}, {\"name\": \"Bob\"}]}");

        List<String> names =
                Json.queryList(data, "users")
                        .map(
                                list -> {
                                    List<String> result = new ArrayList<>();
                                    for (Object obj : list) {
                                        if (obj instanceof Map) {
                                            Optional<String> name =
                                                    Json.queryString(
                                                            (Map<String, Object>) obj, "name");
                                            name.ifPresent(result::add);
                                        }
                                    }
                                    return result;
                                })
                        .orElse(List.of());

        assertEquals(List.of("Alice", "Bob"), names);
    }
}
