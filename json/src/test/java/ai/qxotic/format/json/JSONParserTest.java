package ai.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import java.util.*;
import org.junit.jupiter.api.Test;

class JSONParserTest {

    @Test
    void testNullLiteral() {
        assertSame(JSON.NULL, JSON.parse("null"));
    }

    @Test
    void testBooleanLiterals() {
        assertEquals(true, JSON.parse("true"));
        assertEquals(false, JSON.parse("false"));
    }

    @Test
    void testEmptyString() {
        assertEquals("", JSON.parse("\"\""));
    }

    @Test
    void testEmptyArray() {
        assertEquals(new ArrayList<>(), JSON.parse("[]"));
    }

    @Test
    void testEmptyObject() {
        assertEquals(Map.of(), JSON.parse("{}"));
    }

    @Test
    void testSimpleArray() {
        List<Object> expected = new ArrayList<>();
        expected.add(1L);
        expected.add("two");
        expected.add(true);
        expected.add(JSON.NULL);
        assertEquals(expected, JSON.parse("[1,\"two\",true,null]"));
    }

    @Test
    void testSimpleObject() {
        Map<String, Object> expected = new LinkedHashMap<>();
        expected.put("name", "John");
        expected.put("age", 30L);
        expected.put("active", true);
        assertEquals(expected, JSON.parse("{\"name\":\"John\",\"age\":30,\"active\":true}"));
    }

    @Test
    void testNestedArrays() {
        List<Object> inner1 = new ArrayList<>();
        inner1.add(1L);
        inner1.add(2L);
        List<Object> inner2 = new ArrayList<>();
        inner2.add(3L);
        inner2.add(4L);
        List<Object> expected = new ArrayList<>();
        expected.add(inner1);
        expected.add(inner2);
        assertEquals(expected, JSON.parse("[[1,2],[3,4]]"));
    }

    @Test
    void testNestedObjects() {
        Map<String, Object> inner = new LinkedHashMap<>();
        inner.put("value", 42L);
        Map<String, Object> expected = new LinkedHashMap<>();
        expected.put("outer", inner);
        assertEquals(expected, JSON.parse("{\"outer\":{\"value\":42}}"));
    }

    @Test
    void testMixedNesting() {
        List<Object> arr = new ArrayList<>();
        arr.add(1L);
        Map<String, Object> obj = new LinkedHashMap<>();
        obj.put("key", "value");
        arr.add(obj);
        arr.add(new ArrayList<>());
        assertEquals(arr, JSON.parse("[1,{\"key\":\"value\"},[]]"));
    }

    @Test
    void testWhitespaceBetweenTokens() {
        Map<String, Object> expected = new LinkedHashMap<>();
        expected.put("key", "value");
        assertEquals(expected, JSON.parse("  {  \"key\"  :  \"value\"  }  "));
    }

    @Test
    void testWhitespaceNewlinesAndTabs() {
        Map<String, Object> expected = new LinkedHashMap<>();
        expected.put("a", 1L);
        expected.put("b", 2L);
        String json = "{\n  \"a\": 1,\n  \"b\": 2\n}";
        assertEquals(expected, JSON.parse(json));
    }

    @Test
    void testDuplicateKeys() {
        Map<String, Object> result =
                (Map<String, Object>) JSON.parse("{\"key\":\"first\",\"key\":\"second\"}");
        assertEquals(1, result.size());
        assertEquals("second", result.get("key"));
    }

    @Test
    void testArrayTrailingCommaRejected() {
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("[1,2,]"));
        // Error message should indicate what was expected
        assertTrue(e.getMessage().contains("Expected") || e.getMessage().contains("Unexpected"));
    }

    @Test
    void testObjectTrailingCommaRejected() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("{\"a\":1,}"));
        assertTrue(e.getMessage().contains("Expected"));
    }

    @Test
    void testMissingColon() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("{\"key\" \"value\"}"));
        assertTrue(e.getMessage().contains("Expected"));
    }

    @Test
    void testMissingComma() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("{\"a\":1\"b\":2}"));
        assertTrue(e.getMessage().contains("Expected"));
    }

    @Test
    void testUnterminatedString() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("\"unclosed"));
        assertTrue(e.getMessage().contains("end of input"));
    }

    @Test
    void testUnterminatedObject() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("{\"a\":1"));
        assertTrue(
                e.getMessage().contains("end of input") || e.getMessage().contains("Expected '}'"));
    }

    @Test
    void testUnterminatedArray() {
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("[1,2"));
        assertTrue(
                e.getMessage().contains("end of input") || e.getMessage().contains("Expected ']'"));
    }

    @Test
    void testExtraContentAfterValidJson() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("123 extra"));
        assertTrue(e.getMessage().contains("end of input"));
    }

    @Test
    void testTrailingWhitespaceAllowed() {
        assertEquals(123L, JSON.parse("123   \t\n  "));
    }

    @Test
    void testSingleQuotedStringRejected() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("'single'"));
        assertTrue(e.getMessage().contains("Unexpected character"));
    }

    @Test
    void testSingleValue() {
        assertEquals(42L, JSON.parse("42"));
        assertEquals(3.14, ((Number) JSON.parse("3.14")).doubleValue());
        assertEquals(true, JSON.parse("true"));
        assertSame(JSON.NULL, JSON.parse("null"));
    }

    @Test
    void testParseObjectTyped() {
        Map<String, Object> obj = JSON.parseObject("{\"name\":\"John\",\"age\":30}");
        assertEquals("John", obj.get("name"));
        assertEquals(30L, obj.get("age"));
    }

    @Test
    void testParseArrayTyped() {
        List<Object> arr = JSON.parseArray("[1,\"two\",true]");
        assertEquals(3, arr.size());
        assertEquals(1L, arr.get(0));
        assertEquals("two", arr.get(1));
        assertEquals(true, arr.get(2));
    }

    @Test
    void testParseObjectTypedRejectsNonObjectRoot() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parseObject("[1,2,3]"));
        assertTrue(e.getMessage().contains("Expected JSON object at root"));
    }

    @Test
    void testParseArrayTypedRejectsNonArrayRoot() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parseArray("{\"a\":1}"));
        assertTrue(e.getMessage().contains("Expected JSON array at root"));
    }

    @Test
    void testParseStringTyped() {
        assertEquals("hello", JSON.parseString("\"hello\""));
    }

    @Test
    void testParseStringTypedRejectsNonStringRoot() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parseString("123"));
        assertTrue(e.getMessage().contains("Expected JSON string at root"));
    }

    @Test
    void testParseNumberTyped() {
        assertEquals(123L, JSON.parseNumber("123"));
        assertEquals(3.14, JSON.parseNumber("3.14").doubleValue());
    }

    @Test
    void testParseNumberTypedRejectsNonNumberRoot() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parseNumber("true"));
        assertTrue(e.getMessage().contains("Expected JSON number at root"));
    }

    @Test
    void testParseRejectsNullInput() {
        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> JSON.parse((CharSequence) null));
        assertTrue(e.getMessage().contains("json must not be null"));
    }

    @Test
    void testIsValid() {
        assertTrue(JSON.isValid("{\"a\":1}"));
        assertTrue(JSON.isValid("[1,2,3]"));
        assertFalse(JSON.isValid("{\"a\":}"));
        assertFalse(JSON.isValid(null));
    }
}
