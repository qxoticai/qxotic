package com.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** Tests for Json parsing and serialization. */
class JsonCompleteTest {

    // ===== Null.toString() coverage =====

    @Test
    void testNullToString() {
        assertEquals("null", Json.NULL.toString());
    }

    // ===== isValid() with valid and invalid JSON =====

    @Test
    void testIsValidWithValidJson() {
        assertTrue(Json.isValid("{}"));
        assertTrue(Json.isValid("[]"));
        assertTrue(Json.isValid("true"));
        assertTrue(Json.isValid("false"));
        assertTrue(Json.isValid("null"));
        assertTrue(Json.isValid("123"));
        assertTrue(Json.isValid("\"string\""));
    }

    @Test
    void testIsValidWithInvalidJson() {
        assertFalse(Json.isValid("{"));
        assertFalse(Json.isValid("}"));
        assertFalse(Json.isValid(""));
        assertFalse(Json.isValid("undefined"));
    }

    @Test
    void testIsValidWithOptions() {
        Json.ParseOptions options = Json.ParseOptions.defaults().maxDepth(5);
        assertTrue(Json.isValid("[[[[[]]]]]", options));
        assertFalse(Json.isValid("[[[[[[]]]]]]", options)); // Depth 6 > 5
    }

    // ===== skipValue() uncovered branches =====

    @Test
    void testSkipValueWithClosingBracket() {
        // This triggers the ']' case in skipValue switch
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("]"));
        assertTrue(
                e.getMessage().contains("Expected value") || e.getMessage().contains("Unexpected"));
    }

    @Test
    void testSkipValueWithUnexpectedCharacter() {
        // This triggers the default case with non-digit, non-minus character
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("@"));
        assertTrue(e.getMessage().contains("Unexpected character"));
    }

    // ===== skipStringSimple() uncovered lines =====

    @Test
    void testSkipStringSimpleUnterminated() {
        // String that ends without closing quote
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"unterminated"));
        assertTrue(e.getMessage().contains("end of input"));
    }

    @Test
    void testSkipStringSimpleWithControlCharacter() {
        // String with unescaped control character (use actual control char in string)
        String jsonWithControl = "\"test" + (char) 0x01 + "value\"";
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse(jsonWithControl));
        assertTrue(e.getMessage().contains("Control") || e.getMessage().contains("control"));
    }

    // ===== skipEscapeSequence() uncovered branches =====

    @Test
    void testInvalidEscapeSequence() {
        // Invalid escape character
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\z\""));
        assertTrue(e.getMessage().contains("Invalid escape"));
    }

    @Test
    void testEscapeSequenceWithUnicode() {
        // Valid unicode escape
        assertEquals("A", Json.parse("\"\\u0041\""));
    }

    // ===== scanNumber() '+' sign error =====

    @Test
    void testNumberWithPlusSign() {
        // JSON does not allow + sign before numbers
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("+123"));
        assertTrue(
                e.getMessage().contains("Unexpected '+'")
                        || e.getMessage().contains("Unexpected character"));
    }

    // ===== validateUnicodeEscape() surrogate branches =====

    @Test
    void testUnicodeEscapeIncompleteSurrogate() {
        // High surrogate without low surrogate
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\uD83D\""));
        assertTrue(e.getMessage().contains("Lone surrogate"));
    }

    @Test
    void testUnicodeEscapeInvalidSurrogatePair() {
        // High surrogate followed by something that's not a low surrogate
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\uD83D\\u0041\""));
        assertTrue(
                e.getMessage().contains("surrogate")
                        || e.getMessage().contains("Unexpected character"));
    }

    @Test
    void testUnicodeEscapeLoneLowSurrogate() {
        // Low surrogate without high surrogate
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\uDE00\""));
        assertTrue(e.getMessage().contains("Lone surrogate"));
    }

    // ===== escapeFor() control characters =====

    @Test
    void testStringifyWithBackspace() {
        assertEquals("\"\\b\"", Json.stringify("\b", false));
    }

    @Test
    void testStringifyWithFormFeed() {
        assertEquals("\"\\f\"", Json.stringify("\f", false));
    }

    @Test
    void testStringifyWithCarriageReturn() {
        assertEquals("\"\\r\"", Json.stringify("\r", false));
    }

    @Test
    void testStringifyWithControlCharacters() {
        // Control characters (0x00-0x1F) need unicode escape per RFC 8259
        assertEquals("\"\\u0000\"", Json.stringify("\u0000", false));
        assertEquals("\"\\u001F\"", Json.stringify("\u001f", false)); // uppercase hex
        // DEL (0x7F) is NOT a control character per RFC 8259; not escaped
        assertEquals("\"\u007f\"", Json.stringify("\u007f", false));
    }

    // ===== print() unsupported type error =====

    @Test
    void testStringifyUnsupportedType() {
        class CustomClass {
            @Override
            public String toString() {
                return "custom";
            }
        }

        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Json.stringify(new CustomClass(), false));
        assertTrue(e.getMessage().contains("Cannot serialize"));
    }

    // ===== parseTyped methods with wrong root type =====

    @Test
    void testParseMapWithArrayRoot() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parseMap("[]"));
        assertTrue(e.getMessage().contains("Expected JSON object"));
    }

    @Test
    void testParseListWithObjectRoot() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parseList("{}"));
        assertTrue(e.getMessage().contains("Expected JSON array"));
    }

    @Test
    void testParseStringWithNumberRoot() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parseString("123"));
        assertTrue(e.getMessage().contains("Expected JSON string"));
    }

    @Test
    void testParseNumberWithStringRoot() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parseNumber("\"123\""));
        assertTrue(e.getMessage().contains("Expected JSON number"));
    }

    @Test
    void testParseBooleanWithStringRoot() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parseBoolean("\"true\""));
        assertTrue(e.getMessage().contains("Expected JSON boolean"));
    }

    // ===== Additional edge cases for coverage =====

    @Test
    void testEmptyInput() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse(""));
        assertTrue(
                e.getMessage().contains("Unexpected end")
                        || e.getMessage().contains("Unexpected character"));
    }

    @Test
    void testWhitespaceOnlyInput() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("   \n\t  "));
        assertTrue(
                e.getMessage().contains("Unexpected end")
                        || e.getMessage().contains("Unexpected character"));
    }

    @Test
    void testTrailingContent() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("{} {}"));
        assertTrue(e.getMessage().contains("Expected end of input"));
    }

    @Test
    void testArrayTrailingComma() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("[1, 2,]"));
        assertTrue(e.getMessage().contains("Expected value"));
    }

    @Test
    void testObjectTrailingComma() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("{\"a\": 1,}"));
        assertTrue(e.getMessage().contains("Expected"));
    }

    @Test
    void testNumberStartingWithDot() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse(".123"));
        assertTrue(e.getMessage().contains("Unexpected character"));
    }

    @Test
    void testNumberWithLeadingZeros() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("0123"));
        assertTrue(e.getMessage().contains("Leading zeros"));
    }

    @Test
    void testNumberWithEmptyExponent() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("1e"));
        assertTrue(
                e.getMessage().contains("Exponent missing digits")
                        || e.getMessage().contains("Unexpected end"));
    }

    @Test
    void testNumberWithDecimalPointOnly() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("1."));
        assertTrue(e.getMessage().contains("Expected digit after decimal"));
    }

    @Test
    void testUnicodeEscapeInvalidHex() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\u00GH\""));
        assertTrue(e.getMessage().contains("Invalid hex digit"));
    }

    @Test
    void testUnicodeEscapeIncomplete() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\u00\""));
        assertTrue(
                e.getMessage().contains("Incomplete Unicode escape")
                        || e.getMessage().contains("Unexpected end"));
    }

    @Test
    void testNestedObjectWithDuplicateKeys() {
        Json.ParseOptions options = Json.ParseOptions.defaults().failOnDuplicateKeys(true);
        Json.ParseException e =
                assertThrows(
                        Json.ParseException.class,
                        () -> Json.parse("{\"a\": 1, \"a\": 2}", options));
        assertTrue(e.getMessage().contains("Duplicate key"));
    }

    @Test
    void testDeeplyNestedArray() {
        // Test that depth limit is enforced
        Json.ParseOptions options = Json.ParseOptions.defaults().maxDepth(5);
        StringBuilder sb = new StringBuilder();
        sb.append("[".repeat(6));
        sb.append("null");
        sb.append("]".repeat(6));

        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse(sb.toString(), options));
        assertTrue(e.getMessage().contains("Maximum parsing depth exceeded"));
    }

    @Test
    void testPrettyPrintWithDeepIndent() {
        // Test indent cache fallback for deep nesting
        Map<String, Object> nested = new HashMap<>();
        Map<String, Object> current = nested;

        // Create deeply nested structure
        for (int i = 0; i < 20; i++) {
            Map<String, Object> next = new HashMap<>();
            current.put("level" + i, next);
            current = next;
        }
        current.put("value", "deep");

        String pretty = Json.stringify(nested, true);
        assertTrue(pretty.contains("\n"));
        // Verify it's valid JSON
        Object parsed = Json.parse(pretty);
        assertNotNull(parsed);
    }

    @Test
    void testParseOptionsConfiguration() {
        // ParseOptions doesn't implement equals/hashCode - each instance is unique
        // Test configuration changes instead
        Json.ParseOptions opts1 = Json.ParseOptions.defaults();
        assertEquals(1000, opts1.maxDepth());
        assertTrue(opts1.decimalsAsBigDecimal());
        assertFalse(opts1.failOnDuplicateKeys());

        Json.ParseOptions opts2 = Json.ParseOptions.defaults().maxDepth(500);
        assertEquals(500, opts2.maxDepth());

        Json.ParseOptions opts3 = Json.ParseOptions.defaults().decimalsAsBigDecimal(false);
        assertFalse(opts3.decimalsAsBigDecimal());

        Json.ParseOptions opts4 = Json.ParseOptions.defaults().failOnDuplicateKeys(true);
        assertTrue(opts4.failOnDuplicateKeys());
    }

    @Test
    void testParseExceptionGetters() {
        try {
            Json.parse("{");
            fail("Should have thrown ParseException");
        } catch (Json.ParseException e) {
            // Just verify we can access the getters without exception
            // The actual values depend on the error context
            String msg = e.getMessage();
            int pos = e.getPosition();
            int line = e.getLine();
            int col = e.getColumn();

            // Verify message is present
            assertNotNull(msg);
            assertFalse(msg.isEmpty());

            // Verify getters return values (even if -1 for uninitialized)
            assertTrue(pos >= -1);
            assertTrue(line >= -1);
            assertTrue(col >= -1);
        }
    }

    @Test
    void testParseExceptionSimpleConstructor() {
        // Test simple constructors
        Json.ParseException e1 = new Json.ParseException("message");
        assertEquals("message", e1.getMessage());
        assertEquals(-1, e1.getPosition());

        Exception cause = new RuntimeException("cause");
        Json.ParseException e2 = new Json.ParseException("message", cause);
        assertEquals("message", e2.getMessage());
        assertEquals(cause, e2.getCause());
    }
}
