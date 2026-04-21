package com.qxotic.format.json;

import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * RFC 8259 JSON specification compliance tests. Tests strict adherence to the JSON specification.
 */
class JsonSpecComplianceTest {

    // ===== RFC 8259 Section 2: JSON Grammar =====

    @Test
    void testWhitespaceAllowed() {
        // RFC 8259: ws = *(%x20 / %x09 / %x0A / %x0D)
        // Space, tab, newline, carriage return
        assertEquals(1L, Json.parse(" 1 "));
        assertEquals(1L, Json.parse("\t1\t"));
        assertEquals(1L, Json.parse("\n1\n"));
        assertEquals(1L, Json.parse("\r1\r"));
        assertEquals(1L, Json.parse(" \t\n\r 1 \t\n\r "));

        // Whitespace in objects and arrays
        assertEquals(Map.of("a", 1L), Json.parse(" { \"a\" : 1 } "));
        assertEquals(List.of(1L, 2L), Json.parse(" [ 1 , 2 ] "));
    }

    @Test
    void testNoOtherWhitespaceAllowed() {
        // Form feed (0x0C) is NOT allowed by RFC 8259
        String withFormFeed = "\f1\f";
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse(withFormFeed));
        // Should reject form feed as unexpected character
        assertTrue(e.getMessage().contains("Unexpected"));
    }

    // ===== RFC 8259 Section 3: Values =====

    @Test
    void testNullLiteral() {
        assertSame(Json.NULL, Json.parse("null"));
        assertEquals("null", Json.stringify(Json.NULL));

        // Case sensitive
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("Null"));
        assertTrue(e.getMessage().contains("Unexpected"));

        e = assertThrows(Json.ParseException.class, () -> Json.parse("NULL"));
        assertTrue(e.getMessage().contains("Unexpected"));
    }

    @Test
    void testFalseLiteral() {
        assertEquals(false, Json.parse("false"));
        assertEquals("false", Json.stringify(false));

        // Case sensitive
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("False"));
        assertTrue(e.getMessage().contains("Unexpected"));

        e = assertThrows(Json.ParseException.class, () -> Json.parse("FALSE"));
        assertTrue(e.getMessage().contains("Unexpected"));
    }

    @Test
    void testTrueLiteral() {
        assertEquals(true, Json.parse("true"));
        assertEquals("true", Json.stringify(true));

        // Case sensitive
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("True"));
        assertTrue(e.getMessage().contains("Unexpected"));

        e = assertThrows(Json.ParseException.class, () -> Json.parse("TRUE"));
        assertTrue(e.getMessage().contains("Unexpected"));
    }

    // ===== RFC 8259 Section 6: Numbers =====

    @Test
    void testNumberInteger() {
        // RFC 8259: int = zero / ( digit1-9 *DIGIT )
        assertEquals(0L, Json.parse("0"));
        assertEquals(123L, Json.parse("123"));
        assertEquals(-123L, Json.parse("-123"));

        // No leading zeros (except zero itself)
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("0123"));
        assertTrue(e.getMessage().contains("Leading zeros"));

        e = assertThrows(Json.ParseException.class, () -> Json.parse("00123"));
        assertTrue(e.getMessage().contains("Leading zeros"));

        // Negative zero is allowed (parses as 0L - no negative zero for integers)
        Object parsed = Json.parse("-0");
        assertEquals(0L, parsed);
    }

    @Test
    void testNumberNoPlusSign() {
        // RFC 8259: number = [ minus ] int [ frac ] [ exp ]
        // No plus sign allowed
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("+123"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("plus"));

        e = assertThrows(Json.ParseException.class, () -> Json.parse("+0"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("plus"));

        e = assertThrows(Json.ParseException.class, () -> Json.parse("+3.14"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("plus"));
    }

    @Test
    void testNumberFraction() {
        // RFC 8259: frac = decimal-point 1*DIGIT
        assertEquals(new BigDecimal("0.5"), Json.parse("0.5"));
        assertEquals(new BigDecimal("3.14"), Json.parse("3.14"));
        assertEquals(new BigDecimal("-3.14"), Json.parse("-3.14"));

        // Must have at least one digit after decimal point
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("3."));
        assertTrue(e.getMessage().contains("digit") || e.getMessage().contains("Unexpected"));

        // Must have at least one digit before decimal point (except zero)
        e = assertThrows(Json.ParseException.class, () -> Json.parse(".5"));
        assertTrue(e.getMessage().contains("digit") || e.getMessage().contains("Unexpected"));

        // But .0 after zero is covered by "0.5" case
        // Parser preserves original precision (no stripTrailingZeros)
        assertEquals(new BigDecimal("0.0"), Json.parse("0.0"));
    }

    @Test
    void testNumberExponent() {
        // RFC 8259: exp = e [ minus / plus ] 1*DIGIT
        assertEquals(new BigDecimal("1e2"), Json.parse("1e2"));
        assertEquals(new BigDecimal("1E2"), Json.parse("1E2"));
        assertEquals(new BigDecimal("1e+2"), Json.parse("1e+2"));
        assertEquals(new BigDecimal("1e-2"), Json.parse("1e-2"));
        assertEquals(new BigDecimal("-1e2"), Json.parse("-1e2"));

        // Must have at least one digit after e/E
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("1e"));
        assertTrue(e.getMessage().contains("exponent") || e.getMessage().contains("digit"));

        e = assertThrows(Json.ParseException.class, () -> Json.parse("1e+"));
        assertTrue(e.getMessage().contains("exponent") || e.getMessage().contains("digit"));

        e = assertThrows(Json.ParseException.class, () -> Json.parse("1e-"));
        assertTrue(e.getMessage().contains("exponent") || e.getMessage().contains("digit"));
    }

    @Test
    void testNumberLeadingZerosWithFractionOrExponent() {
        // Leading zeros not allowed even with fraction or exponent
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("01.5"));
        assertTrue(e.getMessage().contains("Leading zeros"));

        e = assertThrows(Json.ParseException.class, () -> Json.parse("00.5"));
        assertTrue(e.getMessage().contains("Leading zeros"));

        e = assertThrows(Json.ParseException.class, () -> Json.parse("01e2"));
        assertTrue(e.getMessage().contains("Leading zeros"));

        e = assertThrows(Json.ParseException.class, () -> Json.parse("00e2"));
        assertTrue(e.getMessage().contains("Leading zeros"));
    }

    @Test
    void testNumberMultipleDecimalPoints() {
        // Only one decimal point allowed
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("1.2.3"));
        // Error message should indicate the issue
        assertNotNull(e.getMessage());
    }

    // ===== RFC 8259 Section 7: Strings =====

    @Test
    void testStringBasic() {
        // Must be double quotes
        assertEquals("test", Json.parse("\"test\""));

        // Single quotes not allowed
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("'test'"));
        // Should throw ParseException
        assertNotNull(e.getMessage());

        // No quotes not allowed
        e = assertThrows(Json.ParseException.class, () -> Json.parse("test"));
        // Should throw ParseException
        assertNotNull(e.getMessage());
    }

    @Test
    void testStringControlCharactersMustBeEscaped() {
        // RFC 8259: Control characters (U+0000 through U+001F) must be escaped
        for (int i = 0x00; i <= 0x1F; i++) {
            char c = (char) i;
            String json = "\"" + c + "\"";
            Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse(json));
            assertTrue(e.getMessage().contains("Control") || e.getMessage().contains("Unexpected"));
        }

        // U+0020 (space) and above don't need escaping
        assertEquals(" ", Json.parse("\" \""));
        assertEquals("!", Json.parse("\"!\""));
    }

    @Test
    void testStringQuoteAndBackslashMustBeEscaped() {
        // " and \ must be escaped
        assertEquals("\"", Json.parse("\"\\\"\""));
        assertEquals("\\", Json.parse("\"\\\\\""));

        // Unescaped " should fail
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"test\"test\""));
        // Should throw ParseException
        assertNotNull(e.getMessage());

        // "\test" - \t is a valid escape (tab), so this parses as "test<tab>est"
        // To test invalid escape, use \ followed by invalid character
        e = assertThrows(Json.ParseException.class, () -> Json.parse("\"test\\xtest\""));
        // Should throw ParseException for invalid escape
        assertNotNull(e.getMessage());
    }

    @Test
    void testStringEscapeSequences() {
        // RFC 8259: \" \\ \/ \b \f \n \r \t
        assertEquals("\"", Json.parse("\"\\\"\""));
        assertEquals("\\", Json.parse("\"\\\\\""));
        assertEquals("/", Json.parse("\"\\/\""));
        assertEquals("\b", Json.parse("\"\\b\""));
        assertEquals("\f", Json.parse("\"\\f\""));
        assertEquals("\n", Json.parse("\"\\n\""));
        assertEquals("\r", Json.parse("\"\\r\""));
        assertEquals("\t", Json.parse("\"\\t\""));

        // Forward slash may be escaped but doesn't need to be
        assertEquals("/", Json.parse("\"/\""));
    }

    @Test
    void testStringUnicodeEscape() {
        // RFC 8259: \\uXXXX where X is hex digit
        assertEquals("A", Json.parse("\"\\u0041\""));
        assertEquals("\u00E9", Json.parse("\"\\u00E9\""));
        assertEquals("\u03A9", Json.parse("\"\\u03A9\""));

        // Must be exactly 4 hex digits
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\u041\""));
        assertTrue(e.getMessage().contains("hex") || e.getMessage().contains("Unexpected"));

        // "\u00410" - after \u0041 (A), there's a '0' character which is valid
        // So this should parse as "A0", not fail
        assertEquals("A0", Json.parse("\"\\u00410\""));

        // Hex digits must be valid
        e = assertThrows(Json.ParseException.class, () -> Json.parse("\"\\u00G0\""));
        assertTrue(e.getMessage().contains("hex") || e.getMessage().contains("Invalid"));
    }

    @Test
    void testStringSurrogatePairs() {
        // RFC 8259: UTF-16 surrogate pairs must be properly encoded
        // Valid surrogate pair
        assertEquals("\uD83D\uDE00", Json.parse("\"\\uD83D\\uDE00\""));

        // Lone high surrogate
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\uD83D\""));
        assertTrue(e.getMessage().contains("surrogate"));

        // Lone low surrogate
        e = assertThrows(Json.ParseException.class, () -> Json.parse("\"\\uDE00\""));
        assertTrue(e.getMessage().contains("surrogate"));

        // Reversed surrogate pair
        e = assertThrows(Json.ParseException.class, () -> Json.parse("\"\\uDE00\\uD83D\""));
        assertTrue(e.getMessage().contains("surrogate") || e.getMessage().contains("Invalid"));
    }

    @Test
    void testStringDirectUnicodeOutput() {
        // RFC 8259: Non-ASCII characters may be output directly
        // (not required to escape as \\uXXXX)
        // Test with various Unicode characters
        String unicode = "\u00E9\u03A9\u4E16\u754C\uD83D\uDE00";
        String json = Json.stringify(unicode, false);
        // Should output direct Unicode, not escaped
        assertEquals("\"" + unicode + "\"", json);

        // Round trip
        assertEquals(unicode, Json.parse(json));
    }

    // ===== RFC 8259 Section 4: Objects =====

    @Test
    void testObjectStructure() {
        // RFC 8259: object = begin-object [ member *( value-separator member ) ] end-object
        assertEquals(Collections.emptyMap(), Json.parse("{}"));

        Map<String, Object> expected = new LinkedHashMap<>();
        expected.put("a", 1L);
        expected.put("b", 2L);
        assertEquals(expected, Json.parse("{\"a\":1,\"b\":2}"));

        // No trailing comma
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("{\"a\":1,}"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("Expected"));

        // No extra comma
        e = assertThrows(Json.ParseException.class, () -> Json.parse("{\"a\":1,,}"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("Expected"));
    }

    @Test
    void testObjectMember() {
        // RFC 8259: member = string name-separator value
        assertEquals(Map.of("key", "value"), Json.parse("{\"key\":\"value\"}"));

        // Key must be string
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("{key:\"value\"}"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("Expected"));

        e = assertThrows(Json.ParseException.class, () -> Json.parse("{1:\"value\"}"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("Expected"));

        // Colon required
        e = assertThrows(Json.ParseException.class, () -> Json.parse("{\"key\" \"value\"}"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("Expected"));
    }

    @Test
    void testObjectDuplicateNames() {
        // RFC 8259: "The names within an object SHOULD be unique."
        // But: "JSON parsers are free to accept and ignore duplicate names."
        // We implement "last wins" which is acceptable
        Map<String, Object> parsed = (Map<String, Object>) Json.parse("{\"a\":1,\"a\":2}");
        assertEquals(1, parsed.size());
        assertEquals("2", parsed.get("a").toString()); // Last wins
    }

    // ===== RFC 8259 Section 5: Arrays =====

    @Test
    void testArrayStructure() {
        // RFC 8259: array = begin-array [ value *( value-separator value ) ] end-array
        assertEquals(Collections.emptyList(), Json.parse("[]"));

        assertEquals(List.of(1L, 2L, 3L), Json.parse("[1,2,3]"));

        // No trailing comma
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("[1,]"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("Expected"));

        // No extra comma
        e = assertThrows(Json.ParseException.class, () -> Json.parse("[1,,2]"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("Expected"));
    }

    // ===== RFC 8259 Section 8: String and Character Issues =====

    @Test
    void testUnicodeBOM() {
        // RFC 8259: "JSON text exchanged between systems that are not part of a closed
        // ecosystem MUST be encoded using UTF-8 [RFC3629]."
        // "Implementations MUST NOT add a byte order mark to the beginning of a JSON text."
        // Our implementation works with CharSequence, not bytes, so BOM is not applicable.
        // But if a BOM character appears in the string, it should be treated as a character.

        // BOM as first character (U+FEFF)
        String withBOM = "\uFEFF\"test\"";
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse(withBOM));
        // Should fail because BOM is not valid JSON
        assertTrue(e.getMessage().contains("Unexpected"));
    }

    // ===== RFC 8259 Compliance: Overall =====

    @Test
    void testValidJSONExamplesFromRFC() {
        // Example from RFC 8259
        String example =
                "{\n"
                        + "  \"Image\": {\n"
                        + "    \"Width\": 800,\n"
                        + "    \"Height\": 600,\n"
                        + "    \"Title\": \"View from 15th Floor\",\n"
                        + "    \"Thumbnail\": {\n"
                        + "      \"Url\": \"http://www.example.com/image/481989943\",\n"
                        + "      \"Height\": 125,\n"
                        + "      \"Width\": 100\n"
                        + "    },\n"
                        + "    \"Animated\": false,\n"
                        + "    \"IDs\": [116, 943, 234, 38793]\n"
                        + "  }\n"
                        + "}\n";

        Object parsed = Json.parse(example);
        assertNotNull(parsed);
        assertInstanceOf(Map.class, parsed);

        // Should be able to stringify it back
        String stringified = Json.stringify(parsed, false);
        Object reparsed = Json.parse(stringified);
        // Compare structure (may have different number types)
        assertInstanceOf(Map.class, reparsed);
    }

    @Test
    void testInvalidJSONExamples() {
        // Examples of invalid JSON per RFC 8259

        // Unclosed array
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("[1,2"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("Expected"));

        // Unclosed object
        e = assertThrows(Json.ParseException.class, () -> Json.parse("{\"a\":1"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("Expected"));

        // Missing comma
        e = assertThrows(Json.ParseException.class, () -> Json.parse("[1 2]"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("Expected"));

        // Extra comma in array
        e = assertThrows(Json.ParseException.class, () -> Json.parse("[1,2,]"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("Expected"));

        // Extra comma in object
        e = assertThrows(Json.ParseException.class, () -> Json.parse("{\"a\":1,}"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("Expected"));
    }

    @Test
    void testJSONTextMustBeObjectOrArray() {
        // RFC 8259: "A JSON text is a serialized value."
        // It can be any JSON value, not just object or array
        assertEquals(1L, Json.parse("1")); // Number is valid JSON text
        assertEquals("test", Json.parse("\"test\"")); // String is valid JSON text
        assertEquals(true, Json.parse("true")); // Boolean is valid JSON text
        assertSame(Json.NULL, Json.parse("null")); // Null is valid JSON text

        // Empty string is not valid JSON
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse(""));
        // Should throw ParseException
        assertNotNull(e.getMessage());

        // Just whitespace is not valid JSON
        e = assertThrows(Json.ParseException.class, () -> Json.parse("   "));
        // Should throw ParseException
        assertNotNull(e.getMessage());
    }
}
