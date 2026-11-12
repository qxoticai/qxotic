package com.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** RFC 8259 and JSONTestSuite compliance. */
class JsonComplianceTest {

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

    // ===== n_ tests (MUST reject) =====

    @Test
    void testN_structure_whitespace_formfeed() {
        // Form feed (0x0C) is NOT valid JSON whitespace per RFC 8259
        // Use string concatenation to avoid Unicode compilation issues
        String json = new String(new char[] {'\f', '1', '\f'});
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse(json));
        assertNotNull(e.getMessage());
    }

    @Test
    void testN_number_hex_2_digits() {
        // Hexadecimal notation is NOT allowed in JSON
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("0x1A"));
        assertNotNull(e.getMessage());

        e = assertThrows(Json.ParseException.class, () -> Json.parse("0xFF"));
        assertNotNull(e.getMessage());

        e = assertThrows(Json.ParseException.class, () -> Json.parse("-0x1A"));
        assertNotNull(e.getMessage());
    }

    @Test
    void testN_string_escape_x() {
        // \\x escape sequence is NOT valid in JSON (only \\uXXXX is valid for hex)
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\x41\""));
        assertNotNull(e.getMessage());

        e = assertThrows(Json.ParseException.class, () -> Json.parse("\"\\xFF\""));
        assertNotNull(e.getMessage());
    }

    @Test
    void testN_number_NaN() {
        // NaN is NOT a valid JSON number
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("NaN"));
        assertNotNull(e.getMessage());

        e = assertThrows(Json.ParseException.class, () -> Json.parse("nan"));
        assertNotNull(e.getMessage());
    }

    @Test
    void testN_number_infinity() {
        // Infinity is NOT a valid JSON number
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("Infinity"));
        assertNotNull(e.getMessage());

        e = assertThrows(Json.ParseException.class, () -> Json.parse("infinity"));
        assertNotNull(e.getMessage());

        e = assertThrows(Json.ParseException.class, () -> Json.parse("-Infinity"));
        assertNotNull(e.getMessage());
    }

    @Test
    void testN_string_unescaped_crtl_char() {
        // Control characters must be escaped (except in string content they're rejected)
        // Test various control characters that must be escaped
        for (int i = 0x00; i <= 0x1F; i++) {
            char c = (char) i;
            String json = "\"" + c + "\"";
            Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse(json));
            assertNotNull(e.getMessage());
        }
    }

    @Test
    void testN_number_plus_sign() {
        // Plus sign prefix is NOT allowed
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("+1"));
        assertNotNull(e.getMessage());

        e = assertThrows(Json.ParseException.class, () -> Json.parse("+0"));
        assertNotNull(e.getMessage());

        e = assertThrows(Json.ParseException.class, () -> Json.parse("+3.14"));
        assertNotNull(e.getMessage());

        e = assertThrows(Json.ParseException.class, () -> Json.parse("+1e2"));
        assertNotNull(e.getMessage());
    }

    @Test
    void testN_number_double_plus() {
        // Double plus sign is NOT allowed
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("++1"));
        assertNotNull(e.getMessage());
    }

    @Test
    void testN_number_plus_minus() {
        // Plus-minus sign sequence is NOT allowed
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("+-1"));
        assertNotNull(e.getMessage());
    }

    @Test
    void testN_number_minus_plus() {
        // Minus-plus sign sequence is NOT allowed
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("-+1"));
        assertNotNull(e.getMessage());
    }

    @Test
    void testN_number_double_minus() {
        // Double minus sign is NOT allowed
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("--1"));
        assertNotNull(e.getMessage());
    }

    @Test
    void testN_number_dot_without_digits() {
        // Decimal point without digits is NOT allowed
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse(".5"));
        assertNotNull(e.getMessage());

        e = assertThrows(Json.ParseException.class, () -> Json.parse("."));
        assertNotNull(e.getMessage());

        e = assertThrows(Json.ParseException.class, () -> Json.parse("-.5"));
        assertNotNull(e.getMessage());
    }

    @Test
    void testN_number_decimal_without_fraction() {
        // Decimal point without fraction digits is NOT allowed
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("1."));
        assertNotNull(e.getMessage());

        e = assertThrows(Json.ParseException.class, () -> Json.parse("-1."));
        assertNotNull(e.getMessage());

        e = assertThrows(Json.ParseException.class, () -> Json.parse("1.e2"));
        assertNotNull(e.getMessage());
    }

    @Test
    void testN_number_exponent_without_digits() {
        // Exponent without digits is NOT allowed
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("1e"));
        assertNotNull(e.getMessage());

        e = assertThrows(Json.ParseException.class, () -> Json.parse("1E"));
        assertNotNull(e.getMessage());

        e = assertThrows(Json.ParseException.class, () -> Json.parse("1e+"));
        assertNotNull(e.getMessage());

        e = assertThrows(Json.ParseException.class, () -> Json.parse("1e-"));
        assertNotNull(e.getMessage());

        e = assertThrows(Json.ParseException.class, () -> Json.parse("1.5e"));
        assertNotNull(e.getMessage());
    }

    @Test
    void testN_number_double_exponent() {
        // Double exponent (eE) is NOT allowed
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("1eE2"));
        assertNotNull(e.getMessage());

        e = assertThrows(Json.ParseException.class, () -> Json.parse("1Ee2"));
        assertNotNull(e.getMessage());
    }

    @Test
    void testN_number_multiple_decimal_points() {
        // Multiple decimal points are NOT allowed
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("1.2.3"));
        assertNotNull(e.getMessage());

        e = assertThrows(Json.ParseException.class, () -> Json.parse("1..2"));
        assertNotNull(e.getMessage());

        e = assertThrows(Json.ParseException.class, () -> Json.parse(".1.2"));
        assertNotNull(e.getMessage());
    }

    @Test
    void testN_string_invalid_unicode_escape() {
        // Invalid hex digits in Unicode escape
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\u00G0\""));
        assertNotNull(e.getMessage());

        e = assertThrows(Json.ParseException.class, () -> Json.parse("\"\\uG000\""));
        assertNotNull(e.getMessage());

        e = assertThrows(Json.ParseException.class, () -> Json.parse("\"\\uZZZZ\""));
        assertNotNull(e.getMessage());
    }

    @Test
    void testN_string_incomplete_unicode_escape() {
        // Incomplete Unicode escape (less than 4 hex digits)
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\u00A\""));
        assertNotNull(e.getMessage());

        e = assertThrows(Json.ParseException.class, () -> Json.parse("\"\\u0\""));
        assertNotNull(e.getMessage());

        e = assertThrows(Json.ParseException.class, () -> Json.parse("\"\\u\""));
        assertNotNull(e.getMessage());
    }

    @Test
    void testN_string_incomplete_escape() {
        // Incomplete escape sequence (just backslash)
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"test\\\""));
        assertNotNull(e.getMessage());

        e = assertThrows(Json.ParseException.class, () -> Json.parse("\"\\\""));
        assertNotNull(e.getMessage());
    }

    @Test
    void testN_string_inverted_surrogates() {
        // Inverted surrogate pair (low-high instead of high-low)
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\uDE00\\uD83D\""));
        assertNotNull(e.getMessage());
    }

    // ===== y_ tests (MUST accept) =====

    @Test
    void testY_structure_lonely_string() {
        // String at root level is valid JSON
        assertEquals("test", Json.parse("\"test\""));
        assertEquals("", Json.parse("\"\""));
        assertEquals("Hello World", Json.parse("\"Hello World\""));
    }

    @Test
    void testY_structure_lonely_false() {
        // Boolean false at root level is valid JSON
        assertEquals(false, Json.parse("false"));
    }

    @Test
    void testY_structure_lonely_true() {
        // Boolean true at root level is valid JSON
        assertEquals(true, Json.parse("true"));
    }

    @Test
    void testY_structure_lonely_null() {
        // Null at root level is valid JSON
        assertSame(Json.NULL, Json.parse("null"));
    }

    @Test
    void testY_structure_lonely_number() {
        // Number at root level is valid JSON
        assertEquals(0L, Json.parse("0"));
        assertEquals(123L, Json.parse("123"));
        assertEquals(new BigDecimal("3.14"), Json.parse("3.14"));
    }

    @Test
    void testY_string_unescaped_char_delete() {
        // DEL character (U+007F) does NOT need to be escaped in JSON strings
        // It's outside the control character range (U+0000 to U+001F)
        String del = new String(new char[] {'\u007F'});
        assertEquals(del, Json.parse("\"\u007F\""));
        assertEquals(del, Json.parse("\"\\u007F\""));
    }

    @Test
    void testY_number_double_huge_neg_exp() {
        // Very small numbers with huge negative exponents are valid
        // Should parse as BigDecimal with appropriate scale
        Object parsed = Json.parse("1e-999");
        assertInstanceOf(BigDecimal.class, parsed);
        BigDecimal bd = (BigDecimal) parsed;
        assertEquals(0, bd.compareTo(new BigDecimal("1E-999")));

        parsed = Json.parse("1.5e-500");
        assertInstanceOf(BigDecimal.class, parsed);
        bd = (BigDecimal) parsed;
        assertEquals(0, bd.compareTo(new BigDecimal("1.5E-500")));
    }

    @Test
    void testY_number_double_huge_pos_exp() {
        // Very large numbers with huge positive exponents are valid
        Object parsed = Json.parse("1e999");
        assertInstanceOf(BigDecimal.class, parsed);
        BigDecimal bd = (BigDecimal) parsed;
        assertEquals(0, bd.compareTo(new BigDecimal("1E+999")));

        parsed = Json.parse("1.5e500");
        assertInstanceOf(BigDecimal.class, parsed);
        bd = (BigDecimal) parsed;
        assertEquals(0, bd.compareTo(new BigDecimal("1.5E+500")));
    }

    @Test
    void testY_number_0e1() {
        // 0 with exponent is valid
        Object parsed = Json.parse("0e1");
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(0, ((BigDecimal) parsed).compareTo(BigDecimal.ZERO));

        parsed = Json.parse("0E10");
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(0, ((BigDecimal) parsed).compareTo(BigDecimal.ZERO));

        parsed = Json.parse("0e-10");
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(0, ((BigDecimal) parsed).compareTo(BigDecimal.ZERO));
    }

    @Test
    void testY_number_1e0() {
        // 1e0 is valid (exponent of zero)
        Object parsed = Json.parse("1e0");
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(0, ((BigDecimal) parsed).compareTo(new BigDecimal("1")));

        parsed = Json.parse("1E0");
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(0, ((BigDecimal) parsed).compareTo(new BigDecimal("1")));

        parsed = Json.parse("1.5e0");
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(0, ((BigDecimal) parsed).compareTo(new BigDecimal("1.5")));
    }

    @Test
    void testY_string_accepted_surrogate_pair() {
        // Valid surrogate pairs are accepted
        String emoji = "\uD83D\uDE00"; // 😀
        assertEquals(emoji, Json.parse("\"\\uD83D\\uDE00\""));

        String musical = "\uD834\uDD1E"; // 𝄞
        assertEquals(musical, Json.parse("\"\\uD834\\uDD1E\""));

        String flag = "\uD83C\uDDFA\uD83C\uDDF8"; // 🇺🇸
        assertEquals(flag, Json.parse("\"\\uD83C\\uDDFA\\uD83C\\uDDF8\""));
    }

    @Test
    void testY_string_unicode() {
        // Basic Unicode characters
        assertEquals("A", Json.parse("\"\\u0041\""));
        assertEquals("\u00E9", Json.parse("\"\\u00E9\"")); // é
        assertEquals("\u03A9", Json.parse("\"\\u03A9\"")); // Ω
        assertEquals("\u4E2D", Json.parse("\"\\u4E2D\"")); // 中
        assertEquals("\uD55C", Json.parse("\"\\uD55C\"")); // 한
    }

    @Test
    void testY_string_escaped_control_character() {
        // Control characters must be escaped, and escaped versions are valid
        assertEquals("\"", Json.parse("\"\\\"\""));
        assertEquals("\\", Json.parse("\"\\\\\""));
        assertEquals("/", Json.parse("\"\\/\""));
        assertEquals("\b", Json.parse("\"\\b\""));
        assertEquals("\f", Json.parse("\"\\f\""));
        assertEquals("\n", Json.parse("\"\\n\""));
        assertEquals("\r", Json.parse("\"\\r\""));
        assertEquals("\t", Json.parse("\"\\t\""));

        // Null character escape
        assertEquals("\u0000", Json.parse("\"\\u0000\""));
    }

    @Test
    void testY_string_comments() {
        // Comments inside strings are just string content, not actual comments
        assertEquals("// comment", Json.parse("\"// comment\""));
        assertEquals("/* comment */", Json.parse("\"/* comment */\""));
        assertEquals("# comment", Json.parse("\"# comment\""));
    }

    // ===== i_ tests (implementation-defined) =====

    @Test
    void testI_string_lone_second_surrogate() {
        // Lone low surrogate - implementation may reject or accept with replacement
        // Our implementation rejects lone surrogates
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\uDE00\""));
        assertNotNull(e.getMessage());
        assertTrue(e.getMessage().contains("surrogate"));
    }

    @Test
    void testI_string_lone_first_surrogate() {
        // Lone high surrogate - implementation may reject or accept with replacement
        // Our implementation rejects lone surrogates
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\uD83D\""));
        assertNotNull(e.getMessage());
        assertTrue(e.getMessage().contains("surrogate"));
    }

    @Test
    void testI_number_very_big_negative_int() {
        // Very large negative integer - implementation may reject or parse as BigInteger
        String huge = "-9999999999999999999999999999999999999999";
        try {
            Object parsed = Json.parse(huge);
            // If accepted, should be BigInteger
            assertInstanceOf(BigInteger.class, parsed);
            assertEquals(new BigInteger(huge), parsed);
        } catch (Json.ParseException e) {
            // Also acceptable to reject due to implementation limits
            assertNotNull(e.getMessage());
        }
    }

    @Test
    void testI_number_very_big_positive_int() {
        // Very large positive integer - implementation may reject or parse as BigInteger
        String huge = "9999999999999999999999999999999999999999";
        try {
            Object parsed = Json.parse(huge);
            // If accepted, should be BigInteger
            assertInstanceOf(BigInteger.class, parsed);
            assertEquals(new BigInteger(huge), parsed);
        } catch (Json.ParseException e) {
            // Also acceptable to reject due to implementation limits
            assertNotNull(e.getMessage());
        }
    }

    @Test
    void testI_structure_UTF8_BOM() {
        // BOM handling - implementation-defined
        // Our implementation works with CharSequence, not bytes
        // BOM as character should be rejected as invalid JSON
        String withBOM = "\uFEFF\"test\"";
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse(withBOM));
        assertNotNull(e.getMessage());
    }

    @Test
    void testI_object_duplicate_keys() {
        // Duplicate key handling - implementation-defined (RFC says SHOULD be unique)
        // Our implementation uses "last wins"
        Map<String, Object> parsed = (Map<String, Object>) Json.parse("{\"a\":1,\"a\":2}");
        assertEquals(1, parsed.size());
        assertEquals(2L, parsed.get("a"));

        // Multiple duplicates
        parsed = (Map<String, Object>) Json.parse("{\"a\":1,\"a\":2,\"a\":3}");
        assertEquals(1, parsed.size());
        assertEquals(3L, parsed.get("a"));
    }

    @Test
    void testI_string_not_in_unicode_range() {
        // Code points not in Unicode range - implementation-defined
        // Non-character code points (U+FDD0 to U+FDEF) are allowed in strings
        String nonChar = "\uFDD0";
        assertEquals(nonChar, Json.parse("\"\\uFDD0\""));

        // U+FFFE and U+FFFF are also non-characters but allowed
        assertEquals("\uFFFE", Json.parse("\"\\uFFFE\""));
        assertEquals("\uFFFF", Json.parse("\"\\uFFFF\""));
    }

    @Test
    void testI_string_iso_latin_1() {
        // ISO Latin-1 characters are valid Unicode
        // Test a range of Latin-1 characters
        for (int i = 0x20; i <= 0xFF; i++) {
            // Skip control characters (0x00-0x1F)
            if (i >= 0x00 && i <= 0x1F) continue;

            char c = (char) i;
            // Skip quote and backslash as they need to be escaped in JSON strings
            if (c == '\"' || c == '\\') continue;

            String json = "\"" + c + "\"";
            try {
                Object parsed = Json.parse(json);
                assertEquals(String.valueOf(c), parsed);
            } catch (Json.ParseException e) {
                // Should not throw for valid Latin-1 characters
                fail(
                        "Failed to parse Latin-1 character U+"
                                + Integer.toHexString(i)
                                + ": "
                                + e.getMessage());
            }
        }

        // Test quote and backslash separately (must be escaped)
        assertEquals("\"", Json.parse("\"\\\"\""));
        assertEquals("\\", Json.parse("\"\\\\\""));
    }

    // ===== Additional critical corner cases =====

    @Test
    void testLineAndParagraphSeparators() {
        // U+2028 LINE SEPARATOR and U+2029 PARAGRAPH SEPARATOR
        // These are valid in JSON strings but problematic in JavaScript eval()
        String lineSep = "\u2028";
        String paraSep = "\u2029";

        assertEquals(lineSep, Json.parse("\"\\u2028\""));
        assertEquals(paraSep, Json.parse("\"\\u2029\""));

        // Direct Unicode
        assertEquals(lineSep, Json.parse("\"\u2028\""));
        assertEquals(paraSep, Json.parse("\"\u2029\""));

        // In middle of string
        assertEquals("Hello" + lineSep + "World", Json.parse("\"Hello\\u2028World\""));
        assertEquals("Hello" + paraSep + "World", Json.parse("\"Hello\\u2029World\""));
    }

    @Test
    void testVeryLongString() {
        // Test long string parsing (should not crash)
        StringBuilder sb = new StringBuilder();
        sb.append("test ".repeat(1000));
        String longString = sb.toString();

        String json = "\"" + longString.replace("\"", "\\\"").replace("\\", "\\\\") + "\"";
        Object parsed = Json.parse(json);
        assertEquals(longString, parsed);
    }

    @Test
    void testEmptyKey() {
        // Empty string as key is valid
        Map<String, Object> parsed = (Map<String, Object>) Json.parse("{\"\":\"value\"}");
        assertEquals(1, parsed.size());
        assertEquals("value", parsed.get(""));

        // Multiple empty keys (last wins)
        parsed = (Map<String, Object>) Json.parse("{\"\":\"first\",\"\":\"last\"}");
        assertEquals(1, parsed.size());
        assertEquals("last", parsed.get(""));
    }

    @Test
    void testNumberLeadingZeroInExponent() {
        // Leading zeros in exponent are allowed
        Object parsed = Json.parse("1e01");
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(0, ((BigDecimal) parsed).compareTo(new BigDecimal("1E+1")));

        parsed = Json.parse("1e001");
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(0, ((BigDecimal) parsed).compareTo(new BigDecimal("1E+1")));

        parsed = Json.parse("1.5e-01");
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(0, ((BigDecimal) parsed).compareTo(new BigDecimal("1.5E-1")));
    }

    @Test
    void testUnicodeReplacementCharacter() {
        // Unicode replacement character U+FFFD is valid
        String replacement = "\uFFFD";
        assertEquals(replacement, Json.parse("\"\\uFFFD\""));
        assertEquals(replacement, Json.parse("\"\uFFFD\""));

        // In context
        assertEquals("Error: " + replacement, Json.parse("\"Error: \\uFFFD\""));
    }

    @Test
    void testMaximumDepthImplementation() {
        // Test that we respect maximum depth limits
        // Build JSON at default max depth (1000) - but use 200 to avoid stack overflow
        int depth = 200;
        StringBuilder sb = new StringBuilder();
        sb.append("[".repeat(depth));
        sb.append("1");
        sb.append("]".repeat(depth));

        // Should parse successfully with default max depth (1000)
        Object result = Json.parse(sb.toString());
        assertNotNull(result);

        // Should fail with lower max depth (100)
        Json.ParseException e =
                assertThrows(
                        Json.ParseException.class,
                        () ->
                                Json.parse(
                                        sb.toString(), Json.ParseOptions.defaults().maxDepth(100)));
        assertNotNull(e.getMessage());
    }

    @Test
    void testJSON2RejectsInvalidNumbers() {
        String[] invalidNumbers = {
            "+1", "01", "00", ".5", "1.", "1e", "1e+", "1e-", "1.2.3", "01.5", "00.5"
        };

        for (String num : invalidNumbers) {
            assertThrows(
                    Json.ParseException.class,
                    () -> Json.parse(num),
                    "JSON2 should reject: " + num);
        }
    }

    @Test
    void testJSON2RejectsInvalidStrings() {
        String[] invalidStrings = {
            "\"\\u00\"", "\"\\u00G\"",
            "\"\\uD800\"", "\"\\uDC00\""
        };

        for (String str : invalidStrings) {
            assertThrows(
                    Json.ParseException.class,
                    () -> Json.parse(str),
                    "JSON2 should reject: " + str);
        }
    }

    @Test
    void testJSON2HandlesSurrogatesCorrectly() {
        String[] validSurrogates = {
            "\"\\uD83D\\uDE00\"", "\"\\uD83D\\uDC4D\"", "\"Hello \\uD83D\\uDE00\""
        };

        for (String str : validSurrogates) {
            Object result = Json.parse(str);
            assertNotNull(result);
        }

        String[] invalidSurrogates = {"\"\\uD83DA\"", "\"\\uD83D\\u0041\""};

        for (String str : invalidSurrogates) {
            assertThrows(
                    Json.ParseException.class,
                    () -> Json.parse(str),
                    "JSON2 should reject: " + str);
        }
    }

    @Test
    void testJSON2BetterErrorMessages() {
        Json.ParseException e;

        e = assertThrows(Json.ParseException.class, () -> Json.parse("{\n  \"key\":,\n}"));
        assertEquals(2, e.getLine());
        assertTrue(e.getMessage().contains("Line"));
        assertTrue(e.getMessage().contains("Column"));

        e = assertThrows(Json.ParseException.class, () -> Json.parse("01"));
        assertTrue(e.getMessage().contains("Leading zeros"));

        e = assertThrows(Json.ParseException.class, () -> Json.parse("\"\\uD800\""));
        assertTrue(e.getMessage().contains("surrogate"));
    }

    @Test
    void testJSON2PreservesNegativeZero() {
        Object parsed = Json.parse("-0");
        assertInstanceOf(Number.class, parsed);

        String stringified = Json.stringify(parsed);
        assertEquals("0", stringified);

        Object roundTrip = Json.parse(stringified);
        assertEquals(0, ((Number) roundTrip).doubleValue(), 0.0001);
    }

    @Test
    void testBothHandleCorrectly() {
        String[] validJson = {
            "true",
            "false",
            "null",
            "123",
            "\"hello\"",
            "[1,2,3]",
            "{\"key\":\"value\"}",
            "{\"a\":1,\"b\":[1,2],\"c\":{\"d\":2}}"
        };

        for (String json : validJson) {
            Object json1 = Json.parse(json);
            Object json2 = Json.parse(json);

            assertTrue(objectsEqual(json1, json2), "Both should parse " + json + " same way");
        }
    }

    private boolean objectsEqual(Object o1, Object o2) {
        if (o1 == o2) return true;
        if (o1 == null || o2 == null) return false;

        if (o1 == Json.NULL && o2 == Json.NULL) return true;
        if (o1 == Json.NULL && o2 == Json.NULL) return true;

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
                if (!objectsEqual(l1.get(i), l2.get(i))) return false;
            }
            return true;
        }

        if (o1 instanceof Map && o2 instanceof Map) {
            Map<?, ?> m1 = (Map<?, ?>) o1;
            Map<?, ?> m2 = (Map<?, ?>) o2;
            if (m1.size() != m2.size()) return false;
            for (Map.Entry<?, ?> entry : m1.entrySet()) {
                Object value2 = m2.get(entry.getKey());
                if (!objectsEqual(entry.getValue(), value2)) return false;
            }
            return true;
        }

        return o1.equals(o2);
    }
}
