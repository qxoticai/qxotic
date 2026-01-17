package ai.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import java.math.BigDecimal;
import java.util.*;
import org.junit.jupiter.api.Test;

/**
 * RFC 8259 JSON specification compliance tests. Tests strict adherence to the JSON specification.
 */
class JSONSpecComplianceTest {

    // ===== RFC 8259 Section 2: JSON Grammar =====

    @Test
    void testWhitespaceAllowed() {
        // RFC 8259: ws = *(%x20 / %x09 / %x0A / %x0D)
        // Space, tab, newline, carriage return
        assertEquals(1L, JSON.parse(" 1 "));
        assertEquals(1L, JSON.parse("\t1\t"));
        assertEquals(1L, JSON.parse("\n1\n"));
        assertEquals(1L, JSON.parse("\r1\r"));
        assertEquals(1L, JSON.parse(" \t\n\r 1 \t\n\r "));

        // Whitespace in objects and arrays
        assertEquals(Map.of("a", 1L), JSON.parse(" { \"a\" : 1 } "));
        assertEquals(List.of(1L, 2L), JSON.parse(" [ 1 , 2 ] "));
    }

    @Test
    void testNoOtherWhitespaceAllowed() {
        // Form feed (0x0C) is NOT allowed by RFC 8259
        String withFormFeed = "\f1\f";
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse(withFormFeed));
        // Should reject form feed as unexpected character
        assertTrue(e.getMessage().contains("Unexpected"));
    }

    // ===== RFC 8259 Section 3: Values =====

    @Test
    void testNullLiteral() {
        assertSame(JSON.NULL, JSON.parse("null"));
        assertEquals("null", JSON.stringify(JSON.NULL));

        // Case sensitive
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("Null"));
        assertTrue(e.getMessage().contains("Unexpected"));

        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("NULL"));
        assertTrue(e.getMessage().contains("Unexpected"));
    }

    @Test
    void testFalseLiteral() {
        assertEquals(false, JSON.parse("false"));
        assertEquals("false", JSON.stringify(false));

        // Case sensitive
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("False"));
        assertTrue(e.getMessage().contains("Unexpected"));

        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("FALSE"));
        assertTrue(e.getMessage().contains("Unexpected"));
    }

    @Test
    void testTrueLiteral() {
        assertEquals(true, JSON.parse("true"));
        assertEquals("true", JSON.stringify(true));

        // Case sensitive
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("True"));
        assertTrue(e.getMessage().contains("Unexpected"));

        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("TRUE"));
        assertTrue(e.getMessage().contains("Unexpected"));
    }

    // ===== RFC 8259 Section 6: Numbers =====

    @Test
    void testNumberInteger() {
        // RFC 8259: int = zero / ( digit1-9 *DIGIT )
        assertEquals(0L, JSON.parse("0"));
        assertEquals(123L, JSON.parse("123"));
        assertEquals(-123L, JSON.parse("-123"));

        // No leading zeros (except zero itself)
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("0123"));
        assertTrue(e.getMessage().contains("Leading zeros"));

        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("00123"));
        assertTrue(e.getMessage().contains("Leading zeros"));

        // Negative zero is allowed (parses as BigDecimal)
        Object parsed = JSON.parse("-0");
        assertTrue(parsed instanceof BigDecimal);
        assertEquals(0, ((BigDecimal) parsed).compareTo(BigDecimal.ZERO));
    }

    @Test
    void testNumberNoPlusSign() {
        // RFC 8259: number = [ minus ] int [ frac ] [ exp ]
        // No plus sign allowed
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("+123"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("plus"));

        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("+0"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("plus"));

        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("+3.14"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("plus"));
    }

    @Test
    void testNumberFraction() {
        // RFC 8259: frac = decimal-point 1*DIGIT
        assertEquals(new BigDecimal("0.5"), JSON.parse("0.5"));
        assertEquals(new BigDecimal("3.14"), JSON.parse("3.14"));
        assertEquals(new BigDecimal("-3.14"), JSON.parse("-3.14"));

        // Must have at least one digit after decimal point
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("3."));
        assertTrue(e.getMessage().contains("digit") || e.getMessage().contains("Unexpected"));

        // Must have at least one digit before decimal point (except zero)
        e = assertThrows(JSON.ParseException.class, () -> JSON.parse(".5"));
        assertTrue(e.getMessage().contains("digit") || e.getMessage().contains("Unexpected"));

        // But .0 after zero is covered by "0.5" case
        // Note: We strip trailing zeros for zero values
        assertEquals(new BigDecimal("0"), JSON.parse("0.0"));
    }

    @Test
    void testNumberExponent() {
        // RFC 8259: exp = e [ minus / plus ] 1*DIGIT
        assertEquals(new BigDecimal("1e2"), JSON.parse("1e2"));
        assertEquals(new BigDecimal("1E2"), JSON.parse("1E2"));
        assertEquals(new BigDecimal("1e+2"), JSON.parse("1e+2"));
        assertEquals(new BigDecimal("1e-2"), JSON.parse("1e-2"));
        assertEquals(new BigDecimal("-1e2"), JSON.parse("-1e2"));

        // Must have at least one digit after e/E
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("1e"));
        assertTrue(e.getMessage().contains("exponent") || e.getMessage().contains("digit"));

        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("1e+"));
        assertTrue(e.getMessage().contains("exponent") || e.getMessage().contains("digit"));

        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("1e-"));
        assertTrue(e.getMessage().contains("exponent") || e.getMessage().contains("digit"));
    }

    @Test
    void testNumberLeadingZerosWithFractionOrExponent() {
        // Leading zeros not allowed even with fraction or exponent
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("01.5"));
        assertTrue(e.getMessage().contains("Leading zeros"));

        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("00.5"));
        assertTrue(e.getMessage().contains("Leading zeros"));

        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("01e2"));
        assertTrue(e.getMessage().contains("Leading zeros"));

        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("00e2"));
        assertTrue(e.getMessage().contains("Leading zeros"));
    }

    @Test
    void testNumberMultipleDecimalPoints() {
        // Only one decimal point allowed
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("1.2.3"));
        // Error message should indicate the issue
        assertNotNull(e.getMessage());
    }

    // ===== RFC 8259 Section 7: Strings =====

    @Test
    void testStringBasic() {
        // Must be double quotes
        assertEquals("test", JSON.parse("\"test\""));

        // Single quotes not allowed
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("'test'"));
        // Should throw ParseException
        assertNotNull(e.getMessage());

        // No quotes not allowed
        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("test"));
        // Should throw ParseException
        assertNotNull(e.getMessage());
    }

    @Test
    void testStringControlCharactersMustBeEscaped() {
        // RFC 8259: Control characters (U+0000 through U+001F) must be escaped
        for (int i = 0x00; i <= 0x1F; i++) {
            char c = (char) i;
            String json = "\"" + c + "\"";
            JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse(json));
            assertTrue(e.getMessage().contains("Control") || e.getMessage().contains("Unexpected"));
        }

        // U+0020 (space) and above don't need escaping
        assertEquals(" ", JSON.parse("\" \""));
        assertEquals("!", JSON.parse("\"!\""));
    }

    @Test
    void testStringQuoteAndBackslashMustBeEscaped() {
        // " and \ must be escaped
        assertEquals("\"", JSON.parse("\"\\\"\""));
        assertEquals("\\", JSON.parse("\"\\\\\""));

        // Unescaped " should fail
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("\"test\"test\""));
        // Should throw ParseException
        assertNotNull(e.getMessage());

        // "\test" - \t is a valid escape (tab), so this parses as "test<tab>est"
        // To test invalid escape, use \ followed by invalid character
        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("\"test\\xtest\""));
        // Should throw ParseException for invalid escape
        assertNotNull(e.getMessage());
    }

    @Test
    void testStringEscapeSequences() {
        // RFC 8259: \" \\ \/ \b \f \n \r \t
        assertEquals("\"", JSON.parse("\"\\\"\""));
        assertEquals("\\", JSON.parse("\"\\\\\""));
        assertEquals("/", JSON.parse("\"\\/\""));
        assertEquals("\b", JSON.parse("\"\\b\""));
        assertEquals("\f", JSON.parse("\"\\f\""));
        assertEquals("\n", JSON.parse("\"\\n\""));
        assertEquals("\r", JSON.parse("\"\\r\""));
        assertEquals("\t", JSON.parse("\"\\t\""));

        // Forward slash may be escaped but doesn't need to be
        assertEquals("/", JSON.parse("\"/\""));
    }

    @Test
    void testStringUnicodeEscape() {
        // RFC 8259: \\uXXXX where X is hex digit
        assertEquals("A", JSON.parse("\"\\u0041\""));
        assertEquals("\u00E9", JSON.parse("\"\\u00E9\""));
        assertEquals("\u03A9", JSON.parse("\"\\u03A9\""));

        // Must be exactly 4 hex digits
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("\"\\u041\""));
        assertTrue(e.getMessage().contains("hex") || e.getMessage().contains("Unexpected"));

        // "\u00410" - after \u0041 (A), there's a '0' character which is valid
        // So this should parse as "A0", not fail
        assertEquals("A0", JSON.parse("\"\\u00410\""));

        // Hex digits must be valid
        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("\"\\u00G0\""));
        assertTrue(e.getMessage().contains("hex") || e.getMessage().contains("Invalid"));
    }

    @Test
    void testStringSurrogatePairs() {
        // RFC 8259: UTF-16 surrogate pairs must be properly encoded
        // Valid surrogate pair
        assertEquals("\uD83D\uDE00", JSON.parse("\"\\uD83D\\uDE00\""));

        // Lone high surrogate
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("\"\\uD83D\""));
        assertTrue(e.getMessage().contains("surrogate"));

        // Lone low surrogate
        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("\"\\uDE00\""));
        assertTrue(e.getMessage().contains("surrogate"));

        // Reversed surrogate pair
        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("\"\\uDE00\\uD83D\""));
        assertTrue(e.getMessage().contains("surrogate") || e.getMessage().contains("Invalid"));
    }

    @Test
    void testStringDirectUnicodeOutput() {
        // RFC 8259: Non-ASCII characters may be output directly
        // (not required to escape as \\uXXXX)
        // Test with various Unicode characters
        String unicode = "\u00E9\u03A9\u4E16\u754C\uD83D\uDE00";
        String json = JSON.stringify(unicode, false);
        // Should output direct Unicode, not escaped
        assertEquals("\"" + unicode + "\"", json);

        // Round trip
        assertEquals(unicode, JSON.parse(json));
    }

    // ===== RFC 8259 Section 4: Objects =====

    @Test
    void testObjectStructure() {
        // RFC 8259: object = begin-object [ member *( value-separator member ) ] end-object
        assertEquals(Collections.emptyMap(), JSON.parse("{}"));

        Map<String, Object> expected = new LinkedHashMap<>();
        expected.put("a", 1L);
        expected.put("b", 2L);
        assertEquals(expected, JSON.parse("{\"a\":1,\"b\":2}"));

        // No trailing comma
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("{\"a\":1,}"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("Expected"));

        // No extra comma
        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("{\"a\":1,,}"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("Expected"));
    }

    @Test
    void testObjectMember() {
        // RFC 8259: member = string name-separator value
        assertEquals(Map.of("key", "value"), JSON.parse("{\"key\":\"value\"}"));

        // Key must be string
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("{key:\"value\"}"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("Expected"));

        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("{1:\"value\"}"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("Expected"));

        // Colon required
        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("{\"key\" \"value\"}"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("Expected"));
    }

    @Test
    void testObjectDuplicateNames() {
        // RFC 8259: "The names within an object SHOULD be unique."
        // But: "JSON parsers are free to accept and ignore duplicate names."
        // We implement "last wins" which is acceptable
        Map<String, Object> parsed = (Map<String, Object>) JSON.parse("{\"a\":1,\"a\":2}");
        assertEquals(1, parsed.size());
        assertEquals("2", parsed.get("a").toString()); // Last wins
    }

    // ===== RFC 8259 Section 5: Arrays =====

    @Test
    void testArrayStructure() {
        // RFC 8259: array = begin-array [ value *( value-separator value ) ] end-array
        assertEquals(Collections.emptyList(), JSON.parse("[]"));

        assertEquals(List.of(1L, 2L, 3L), JSON.parse("[1,2,3]"));

        // No trailing comma
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("[1,]"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("Expected"));

        // No extra comma
        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("[1,,2]"));
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
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse(withBOM));
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

        Object parsed = JSON.parse(example);
        assertNotNull(parsed);
        assertTrue(parsed instanceof Map);

        // Should be able to stringify it back
        String stringified = JSON.stringify(parsed, false);
        Object reparsed = JSON.parse(stringified);
        // Compare structure (may have different number types)
        assertTrue(reparsed instanceof Map);
    }

    @Test
    void testInvalidJSONExamples() {
        // Examples of invalid JSON per RFC 8259

        // Unclosed array
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse("[1,2"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("Expected"));

        // Unclosed object
        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("{\"a\":1"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("Expected"));

        // Missing comma
        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("[1 2]"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("Expected"));

        // Extra comma in array
        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("[1,2,]"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("Expected"));

        // Extra comma in object
        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("{\"a\":1,}"));
        assertTrue(e.getMessage().contains("Unexpected") || e.getMessage().contains("Expected"));
    }

    @Test
    void testJSONTextMustBeObjectOrArray() {
        // RFC 8259: "A JSON text is a serialized value."
        // It can be any JSON value, not just object or array
        assertEquals(1L, JSON.parse("1")); // Number is valid JSON text
        assertEquals("test", JSON.parse("\"test\"")); // String is valid JSON text
        assertEquals(true, JSON.parse("true")); // Boolean is valid JSON text
        assertSame(JSON.NULL, JSON.parse("null")); // Null is valid JSON text

        // Empty string is not valid JSON
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse(""));
        // Should throw ParseException
        assertNotNull(e.getMessage());

        // Just whitespace is not valid JSON
        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("   "));
        // Should throw ParseException
        assertNotNull(e.getMessage());
    }
}
