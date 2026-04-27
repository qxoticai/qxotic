package com.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * JSONTestSuite compliance tests. Tests critical edge cases identified from JSONTestSuite research.
 * Follows JSONTestSuite naming convention (y_ = must accept, n_ = must reject, i_ =
 * implementation-defined).
 */
class JsonTestSuiteComplianceTest {

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
}
