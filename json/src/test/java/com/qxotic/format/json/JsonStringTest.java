package com.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class JsonStringTest {

    @Test
    void testEmptyString() {
        assertEquals("", Json.parse("\"\""));
    }

    @Test
    void testSimpleString() {
        assertEquals("hello", Json.parse("\"hello\""));
    }

    @Test
    void testEscapedQuote() {
        assertEquals("\"quoted\"", Json.parse("\"\\\"quoted\\\"\""));
    }

    @Test
    void testEscapedBackslash() {
        assertEquals("\\", Json.parse("\"\\\\\""));
    }

    @Test
    void testEscapedForwardSlash() {
        assertEquals("/", Json.parse("\"\\/\""));
    }

    @Test
    void testEscapedBackspace() {
        assertEquals("\b", Json.parse("\"\\b\""));
    }

    @Test
    void testEscapedFormFeed() {
        assertEquals("\f", Json.parse("\"\\f\""));
    }

    @Test
    void testEscapedNewline() {
        assertEquals("\n", Json.parse("\"\\n\""));
    }

    @Test
    void testEscapedCarriageReturn() {
        assertEquals("\r", Json.parse("\"\\r\""));
    }

    @Test
    void testEscapedTab() {
        assertEquals("\t", Json.parse("\"\\t\""));
    }

    @Test
    void testEscapedAscii() {
        assertEquals("A", Json.parse("\"\\u0041\""));
    }

    @Test
    void testStringWithSpaces() {
        assertEquals("hello world", Json.parse("\"hello world\""));
    }

    @Test
    void testStringWithSpecialChars() {
        assertEquals("!@#$%^&*()", Json.parse("\"!@#$%^&*()\""));
    }

    @Test
    void testUnescapedControlCharNULRejected() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\u0000\""));
        assertTrue(e.getMessage().contains("Control character"));
    }

    @Test
    void testControlCharsMustBeEscaped() {
        for (int cp = 0x00; cp <= 0x1F; cp++) {
            String json = "\"" + (char) cp + "\"";
            Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse(json));
            assertTrue(
                    e.getMessage().contains("Control character"), "U+" + Integer.toHexString(cp));
        }
    }

    @Test
    void testValidUnescapedChars() {
        assertEquals(" ", Json.parse("\" \""));
        assertEquals("!", Json.parse("\"!\""));
        assertEquals("#", Json.parse("\"#\""));
        assertEquals("$", Json.parse("\"$\""));
        assertEquals("%", Json.parse("\"%\""));
        assertEquals("&", Json.parse("\"&\""));
        assertEquals("'", Json.parse("\"'\""));
        assertEquals("(", Json.parse("\"(\""));
        assertEquals(")", Json.parse("\")\""));
        assertEquals("*", Json.parse("\"*\""));
        assertEquals("+", Json.parse("\"+\""));
        assertEquals(",", Json.parse("\",\""));
        assertEquals("-", Json.parse("\"-\""));
        assertEquals(".", Json.parse("\".\""));
        assertEquals("/", Json.parse("\"/\""));
    }

    @Test
    void testColonInString() {
        assertEquals("key:value", Json.parse("\"key:value\""));
    }

    @Test
    void testCommaInString() {
        assertEquals("a,b,c", Json.parse("\"a,b,c\""));
    }

    @Test
    void testBracketsInString() {
        assertEquals("[]{}", Json.parse("\"[]{}\""));
    }

    @Test
    void testUnicodeChinese() {
        assertEquals("中文", Json.parse("\"\\u4e2d\\u6587\""));
    }

    @Test
    void testUnicodeJapanese() {
        assertEquals("日本語", Json.parse("\"\\u65e5\\u672c\\u8a9e\""));
    }

    @Test
    void testUnicodeKorean() {
        assertEquals("한국어", Json.parse("\"\\ud55c\\uad6d\\uc5b4\""));
    }

    @Test
    void testUnicodeEmoji() {
        assertEquals("\uD83D\uDE00", Json.parse("\"\\uD83D\\uDE00\""));
    }

    @Test
    void testUnicodeMultipleCodePoints() {
        String expected = "Hello 世界 🌍";
        assertEquals(expected, Json.parse("\"Hello \\u4e16\\u754c \\uD83C\\uDF0D\""));
    }

    @Test
    void testIncompleteUnicodeEscape() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\u\""));
        assertTrue(e.getMessage().contains("Incomplete Unicode"));
    }

    @Test
    void testInvalidHexDigit() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\uG000\""));
        assertTrue(e.getMessage().contains("Invalid hex digit"));
    }

    @Test
    void testLoneHighSurrogateRejected() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\uD800\""));
        assertTrue(e.getMessage().contains("Lone"));
    }

    @Test
    void testLoneLowSurrogateRejected() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\uDC00\""));
        assertTrue(e.getMessage().contains("Lone"));
    }

    @Test
    void testHighSurrogateFollowedByNonLowSurrogate() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\uD83D\\u0041\""));
        assertTrue(e.getMessage().contains("Unexpected"));
    }

    @Test
    void testValidSurrogatePair() {
        assertEquals("\uD83D\uDE00", Json.parse("\"\\uD83D\\uDE00\""));
    }

    @Test
    void testMultipleSurrogatePairs() {
        String expected = "\uD83D\uDE00\uD83D\uDC4D\uD83C\uDF89";
        assertEquals(expected, Json.parse("\"\\uD83D\\uDE00\\uD83D\\uDC4D\\uD83C\\uDF89\""));
    }

    @Test
    void testMixedContent() {
        String expected = "Hello\n\t\"world\"";
        assertEquals(expected, Json.parse("\"Hello\\n\\t\\\"world\\\"\""));
    }

    @Test
    void testEscapedSolidus() {
        assertEquals("/", Json.parse("\"\\/\""));
    }

    @Test
    void testStringWithLineBreakInside() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"line1\nline2\""));
        assertTrue(e.getMessage().contains("Control character"));
    }

    @Test
    void testStringWithCarriageReturnInside() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"line1\rline2\""));
        assertTrue(e.getMessage().contains("Control character"));
    }

    // === escapeString / unescapeString ===

    /** Covers every branch in escapeFor: ", \, \b, \f, \n, \r, \t, control, DEL, no-escape */
    @Test
    void testEscapeStringAllBranches() {
        assertEquals("\\\"", Json.escapeString("\"")); // case '"'
        assertEquals("\\\\", Json.escapeString("\\")); // case '\\'
        assertEquals("\\b", Json.escapeString("\b")); // case '\b'
        assertEquals("\\f", Json.escapeString("\f")); // case '\f'
        assertEquals("\\n", Json.escapeString("\n")); // case '\n'
        assertEquals("\\r", Json.escapeString("\r")); // case '\r'
        assertEquals("\\t", Json.escapeString("\t")); // case '\t'
        assertEquals("\\u0000", Json.escapeString("\u0000")); // control ch < 0x20
        assertEquals("\\u001F", Json.escapeString("\u001F")); // control ch < 0x20
        assertEquals("\u007F", Json.escapeString("\u007F")); // DEL (0x7F) not escaped
        assertEquals("hello", Json.escapeString("hello")); // default: no escape
        assertEquals("/", Json.escapeString("/")); // default: slash not escaped
        assertEquals("", Json.escapeString("")); // empty
    }

    /**
     * Covers every branch in unescapeString switch: ", \, /, b, f, n, r, t, u (regular, high+low,
     * lone high, lone low, non-low after high), default invalid, trailing backslash, no-backslash
     * fast path
     */
    @Test
    void testUnescapeStringAllBranches() {
        assertEquals("\"", Json.unescapeString("\\\"")); // case '"'
        assertEquals("\\", Json.unescapeString("\\\\")); // case '\\'
        assertEquals("/", Json.unescapeString("\\/")); // case '/'
        assertEquals("\b", Json.unescapeString("\\b")); // case 'b'
        assertEquals("\f", Json.unescapeString("\\f")); // case 'f'
        assertEquals("\n", Json.unescapeString("\\n")); // case 'n'
        assertEquals("\r", Json.unescapeString("\\r")); // case 'r'
        assertEquals("\t", Json.unescapeString("\\t")); // case 't'
        assertEquals("A", Json.unescapeString("\\u0041")); // u: regular code point
        assertEquals("\uD83D\uDE00", Json.unescapeString("\\uD83D\\uDE00")); // u: surrogate pair
        assertEquals("plain", Json.unescapeString("plain")); // fast path: no backslash
        assertEquals("", Json.unescapeString("")); // empty
    }

    @Test
    void testUnescapeStringUnicodeEdgeCases() {
        // BMP boundary values
        assertEquals("\uFFFF", Json.unescapeString("\\uFFFF"));
        assertEquals("\uE000", Json.unescapeString("\\uE000"));
        // High surrogate alone at end of string
        IllegalArgumentException ex1 =
                assertThrows(
                        IllegalArgumentException.class, () -> Json.unescapeString("\\" + "uD800"));
        assertTrue(ex1.getMessage().contains("Lone surrogate"));
        // Low surrogate alone
        IllegalArgumentException ex2 =
                assertThrows(
                        IllegalArgumentException.class, () -> Json.unescapeString("\\" + "uDC00"));
        assertTrue(ex2.getMessage().contains("Lone surrogate"));
        // High surrogate followed by non-backslash-u
        IllegalArgumentException ex3 =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Json.unescapeString("\\" + "uD800\\n"));
        assertTrue(ex3.getMessage().contains("Lone surrogate"));
        // High surrogate followed by backslash-u but non-low-surrogate
        IllegalArgumentException ex4 =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Json.unescapeString("\\" + "uD83D\\u0041"));
        assertTrue(ex4.getMessage().contains("Unexpected character after high surrogate"));
        // High surrogate followed by backslash-u with invalid hex in low
        IllegalArgumentException ex5 =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Json.unescapeString("\\" + "uD83D" + "\\" + "uGGGG"));
        assertTrue(ex5.getMessage().contains("Invalid hex digit"));
    }

    @Test
    void testUnescapeStringInvalidEscapes() {
        // Trailing backslash
        IllegalArgumentException e1 =
                assertThrows(IllegalArgumentException.class, () -> Json.unescapeString("\\"));
        assertTrue(e1.getMessage().contains("Invalid escape"));
        // Unknown escape character
        IllegalArgumentException e2 =
                assertThrows(IllegalArgumentException.class, () -> Json.unescapeString("\\z"));
        assertTrue(e2.getMessage().contains("Invalid escape"));
        IllegalArgumentException e3 =
                assertThrows(IllegalArgumentException.class, () -> Json.unescapeString("\\x"));
        assertTrue(e3.getMessage().contains("Invalid escape"));
        // Incomplete Unicode (too short)
        IllegalArgumentException e4 =
                assertThrows(
                        IllegalArgumentException.class, () -> Json.unescapeString("\\" + "u00"));
        assertTrue(e4.getMessage().contains("Incomplete Unicode"));
        // Invalid hex digit
        IllegalArgumentException e5 =
                assertThrows(
                        IllegalArgumentException.class, () -> Json.unescapeString("\\" + "u00GH"));
        assertTrue(e5.getMessage().contains("Invalid hex digit"));
    }

    @Test
    void testUnescapeStringMixedCaseHex() {
        assertEquals("A", Json.unescapeString("\\u0041")); // uppercase
        assertEquals("a", Json.unescapeString("\\u0061")); // lowercase
        assertEquals("\u00AF", Json.unescapeString("\\u00AF")); // uppercase
        assertEquals("\u00AF", Json.unescapeString("\\u00af")); // lowercase
        assertEquals("\u00AF", Json.unescapeString("\\u00aF")); // mixed: aF
        assertEquals("\u00AF", Json.unescapeString("\\u00Af")); // mixed: Af
        assertEquals("ª", Json.unescapeString("\\u00aA")); // mixed: aA
        assertEquals("ª", Json.unescapeString("\\u00AA")); // uppercase
        assertEquals("ª", Json.unescapeString("\\u00aa")); // lowercase
    }

    @Test
    void testEscapeUnescapeRoundTrip() {
        String[] testCases = {
            "",
            "hello",
            "hello\nworld",
            "tab\there",
            "quote\"inside",
            "back\\slash",
            "forward/slash",
            "mixed\n\t\r\b\f",
            "中文",
            "\uD83D\uDE00",
            "\u0000\u001F\u007F",
            "surrogate\uD83D\uDE00pair"
        };
        for (String original : testCases) {
            String escaped = Json.escapeString(original);
            String unescaped = Json.unescapeString(escaped);
            assertEquals(original, unescaped, "Round-trip failed for: " + original);
        }
    }

    @Test
    void testEscapeStringMixedContent() {
        assertEquals(
                "Hello\\n\\t\\\"world\\\"\\b\\f\\r\\u0000",
                Json.escapeString("Hello\n\t\"world\"\b\f\r\u0000"));
    }

    @Test
    void testUnescapeStringMixedEscapes() {
        assertEquals(
                "Hello\n\t\"world\"\b\f\r\u0000",
                Json.unescapeString("Hello\\n\\t\\\"world\\\"\\b\\f\\r\\u0000"));
    }

    @Test
    void testEscapeStringWithCharSequence() {
        CharSequence cs = new StringBuilder("a\nb");
        assertEquals("a\\nb", Json.escapeString(cs));
        // no-escape fast path with non-String CharSequence
        assertEquals("hello", Json.escapeString(new StringBuilder("hello")));
    }

    @Test
    void testUnescapeStringWithCharSequence() {
        CharSequence cs = new StringBuilder("Hello\\nworld");
        assertEquals("Hello\nworld", Json.unescapeString(cs));
        // no-backslash fast path with non-String CharSequence
        assertEquals("plain", Json.unescapeString(new StringBuilder("plain")));
    }

    @Test
    void testEscapeStringAllControlChars() {
        for (int cp = 0x00; cp <= 0x1F; cp++) {
            String input = String.valueOf((char) cp);
            String expected;
            switch (cp) {
                case 0x08:
                    expected = "\\b";
                    break;
                case 0x09:
                    expected = "\\t";
                    break;
                case 0x0A:
                    expected = "\\n";
                    break;
                case 0x0C:
                    expected = "\\f";
                    break;
                case 0x0D:
                    expected = "\\r";
                    break;
                default:
                    expected = String.format("\\u%04X", cp);
                    break;
            }
            assertEquals(expected, Json.escapeString(input), "U+" + Integer.toHexString(cp));
        }
    }

    @Test
    void testEscapeStringNonAsciiNotEscaped() {
        assertEquals("\u0080", Json.escapeString("\u0080"));
        assertEquals("\u00FF", Json.escapeString("\u00FF"));
        assertEquals("中文", Json.escapeString("中文"));
        assertEquals("\uD83D\uDE00", Json.escapeString("\uD83D\uDE00"));
    }

    @Test
    void testUnescapeStringIncompleteUnicodeVariants() {
        // 0 digits after backslash-u
        IllegalArgumentException e1 =
                assertThrows(IllegalArgumentException.class, () -> Json.unescapeString("\\" + "u"));
        assertTrue(e1.getMessage().contains("Incomplete Unicode"));
        // 1 hex digit
        IllegalArgumentException e2 =
                assertThrows(
                        IllegalArgumentException.class, () -> Json.unescapeString("\\" + "u0"));
        assertTrue(e2.getMessage().contains("Incomplete Unicode"));
        // 2 hex digits
        IllegalArgumentException e3 =
                assertThrows(
                        IllegalArgumentException.class, () -> Json.unescapeString("\\" + "u00"));
        assertTrue(e3.getMessage().contains("Incomplete Unicode"));
        // 3 hex digits
        IllegalArgumentException e4 =
                assertThrows(
                        IllegalArgumentException.class, () -> Json.unescapeString("\\" + "u000"));
        assertTrue(e4.getMessage().contains("Incomplete Unicode"));
    }

    @Test
    void testEscapeStringConsecutiveEscapes() {
        assertEquals("\\b\\f\\n\\r\\t\\\"\\\\", Json.escapeString("\b\f\n\r\t\"\\"));
    }

    @Test
    void testUnescapeStringConsecutiveEscapes() {
        assertEquals("\b\f\n\r\t\"\\", Json.unescapeString("\\b\\f\\n\\r\\t\\\"\\\\"));
    }

    @Test
    void testEscapeStringNullThrows() {
        assertThrows(NullPointerException.class, () -> Json.escapeString(null));
    }

    @Test
    void testUnescapeStringNullThrows() {
        assertThrows(NullPointerException.class, () -> Json.unescapeString(null));
    }
}
