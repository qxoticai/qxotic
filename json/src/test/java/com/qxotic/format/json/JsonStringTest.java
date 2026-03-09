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
}
