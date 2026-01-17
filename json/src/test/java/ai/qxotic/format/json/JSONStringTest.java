package ai.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class JSONStringTest {

    @Test
    void testEmptyString() {
        assertEquals("", JSON.parse("\"\""));
    }

    @Test
    void testSimpleString() {
        assertEquals("hello", JSON.parse("\"hello\""));
    }

    @Test
    void testEscapedQuote() {
        assertEquals("\"quoted\"", JSON.parse("\"\\\"quoted\\\"\""));
    }

    @Test
    void testEscapedBackslash() {
        assertEquals("\\", JSON.parse("\"\\\\\""));
    }

    @Test
    void testEscapedForwardSlash() {
        assertEquals("/", JSON.parse("\"\\/\""));
    }

    @Test
    void testEscapedBackspace() {
        assertEquals("\b", JSON.parse("\"\\b\""));
    }

    @Test
    void testEscapedFormFeed() {
        assertEquals("\f", JSON.parse("\"\\f\""));
    }

    @Test
    void testEscapedNewline() {
        assertEquals("\n", JSON.parse("\"\\n\""));
    }

    @Test
    void testEscapedCarriageReturn() {
        assertEquals("\r", JSON.parse("\"\\r\""));
    }

    @Test
    void testEscapedTab() {
        assertEquals("\t", JSON.parse("\"\\t\""));
    }

    @Test
    void testEscapedAscii() {
        assertEquals("A", JSON.parse("\"\\u0041\""));
    }

    @Test
    void testStringWithSpaces() {
        assertEquals("hello world", JSON.parse("\"hello world\""));
    }

    @Test
    void testStringWithSpecialChars() {
        assertEquals("!@#$%^&*()", JSON.parse("\"!@#$%^&*()\""));
    }

    @Test
    void testUnescapedControlCharNULRejected() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("\"\u0000\""));
        assertTrue(e.getMessage().contains("Control character"));
    }

    @Test
    void testControlCharsMustBeEscaped() {
        for (int cp = 0x00; cp <= 0x1F; cp++) {
            String json = "\"" + (char) cp + "\"";
            JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse(json));
            assertTrue(
                    e.getMessage().contains("Control character"), "U+" + Integer.toHexString(cp));
        }
    }

    @Test
    void testValidUnescapedChars() {
        assertEquals(" ", JSON.parse("\" \""));
        assertEquals("!", JSON.parse("\"!\""));
        assertEquals("#", JSON.parse("\"#\""));
        assertEquals("$", JSON.parse("\"$\""));
        assertEquals("%", JSON.parse("\"%\""));
        assertEquals("&", JSON.parse("\"&\""));
        assertEquals("'", JSON.parse("\"'\""));
        assertEquals("(", JSON.parse("\"(\""));
        assertEquals(")", JSON.parse("\")\""));
        assertEquals("*", JSON.parse("\"*\""));
        assertEquals("+", JSON.parse("\"+\""));
        assertEquals(",", JSON.parse("\",\""));
        assertEquals("-", JSON.parse("\"-\""));
        assertEquals(".", JSON.parse("\".\""));
        assertEquals("/", JSON.parse("\"/\""));
    }

    @Test
    void testColonInString() {
        assertEquals("key:value", JSON.parse("\"key:value\""));
    }

    @Test
    void testCommaInString() {
        assertEquals("a,b,c", JSON.parse("\"a,b,c\""));
    }

    @Test
    void testBracketsInString() {
        assertEquals("[]{}", JSON.parse("\"[]{}\""));
    }

    @Test
    void testUnicodeChinese() {
        assertEquals("中文", JSON.parse("\"\\u4e2d\\u6587\""));
    }

    @Test
    void testUnicodeJapanese() {
        assertEquals("日本語", JSON.parse("\"\\u65e5\\u672c\\u8a9e\""));
    }

    @Test
    void testUnicodeKorean() {
        assertEquals("한국어", JSON.parse("\"\\ud55c\\uad6d\\uc5b4\""));
    }

    @Test
    void testUnicodeEmoji() {
        assertEquals("\uD83D\uDE00", JSON.parse("\"\\uD83D\\uDE00\""));
    }

    @Test
    void testUnicodeMultipleCodePoints() {
        String expected = "Hello 世界 🌍";
        assertEquals(expected, JSON.parse("\"Hello \\u4e16\\u754c \\uD83C\\uDF0D\""));
    }

    @Test
    void testIncompleteUnicodeEscape() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("\"\\u\""));
        assertTrue(e.getMessage().contains("Incomplete Unicode"));
    }

    @Test
    void testInvalidHexDigit() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("\"\\uG000\""));
        assertTrue(e.getMessage().contains("Invalid hex digit"));
    }

    @Test
    void testLoneHighSurrogateRejected() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("\"\\uD800\""));
        assertTrue(e.getMessage().contains("Lone"));
    }

    @Test
    void testLoneLowSurrogateRejected() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("\"\\uDC00\""));
        assertTrue(e.getMessage().contains("Lone"));
    }

    @Test
    void testHighSurrogateFollowedByNonLowSurrogate() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("\"\\uD83D\\u0041\""));
        assertTrue(e.getMessage().contains("Unexpected"));
    }

    @Test
    void testValidSurrogatePair() {
        assertEquals("\uD83D\uDE00", JSON.parse("\"\\uD83D\\uDE00\""));
    }

    @Test
    void testMultipleSurrogatePairs() {
        String expected = "\uD83D\uDE00\uD83D\uDC4D\uD83C\uDF89";
        assertEquals(expected, JSON.parse("\"\\uD83D\\uDE00\\uD83D\\uDC4D\\uD83C\\uDF89\""));
    }

    @Test
    void testMixedContent() {
        String expected = "Hello\n\t\"world\"";
        assertEquals(expected, JSON.parse("\"Hello\\n\\t\\\"world\\\"\""));
    }

    @Test
    void testEscapedSolidus() {
        assertEquals("/", JSON.parse("\"\\/\""));
    }

    @Test
    void testStringWithLineBreakInside() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("\"line1\nline2\""));
        assertTrue(e.getMessage().contains("Control character"));
    }

    @Test
    void testStringWithCarriageReturnInside() {
        JSON.ParseException e =
                assertThrows(JSON.ParseException.class, () -> JSON.parse("\"line1\rline2\""));
        assertTrue(e.getMessage().contains("Control character"));
    }
}
