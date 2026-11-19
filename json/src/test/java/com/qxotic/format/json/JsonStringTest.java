package com.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;

/** String parsing, escapes and Unicode handling. */
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

    /** A surrogate escape without its pair is that char: RFC 8259 grammar, and what we print. */
    @Test
    void aLoneSurrogateEscapeIsThatChar() {
        String[][] cases = {
            {"\"\\uD800\"", "\uD800"},
            {"\"\\uDC00\"", "\uDC00"},
            {"\"\\uD83D\\u0041\"", "\uD83DA"},
            {"\"\\uD83D\\uD83D\"", "\uD83D\uD83D"},
            {"\"\\uD800\\uDBFF\"", "\uD800\uDBFF"},
            {"\"\\uDE00\\uD83D\"", "\uDE00\uD83D"},
        };
        for (String[] c : cases) assertEquals(c[1], Json.parse(c[0]), c[0]);
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
        // a lone surrogate escape reads back as that char, wherever it stands
        assertEquals("\uD800", Json.unescapeString("\\" + "uD800"));
        assertEquals("\uDC00", Json.unescapeString("\\" + "uDC00"));
        assertEquals("\uD800\n", Json.unescapeString("\\" + "uD800\\n"));
        assertEquals("\uD83DA", Json.unescapeString("\\" + "uD83D\\u0041"));
        assertEquals("\uDE00\uD83D", Json.unescapeString("\\" + "uDE00\\" + "uD83D"));
        // High surrogate followed by backslash-u with invalid hex in low
        Json.ParseException ex5 =
                assertThrows(
                        Json.ParseException.class,
                        () -> Json.unescapeString("\\" + "uD83D" + "\\" + "uGGGG"));
        assertTrue(ex5.getMessage().contains("Invalid hex digit"));
    }

    @Test
    void testUnescapeStringInvalidEscapes() {
        // Trailing backslash
        Json.ParseException e1 =
                assertThrows(Json.ParseException.class, () -> Json.unescapeString("\\"));
        assertTrue(e1.getMessage().contains("Invalid escape"));
        // Unknown escape character
        Json.ParseException e2 =
                assertThrows(Json.ParseException.class, () -> Json.unescapeString("\\z"));
        assertTrue(e2.getMessage().contains("Invalid escape"));
        Json.ParseException e3 =
                assertThrows(Json.ParseException.class, () -> Json.unescapeString("\\x"));
        assertTrue(e3.getMessage().contains("Invalid escape"));
        // Incomplete Unicode (too short)
        Json.ParseException e4 =
                assertThrows(Json.ParseException.class, () -> Json.unescapeString("\\" + "u00"));
        assertTrue(e4.getMessage().contains("Incomplete Unicode"));
        // Invalid hex digit
        Json.ParseException e5 =
                assertThrows(Json.ParseException.class, () -> Json.unescapeString("\\" + "u00GH"));
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
        Json.ParseException e1 =
                assertThrows(Json.ParseException.class, () -> Json.unescapeString("\\" + "u"));
        assertTrue(e1.getMessage().contains("Incomplete Unicode"));
        // 1 hex digit
        Json.ParseException e2 =
                assertThrows(Json.ParseException.class, () -> Json.unescapeString("\\" + "u0"));
        assertTrue(e2.getMessage().contains("Incomplete Unicode"));
        // 2 hex digits
        Json.ParseException e3 =
                assertThrows(Json.ParseException.class, () -> Json.unescapeString("\\" + "u00"));
        assertTrue(e3.getMessage().contains("Incomplete Unicode"));
        // 3 hex digits
        Json.ParseException e4 =
                assertThrows(Json.ParseException.class, () -> Json.unescapeString("\\" + "u000"));
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

    @Test
    void testBMPChineseCharacter() {
        assertEquals("中", Json.parse("\"\\u4e2d\""));
    }

    @Test
    void testBMPJapaneseCharacter() {
        assertEquals("日", Json.parse("\"\\u65e5\""));
    }

    @Test
    void testBMPCyrillicCharacter() {
        assertEquals("Я", Json.parse("\"\\u042F\""));
    }

    @Test
    void testBMPGreekCharacter() {
        assertEquals("Ω", Json.parse("\"\\u03A9\""));
    }

    @Test
    void testBMPArabicCharacter() {
        assertEquals("ع", Json.parse("\"\\u0639\""));
    }

    @Test
    void testBMPHebrewCharacter() {
        assertEquals("א", Json.parse("\"\\u05D0\""));
    }

    @Test
    void testBMPThaiCharacter() {
        assertEquals("ก", Json.parse("\"\\u0E01\""));
    }

    @Test
    void testSupplementaryPlaneSmilingFace() {
        assertEquals("\uD83D\uDE42", Json.parse("\"\\uD83D\\uDE42\""));
    }

    @Test
    void testSupplementaryPlaneThumbsUp() {
        assertEquals("\uD83D\uDC4D", Json.parse("\"\\uD83D\\uDC4D\""));
    }

    @Test
    void testSupplementaryPlaneFlag() {
        assertEquals("\uD83C\uDDFA\uD83C\uDDF8", Json.parse("\"\\uD83C\\uDDFA\\uD83C\\uDDF8\""));
    }

    @Test
    void testSupplementaryPlaneMusicalNote() {
        assertEquals("\uD83C\uDFB5", Json.parse("\"\\uD83C\\uDFB5\""));
    }

    @Test
    void testMixedBMPAndSupplementary() {
        String expected = "A\uD83D\uDE00Z";
        assertEquals(expected, Json.parse("\"A\\uD83D\\uDE00Z\""));
    }

    @Test
    void testUnicodeWithText() {
        String expected = "Hello 世界";
        assertEquals(expected, Json.parse("\"Hello \\u4e16\\u754c\""));
    }

    @Test
    void testUnicodeEmojiWithText() {
        String expected = "Hi 😀";
        assertEquals(expected, Json.parse("\"Hi \\uD83D\\uDE00\""));
    }

    @Test
    void testSurrogatePairInMiddleOfString() {
        String expected = "Hello\uD83D\uDE00World";
        assertEquals(expected, Json.parse("\"Hello\\uD83D\\uDE00World\""));
    }

    @Test
    void testMultipleSurrogatePairsUnicode() {
        String expected = "\uD83D\uDE00\uD83D\uDE01\uD83D\uDE02";
        assertEquals(expected, Json.parse("\"\\uD83D\\uDE00\\uD83D\\uDE01\\uD83D\\uDE02\""));
    }

    @Test
    void testMaximumCodePoint() {
        String expected = "\uDBFF\uDFFF";
        assertEquals(expected, Json.parse("\"\\uDBFF\\uDFFF\""));
    }

    @Test
    void testMinimumSurrogatePair() {
        String expected = "\uD800\uDC00";
        assertEquals(expected, Json.parse("\"\\uD800\\uDC00\""));
    }

    @Test
    void testZeroWidthJoiner() {
        assertEquals("\u200D", Json.parse("\"\\u200D\""));
    }

    @Test
    void testVariationSelector() {
        assertEquals("\uFE0F", Json.parse("\"\\uFE0F\""));
    }

    @Test
    void testCJKUnifiedIdeographs() {
        assertEquals("\u4E00", Json.parse("\"\\u4E00\""));
        assertEquals("\u9FA5", Json.parse("\"\\u9FA5\""));
    }

    // ===== Unicode boundary tests =====

    @Test
    void testControlCharacterRangeBoundaries() {
        // Test boundaries of control character range (U+0000 to U+001F)

        // U+0000 (null) - must be escaped
        assertEquals("\u0000", Json.parse("\"\\u0000\""));

        // U+001F (unit separator) - must be escaped
        assertEquals("\u001F", Json.parse("\"\\u001F\""));

        // U+0020 (space) - does NOT need to be escaped
        assertEquals(" ", Json.parse("\" \""));

        // U+007F (DEL) - does NOT need to be escaped (outside control range)
        String del = new String(new char[] {'\u007F'});
        assertEquals(del, Json.parse("\"" + del + "\""));
    }

    @Test
    void testAllUnicodeCategories() {
        // Test characters from various Unicode categories

        // Letter categories
        assertEquals("A", Json.parse("\"A\"")); // Lu: Uppercase letter
        assertEquals("a", Json.parse("\"a\"")); // Ll: Lowercase letter
        assertEquals("À", Json.parse("\"À\"")); // Lt: Titlecase letter
        assertEquals("ƻ", Json.parse("\"ƻ\"")); // Lm: Modifier letter
        assertEquals("ʰ", Json.parse("\"ʰ\"")); // Lo: Other letter

        // Mark categories
        assertEquals("̃", Json.parse("\"̃\"")); // Mn: Non-spacing mark
        assertEquals("̈", Json.parse("\"̈\"")); // Mc: Spacing combining mark
        assertEquals("゙", Json.parse("\"゙\"")); // Me: Enclosing mark

        // Number categories
        assertEquals("1", Json.parse("\"1\"")); // Nd: Decimal digit number
        assertEquals("Ⅳ", Json.parse("\"Ⅳ\"")); // Nl: Letter number
        assertEquals("½", Json.parse("\"½\"")); // No: Other number

        // Punctuation categories
        assertEquals(".", Json.parse("\".\"")); // Po: Other punctuation
        assertEquals("(", Json.parse("\"(\"")); // Ps: Open punctuation
        assertEquals(")", Json.parse("\")\"")); // Pe: Close punctuation
        assertEquals("_", Json.parse("\"_\"")); // Pc: Connector punctuation
        assertEquals("+", Json.parse("\"+\"")); // Pd: Dash punctuation
        assertEquals("<", Json.parse("\"<\"")); // Pi: Initial punctuation
        assertEquals(">", Json.parse("\">\"")); // Pf: Final punctuation

        // Symbol categories
        assertEquals("$", Json.parse("\"$\"")); // Sc: Currency symbol
        assertEquals("^", Json.parse("\"^\"")); // Sk: Modifier symbol
        assertEquals("+", Json.parse("\"+\"")); // Sm: Math symbol
        assertEquals("©", Json.parse("\"©\"")); // So: Other symbol

        // Separator categories
        assertEquals(" ", Json.parse("\" \"")); // Zs: Space separator
        assertEquals("\u2028", Json.parse("\"\\u2028\"")); // Zl: Line separator
        assertEquals("\u2029", Json.parse("\"\\u2029\"")); // Zp: Paragraph separator

        // Other categories
        assertEquals("\u0000", Json.parse("\"\\u0000\"")); // Cc: Control character (escaped)
        assertEquals("\u0080", Json.parse("\"\\u0080\"")); // Co: Private use (outside BMP)
    }

    @Test
    void testUnicodeNonCharacters() {
        // Non-characters are allowed in JSON strings
        // U+FDD0 to U+FDEF are permanently reserved non-characters
        for (int i = 0xFDD0; i <= 0xFDEF; i++) {
            char c = (char) i;
            String json = "\"\\u" + String.format("%04X", i) + "\"";
            assertEquals(String.valueOf(c), Json.parse(json));
        }

        // U+FFFE and U+FFFF are also non-characters
        assertEquals("\uFFFE", Json.parse("\"\\uFFFE\""));
        assertEquals("\uFFFF", Json.parse("\"\\uFFFF\""));

        // Last two code points of each plane (U+__FFFE and U+__FFFF)
        // Test Supplementary Plane A (U+10FFFE, U+10FFFF)
        String highNonChar1 = "\uDBFF\uDFFE"; // U+10FFFE
        String highNonChar2 = "\uDBFF\uDFFF"; // U+10FFFF
        assertEquals(highNonChar1, Json.parse("\"\\uDBFF\\uDFFE\""));
        assertEquals(highNonChar2, Json.parse("\"\\uDBFF\\uDFFF\""));
    }

    @Test
    void testUnicodePlaneBoundaries() {
        // Test boundaries between Unicode planes

        // Last BMP character (U+FFFF) is non-character, but valid in string
        assertEquals("\uFFFF", Json.parse("\"\\uFFFF\""));

        // First Supplementary Plane character (U+10000)
        String firstSupplementary = "\uD800\uDC00"; // U+10000
        assertEquals(firstSupplementary, Json.parse("\"\\uD800\\uDC00\""));

        // Last valid Unicode character (U+10FFFF)
        String lastValid = "\uDBFF\uDFFF"; // U+10FFFF (non-character but valid)
        assertEquals(lastValid, Json.parse("\"\\uDBFF\\uDFFF\""));

        // Test various planes
        // Plane 1: Supplementary Multilingual Plane (U+10000 to U+1FFFF)
        String plane1Char = "\uD800\uDFFF"; // U+103FF example
        assertEquals(plane1Char, Json.parse("\"\\uD800\\uDFFF\""));

        // Plane 2: Supplementary Ideographic Plane (U+20000 to U+2FFFF)
        String plane2Char = "\uD840\uDC00"; // U+20000
        assertEquals(plane2Char, Json.parse("\"\\uD840\\uDC00\""));

        // Plane 14: Supplementary Special-purpose Plane (U+E0000 to U+EFFFF)
        String plane14Char = "\uDB40\uDC00"; // U+E0000
        assertEquals(plane14Char, Json.parse("\"\\uDB40\\uDC00\""));
    }

    @Test
    void testUnicodeNormalizationForms() {
        // Test that JSON parser handles Unicode normalization forms correctly
        // Note: JSON does NOT require normalization, but should preserve input

        // Decomposed form (NFD)
        String decomposed = "e\u0301"; // e + combining acute accent
        assertEquals(decomposed, Json.parse("\"e\\u0301\""));

        // Composed form (NFC)
        String composed = "é"; // precomposed é
        assertEquals(composed, Json.parse("\"é\""));
        assertEquals(composed, Json.parse("\"\\u00E9\""));

        // Both should parse as given (no normalization applied)
        assertNotEquals(decomposed, composed); // Different strings
        assertNotEquals(Json.parse("\"e\\u0301\""), Json.parse("\"\\u00E9\""));
    }

    @Test
    void testUnicodeCombiningCharacters() {
        // Test strings with multiple combining characters

        // Base character with multiple combining marks
        String complex = "g\u0303\u0327"; // g + tilde + cedilla
        assertEquals(complex, Json.parse("\"g\\u0303\\u0327\""));

        // Emoji with variation selector
        String emojiVS = "\uD83D\uDE00\uFE0F"; // 😀 + variation selector
        assertEquals(emojiVS, Json.parse("\"\\uD83D\\uDE00\\uFE0F\""));

        // Zero-width joiner sequences
        String zwjSequence = "\uD83D\uDC68\u200D\uD83D\uDC69\u200D\uD83D\uDC67"; // 👨‍👩‍👧
        assertEquals(
                zwjSequence,
                Json.parse("\"\\uD83D\\uDC68\\u200D\\uD83D\\uDC69\\u200D\\uD83D\\uDC67\""));
    }

    @Test
    void testFullUnicodeRangeSampling() {
        // Sample various code points across Unicode range
        int[] samplePoints = {
            0x0000, 0x0001, 0x001F, 0x0020, 0x007F, 0x0080, 0x00FF, 0x0100, 0x07FF, 0x0800, 0x0FFF,
            0x1000, 0x1FFF, 0x2000, 0x2FFF, 0x3000, 0x3FFF, 0x4000, 0x4FFF, 0x5000, 0x5FFF, 0x6000,
            0x6FFF, 0x7000, 0x7FFF, 0x8000, 0x8FFF, 0x9000, 0x9FFF, 0xA000, 0xAFFF, 0xB000, 0xBFFF,
            0xC000, 0xCFFF, 0xD000, 0xDFFF, 0xE000, 0xEFFF, 0xF000, 0xFFFF, 0x10000, 0x1FFFF,
            0x20000, 0x2FFFF, 0x30000, 0x3FFFF, 0x40000, 0x4FFFF, 0x50000, 0x5FFFF, 0x60000,
            0x6FFFF, 0x70000, 0x7FFFF, 0x80000, 0x8FFFF, 0x90000, 0x9FFFF, 0xA0000, 0xAFFFF,
            0xB0000, 0xBFFFF, 0xC0000, 0xCFFFF, 0xD0000, 0xDFFFF, 0xE0000, 0xEFFFF, 0xF0000,
            0xFFFFF, 0x100000, 0x10FFFF
        };

        for (int codePoint : samplePoints) {
            if (codePoint <= 0xFFFF) {
                // BMP character
                if (!Character.isSurrogate((char) codePoint)) {
                    char c = (char) codePoint;
                    String json = "\"\\u" + String.format("%04X", codePoint) + "\"";

                    // Skip control characters that must be escaped (we're testing escaped version)
                    if (codePoint >= 0x0000 && codePoint <= 0x001F) {
                        // Control character - test escaped version
                        assertEquals(String.valueOf(c), Json.parse(json));
                    } else {
                        // Non-control character - test both escaped and direct
                        assertEquals(String.valueOf(c), Json.parse(json));
                        if (codePoint != 0x0022 && codePoint != 0x005C) { // Not quote or backslash
                            assertEquals(String.valueOf(c), Json.parse("\"" + c + "\""));
                        }
                    }
                }
            } else {
                // Supplementary plane character
                char[] chars = Character.toChars(codePoint);
                String json =
                        "\"\\u"
                                + String.format("%04X", (int) chars[0])
                                + "\\u"
                                + String.format("%04X", (int) chars[1])
                                + "\"";
                assertEquals(new String(chars), Json.parse(json));
            }
        }
    }

    @Test
    void testUnicodeReplacementCharacter() {
        // Unicode replacement character U+FFFD
        String replacement = "\uFFFD";
        assertEquals(replacement, Json.parse("\"\\uFFFD\""));
        assertEquals(replacement, Json.parse("\"" + replacement + "\""));

        // Used in error contexts
        assertEquals("Parse error: " + replacement, Json.parse("\"Parse error: \\uFFFD\""));
    }

    @Test
    void testBidirectionalText() {
        // Test bidirectional text handling
        String mixedDirection = "Hello שלום";
        assertEquals(mixedDirection, Json.parse("\"Hello \\u05E9\\u05DC\\u05D5\\u05DD\""));

        // Right-to-left override
        String rlo = "\u202E"; // RIGHT-TO-LEFT OVERRIDE
        assertEquals(rlo, Json.parse("\"\\u202E\""));

        // Left-to-right override
        String lro = "\u202D"; // LEFT-TO-RIGHT OVERRIDE
        assertEquals(lro, Json.parse("\"\\u202D\""));
    }

    @Test
    void testUnicodeEscapeSequencesInStrings() {
        // Test that Unicode escapes work correctly within strings

        // Multiple escapes in one string
        assertEquals("ABC", Json.parse("\"\\u0041\\u0042\\u0043\""));

        // Mixed escaped and literal
        assertEquals("A\u0042C", Json.parse("\"A\\u0042C\""));

        // Escape sequences that form valid UTF-16
        assertEquals("\u0041\u0042\u0043", Json.parse("\"\\u0041\\u0042\\u0043\""));

        // Supplementary plane via escapes
        assertEquals("\uD83D\uDE00", Json.parse("\"\\uD83D\\uDE00\""));
    }

    /** A lone surrogate is escaped wherever it stands; a pair, in order, is kept. */
    @Test
    void escapeStringEscapesEveryLoneSurrogate() {
        assertEquals("a\\uD800", Json.escapeString("a\uD800"), "a high surrogate at the end");
        assertEquals("\\uDC00b", Json.escapeString("\uDC00b"), "a low surrogate alone");
        assertEquals("\\uDE00\\uD83D", Json.escapeString("\uDE00\uD83D"), "a pair reversed");
        assertEquals("\\uD83Dx", Json.escapeString("\uD83Dx"), "a high surrogate before a letter");
        assertEquals("😀\\uD83D", Json.escapeString("😀\uD83D"), "a pair, then a lone high");
    }

    /** What the printer writes for a lone surrogate, the parser reads back. */
    @Test
    void aLoneSurrogateRoundTrips() {
        for (String text : new String[] {"\uD800", "a\uDC00b", "\uDE00\uD83D", "😀\uD83D"}) {
            assertEquals(text, Json.unescapeString(Json.escapeString(text)));
            assertEquals(text, Json.parse(Json.stringify(text)));
        }
    }

    /** Escapes first, last, back to back and between text, from a String and any CharSequence. */
    @Test
    void unescapeStringDecodesEscapesAnywhere() {
        String[][] cases = {
            {"\\nabc", "\nabc"},
            {"abc\\n", "abc\n"},
            {"\\n\\t\\\"", "\n\t\""},
            {"a\\nb\\tc", "a\nb\tc"},
            {"\\u0041bc\\u0044", "AbcD"},
        };
        for (String[] c : cases) {
            assertEquals(c[1], Json.unescapeString(c[0]), c[0]);
            assertEquals(c[1], Json.unescapeString(new StringBuilder(c[0])), c[0] + ", a builder");
        }
    }

    /** unescapeString decodes escapes and keeps everything else, raw quotes and controls too. */
    @Test
    void unescapeStringKeepsWhatIsNotAnEscape() {
        assertEquals("a\"b", Json.unescapeString("a\"b"));
        assertEquals("a\nb\u0000", Json.unescapeString("a\nb\u0000"));
        assertEquals("a\"b\n", Json.unescapeString("a\"b\\n"));
    }

    /**
     * Every BMP character but the surrogates, and seeded random text, round-trips through
     * escapeString, then unescapeString and the parser alike.
     */
    @Test
    void everyCharacterRoundTrips() {
        StringBuilder all = new StringBuilder();
        for (int c = 0; c <= 0xFFFF; c++) {
            if (!Character.isSurrogate((char) c)) {
                all.append((char) c);
            }
        }
        List<String> texts = new ArrayList<>(List.of(all.toString()));
        Random random = new Random(42);
        for (int n = 0; n < 1000; n++) {
            StringBuilder text = new StringBuilder();
            for (int i = random.nextInt(40); i > 0; i--) {
                text.appendCodePoint(randomCodePoint(random));
            }
            texts.add(text.toString());
        }
        for (String text : texts) {
            String escaped = Json.escapeString(text);
            assertEquals(text, Json.unescapeString(escaped));
            assertEquals(text, Json.parse("\"" + escaped + "\""));
        }
    }

    /** ASCII with its controls and quotes, any other BMP character, or a supplementary one. */
    private static int randomCodePoint(Random random) {
        switch (random.nextInt(3)) {
            case 0:
                return random.nextInt(0x80);
            case 1:
                int c = random.nextInt(0xF800); // the BMP less its 0x800 surrogates
                return c < 0xD800 ? c : c + 0x800;
            default:
                return 0x10000 + random.nextInt(0x100000);
        }
    }
}
