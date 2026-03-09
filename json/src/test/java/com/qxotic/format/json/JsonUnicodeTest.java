package com.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class JsonUnicodeTest {

    @Test
    void testBMPCharacter() {
        assertEquals("A", Json.parse("\"\\u0041\""));
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
    void testSupplementaryPlaneEmoji() {
        assertEquals("\uD83D\uDE00", Json.parse("\"\\uD83D\\uDE00\""));
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
    void testMultipleSurrogatePairs() {
        String expected = "\uD83D\uDE00\uD83D\uDE01\uD83D\uDE02";
        assertEquals(expected, Json.parse("\"\\uD83D\\uDE00\\uD83D\\uDE01\\uD83D\\uDE02\""));
    }

    @Test
    void testHighSurrogateFollowedByAnotherHighSurrogate() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\uD83D\\uD83D\""));
        assertTrue(e.getMessage().contains("Unexpected"));
    }

    @Test
    void testInvalidSurrogateSequence() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\uD800\\uDBFF\""));
        assertTrue(e.getMessage().contains("Unexpected"));
    }

    @Test
    void testReversedSurrogateOrder() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\uDE00\\uD83D\""));
        assertTrue(e.getMessage().contains("Lone"));
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
}
