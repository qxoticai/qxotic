package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.toknroll.gguf.ModelFamilyTokenizers;
import com.qxotic.toknroll.impl.SentencePieceBpeModel;
import com.qxotic.toknroll.impl.VocabularyImpl;
import java.util.Optional;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Comprehensive unit tests for SentencePiece BPE model.
 *
 * <p>Covers:
 *
 * <ul>
 *   <li>Synthetic controlled-vocabulary tests (no network needed)
 *   <li>Real model round-trip tests for Gemma 4 and Mistral v0.3
 *   <li>Edge cases: empty strings, unicode, byte fallback, long texts
 * </ul>
 */
class SentencePieceBpeModelComprehensiveTest {

    // ------------------------------------------------------------------
    // Synthetic model tests
    // ------------------------------------------------------------------

    @Test
    void emptyStringRoundTrip() {
        SentencePieceBpeModel model = minimalModel();
        IntSequence encoded = model.encode("");
        assertEquals(0, encoded.length());
        assertEquals("", model.decode(encoded));
        assertEquals(0, model.countTokens(""));
    }

    @Test
    void singleTokenRoundTrip() {
        SentencePieceBpeModel model = minimalModel();
        IntSequence encoded = model.encode(" hello");
        assertArrayEquals(new int[] {10}, encoded.toArray());
        assertEquals(" hello", model.decode(encoded));
    }

    @Test
    void greedyMergeToHighestScore() {
        SentencePieceBpeModel model = minimalModel();
        // " hello" -> " " + "h" + "e" + "l" + "l" + "o" should merge greedily to " hello"
        IntSequence encoded = model.encode(" hello");
        assertArrayEquals(new int[] {10}, encoded.toArray());
        assertEquals(" hello", model.decode(encoded));
    }

    @Test
    void spaceRunTokenization() {
        String[] tokens = {"<0x00>", " ", "  ", "   ", "a"};
        float[] scores = {0f, 0f, 1f, 1f, 0f};
        int[] types = normalTypes(tokens.length);
        types[0] = StandardTokenType.BYTE.getId();
        SentencePieceBpeModel model =
                SentencePieceBpeModel.fromVocabulary(new VocabularyImpl(tokens, types), scores);

        assertArrayEquals(new int[] {2}, model.encode("  ").toArray());
        assertArrayEquals(new int[] {3}, model.encode("   ").toArray());
    }

    @Test
    void unicodeCharacterHandling() {
        String[] tokens = {"<0x00>", " ", "é", "l", "o", "él", "lo", "élo", " élo"};
        float[] scores = {0f, 0f, 0f, 0f, 0f, 1f, 2f, 3f, 4f};
        int[] types = normalTypes(9);
        types[0] = StandardTokenType.BYTE.getId();
        SentencePieceBpeModel model =
                SentencePieceBpeModel.fromVocabulary(new VocabularyImpl(tokens, types), scores);

        IntSequence encoded = model.encode(" élo");
        assertArrayEquals(new int[] {8}, encoded.toArray());
        assertEquals(" élo", model.decode(encoded));
    }

    @Test
    void byteFallbackForUnknownChars() {
        // Full byte vocab: <0x00> through <0xFF>
        String[] tokens = new String[257];
        float[] scores = new float[257];
        int[] types = new int[257];
        tokens[0] = "<0x00>";
        scores[0] = 0f;
        types[0] = StandardTokenType.BYTE.getId();
        for (int i = 1; i < 256; i++) {
            tokens[i] = String.format("<0x%02X>", i);
            scores[i] = 0f;
            types[i] = StandardTokenType.BYTE.getId();
        }
        tokens[256] = " ";
        scores[256] = 0f;
        types[256] = StandardTokenType.NORMAL.getId();

        SentencePieceBpeModel model =
                SentencePieceBpeModel.fromVocabulary(new VocabularyImpl(tokens, types), scores);

        IntSequence encoded = model.encode("b");
        assertEquals(1, encoded.length());
        assertEquals("b", model.decode(encoded));

        IntSequence encoded2 = model.encode("☠");
        assertTrue(encoded2.length() > 0);
        assertEquals("☠", model.decode(encoded2));
    }

    @Test
    void fastPathProducesValidResult() {
        // Large vocab to force fast merge path (threshold > 128 tokens).
        // Use single-char tokens so encoding doesn't fall back to bytes.
        int vocabSize = 300;
        String[] tokens = new String[vocabSize];
        float[] scores = new float[vocabSize];
        int[] types = new int[vocabSize];
        tokens[0] = "<0x00>";
        scores[0] = 0f;
        types[0] = StandardTokenType.BYTE.getId();
        tokens[1] = " ";
        scores[1] = 0f;
        types[1] = StandardTokenType.NORMAL.getId();

        // Fill with single printable ASCII chars starting from '!' (33)
        for (int i = 2; i < vocabSize; i++) {
            tokens[i] = String.valueOf((char) (i + 31));
            scores[i] = i;
            types[i] = StandardTokenType.NORMAL.getId();
        }

        SentencePieceBpeModel model =
                SentencePieceBpeModel.fromVocabulary(new VocabularyImpl(tokens, types), scores);

        // Build a long input that produces >128 initial tokens using single chars
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 200; i++) {
            sb.append(" ").append(tokens[(i % (vocabSize - 2)) + 2]);
        }
        String text = sb.toString();

        IntSequence encoded = model.encode(text);
        assertTrue(encoded.length() > 128, "Should trigger fast merge path");

        String decoded = model.decode(encoded);
        assertEquals(text, decoded, "Fast path round-trip failed");
    }

    @Test
    void countTokensMatchesEncodeLength() {
        SentencePieceBpeModel model = minimalModel();
        String[] texts = {"", " hello", " hello world"};
        for (String text : texts) {
            int count = model.countTokens(text);
            int length = model.encode(text).length();
            assertEquals(
                    length, count, "countTokens should match encode length for: [" + text + "]");
        }
    }

    @Test
    void expectedTokensPerCharIsPositive() {
        assertTrue(minimalModel().expectedTokensPerChar() > 0f);
    }

    // ------------------------------------------------------------------
    // Real model tests: Gemma 4
    // ------------------------------------------------------------------

    @Test
    @Tag("network")
    void gemma4BasicRoundTrips() {
        String familyId = "google.gemma4";
        Optional<Tokenizer> tokenizer = ModelFamilyTokenizers.create(familyId);
        assumeTrue(tokenizer.isPresent(), "Gemma4 tokenizer unavailable");

        String[] testCases = {
            "",
            "Hello",
            "Hello world",
            "The quick brown fox jumps over the lazy dog",
            "12345",
            "Unicode: éàùç",
            "Mixed: Hello123 éàù!",
            "Multiple   spaces   here",
        };

        for (String text : testCases) {
            IntSequence encoded = tokenizer.get().encode(text);
            String decoded = tokenizer.get().decode(encoded);
            assertEquals(text, decoded, "Round-trip failed for: [" + text + "]");
            assertEquals(
                    encoded.length(),
                    tokenizer.get().countTokens(text),
                    "countTokens mismatch for: [" + text + "]");
        }
    }

    @Test
    @Tag("network")
    void gemma4TrickyStrings() {
        String familyId = "google.gemma4";
        Optional<Tokenizer> tokenizer = ModelFamilyTokenizers.create(familyId);
        assumeTrue(tokenizer.isPresent(), "Gemma4 tokenizer unavailable");

        String[] trickyCases = {
            // Empty and whitespace
            "",
            " ",
            "  ",
            "   ",
            "\t",
            "\n",
            "\r\n",
            " \t\n ",

            // Unicode and emoji
            "Emoji: 😀🎉✨🔥",
            "Family emoji: 👨‍👩‍👧‍👦",
            "Flags: 🇺🇸🇬🇧🇯🇵",
            "Symbols: ♠♥♦♣",
            "Math: ∑∏∫√",
            "Currency: €£¥₹",

            // Accented and composed characters
            "Café",
            "Naïve",
            "Résumé",
            "Zürich",
            "Ålesund",
            "Ångström",
            "Crème brûlée",

            // Zero-width and control characters
            "Zero\u200Bwidth",
            "BOM\uFEFFtext",
            "NBSP\u00A0text",
            "Ideographic\u3000space",

            // Combining characters
            "cafe\u0301",
            "naive\u0308",
            "coop\u0308erate",

            // Mixed scripts
            "日本語テキスト",
            "中文文本",
            "한국어텍스트",
            "العربية",
            "हिन्दी",
            "Русский",
            "Ελληνικά",
            "עברית",

            // Numbers and special formats
            "3.14159",
            "1,234,567.89",
            "0xDEADBEEF",
            "2^64",
            "H₂O",
            "E=mc²",

            // URLs and code
            "https://example.com/path?query=1",
            "email@domain.com",
            "def hello():\n    return 'world'",
            "<html>\n  <body>Hello</body>\n</html>",
            "SELECT * FROM users WHERE id = 1;",

            // Edge cases
            "a",
            "A",
            "1",
            "!",
            "\"quoted\"",
            "'apostrophe'",
            "(parentheses)",
            "[brackets]",
            "{braces}",
            "<angles>",
            "path/to/file.txt",
            "C:\\Users\\name",
            "`backticks`",
            "~tilde~",
            "@mention",
            "#hashtag",
            "$dollar",
            "%percent",
            "^caret",
            "&ampersand",
            "*asterisk",
            "_underscore_",
            "+plus",
            "=equals",
            "|pipe|",
            "\\backslash",
            "/slash/",
            "?question?",
            "!exclamation!",
            ":colon:",
            ";semicolon;",
            ",comma,",
            ".period.",
            "...ellipsis...",
        };

        for (String text : trickyCases) {
            IntSequence encoded = tokenizer.get().encode(text);
            String decoded = tokenizer.get().decode(encoded);
            assertEquals(
                    java.text.Normalizer.normalize(text, java.text.Normalizer.Form.NFC),
                    decoded,
                    "Round-trip failed for tricky case: [" + text + "]");
            assertEquals(
                    encoded.length(),
                    tokenizer.get().countTokens(text),
                    "countTokens mismatch for: [" + text + "]");
            assertTrue(encoded.length() >= 0);
        }
    }

    @Test
    @Tag("network")
    void gemma4LongTextRoundTrip() {
        String familyId = "google.gemma4";
        Optional<Tokenizer> tokenizer = ModelFamilyTokenizers.create(familyId);
        assumeTrue(tokenizer.isPresent(), "Gemma4 tokenizer unavailable");

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 1000; i++) {
            sb.append("The quick brown fox jumps over the lazy dog. ");
        }
        String text = sb.toString();

        IntSequence encoded = tokenizer.get().encode(text);
        assertTrue(encoded.length() > 0);
        String decoded = tokenizer.get().decode(encoded);
        assertEquals(text, decoded);
    }

    // ------------------------------------------------------------------
    // Real model tests: Mistral v0.3 SP-BPE
    // ------------------------------------------------------------------

    @Test
    @Tag("network")
    void mistralV03SpBpeBasicRoundTrips() {
        String familyId = "mistral.v0_3_spbpe";
        Optional<Tokenizer> tokenizer = ModelFamilyTokenizers.create(familyId);
        assumeTrue(tokenizer.isPresent(), "Mistral v0.3 tokenizer unavailable");

        String[] testCases = {
            "",
            "Hello",
            "Hello world",
            "The quick brown fox jumps over the lazy dog",
            "12345",
            "Unicode: éàùç",
            "Mixed: Hello123 éàù!",
            "Multiple   spaces   here",
            "Tabs\tand\nnewlines",
        };

        for (String text : testCases) {
            IntSequence encoded = tokenizer.get().encode(text);
            String decoded = tokenizer.get().decode(encoded);
            assertEquals(text, decoded, "Round-trip failed for: [" + text + "]");
            assertEquals(
                    encoded.length(),
                    tokenizer.get().countTokens(text),
                    "countTokens mismatch for: [" + text + "]");
        }
    }

    @Test
    @Tag("network")
    void mistralV03SpBpeTrickyStrings() {
        String familyId = "mistral.v0_3_spbpe";
        Optional<Tokenizer> tokenizer = ModelFamilyTokenizers.create(familyId);
        assumeTrue(tokenizer.isPresent(), "Mistral v0.3 tokenizer unavailable");

        String[] trickyCases = {
            // Empty and whitespace
            "",
            " ",
            "  ",
            "   ",
            "\t",
            "\n",
            "\r\n",
            " \t\n ",

            // Unicode and emoji
            "Emoji: 😀🎉✨🔥",
            "Family emoji: 👨‍👩‍👧‍👦",
            "Flags: 🇺🇸🇬🇧🇯🇵",
            "Symbols: ♠♥♦♣",
            "Math: ∑∏∫√",
            "Currency: €£¥₹",

            // Accented and composed characters
            "Café",
            "Naïve",
            "Résumé",
            "Zürich",
            "Ålesund",
            "Ångström",
            "Crème brûlée",

            // Zero-width and control characters
            "Zero\u200Bwidth",
            "BOM\uFEFFtext",
            "NBSP\u00A0text",
            "Ideographic\u3000space",

            // Combining characters
            "cafe\u0301",
            "naive\u0308",
            "coop\u0308erate",

            // Mixed scripts
            "日本語テキスト",
            "中文文本",
            "한국어텍스트",
            "العربية",
            "हिन्दी",
            "Русский",
            "Ελληνικά",
            "עברית",

            // Numbers and special formats
            "3.14159",
            "1,234,567.89",
            "0xDEADBEEF",
            "2^64",
            "H₂O",
            "E=mc²",

            // URLs and code
            "https://example.com/path?query=1",
            "email@domain.com",
            "def hello():\n    return 'world'",
            "<html>\n  <body>Hello</body>\n</html>",
            "SELECT * FROM users WHERE id = 1;",

            // Edge cases
            "a",
            "A",
            "1",
            "!",
            "\"quoted\"",
            "'apostrophe'",
            "(parentheses)",
            "[brackets]",
            "{braces}",
            "<angles>",
            "path/to/file.txt",
            "C:\\Users\\name",
            "`backticks`",
            "~tilde~",
            "@mention",
            "#hashtag",
            "$dollar",
            "%percent",
            "^caret",
            "&ampersand",
            "*asterisk",
            "_underscore_",
            "+plus",
            "=equals",
            "|pipe|",
            "\\backslash",
            "/slash/",
            "?question?",
            "!exclamation!",
            ":colon:",
            ";semicolon;",
            ",comma,",
            ".period.",
            "...ellipsis...",
        };

        for (String text : trickyCases) {
            IntSequence encoded = tokenizer.get().encode(text);
            String decoded = tokenizer.get().decode(encoded);
            assertEquals(text, decoded, "Round-trip failed for tricky case: [" + text + "]");
            assertEquals(
                    encoded.length(),
                    tokenizer.get().countTokens(text),
                    "countTokens mismatch for: [" + text + "]");
            assertTrue(encoded.length() >= 0);
        }
    }

    @Test
    @Tag("network")
    void mistralV03SpBpeLongText() {
        String familyId = "mistral.v0_3_spbpe";
        Optional<Tokenizer> tokenizer = ModelFamilyTokenizers.create(familyId);
        assumeTrue(tokenizer.isPresent(), "Mistral v0.3 tokenizer unavailable");

        // Generate text that will produce >128 tokens to trigger fast path
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 5000; i++) {
            sb.append("Hello world ");
        }
        String text = sb.toString();

        IntSequence encoded = tokenizer.get().encode(text);
        assertTrue(encoded.length() > 128, "Should trigger fast merge path");
        String decoded = tokenizer.get().decode(encoded);
        assertEquals(text, decoded);
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /** Minimal model with no merge conflicts for basic tests. */
    private static SentencePieceBpeModel minimalModel() {
        String[] tokens = {
            "<0x00>",
            " ",
            "h",
            "e",
            "l",
            "o",
            "he",
            "ll",
            "llo",
            "hello",
            " hello",
            "w",
            "r",
            "d",
            "world",
            " world",
            " hello world"
        };
        float[] scores = {0f, 0f, 0f, 0f, 0f, 0f, 1f, 2f, 3f, 4f, 5f, 0f, 0f, 0f, 4f, 5f, 6f};
        int[] types = normalTypes(tokens.length);
        types[0] = StandardTokenType.BYTE.getId();
        return SentencePieceBpeModel.fromVocabulary(new VocabularyImpl(tokens, types), scores);
    }

    private static int[] normalTypes(int length) {
        int[] types = new int[length];
        for (int i = 0; i < length; i++) {
            types[i] = StandardTokenType.NORMAL.getId();
        }
        return types;
    }

    private static int[] byteTypes(int length) {
        int[] types = normalTypes(length);
        types[0] = StandardTokenType.BYTE.getId();
        return types;
    }
}
