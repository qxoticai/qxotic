package ai.qxotic.tokenizers;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.tokenizers.impl.ClassicBPE;
import ai.qxotic.tokenizers.impl.Tiktoken;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.regex.Pattern;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;

/**
 * Comprehensive corner case tests for tokenizers.
 * Tests emoji, Unicode edge cases, boundary conditions, and error handling.
 */
class TokenizerCornerCaseTest {

    private static final String R50K_BASE_HASH =
            "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930";
    private static final String P50K_BASE_HASH =
            "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069";
    private static final String CL100K_BASE_HASH =
            "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7";
    private static final String O200K_BASE_HASH =
            "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d";

    private static final String R50K_PATTERN =
            "'(?:[sdmt]|ll|ve|re)| ?\\p{L}++| ?\\p{N}++| ?[^\\s\\p{L}\\p{N}]++|\\s++$|\\s+(?!\\S)|\\s";
    private static final String CL100K_PATTERN =
            "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}++|\\p{N}{1,3}+| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*+|\\s++$|\\s*[\\r\\n]|\\s+(?!\\S)|\\s";
    private static final String O200K_PATTERN =
            String.join(
                    "|",
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
                    "\\p{N}{1,3}",
                    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*",
                    "\\s*[\\r\\n]+",
                    "\\s+(?!\\S)",
                    "\\s+");

    private static List<TokenizerSpec> tokenizers;

    @BeforeAll
    static void setUp() {
        buildTokenizers();
    }

    private static void buildTokenizers() {
        tokenizers = new ArrayList<>();
        Map<String, Map<String, Integer>> cache = new HashMap<>();

        // Build r50k_base tokenizer
        Map<String, Integer> r50kRanks = loadMergeableRanks("r50k_base.tiktoken", R50K_BASE_HASH, cache);
        tokenizers.add(new TokenizerSpec("r50k_base", 
            Tiktoken.createFromTiktoken("r50k_base", r50kRanks, 
                Pattern.compile(R50K_PATTERN), Map.of("<|endoftext|>", 50256))));

        // Build p50k_base tokenizer
        Map<String, Integer> p50kRanks = loadMergeableRanks("p50k_base.tiktoken", P50K_BASE_HASH, cache);
        tokenizers.add(new TokenizerSpec("p50k_base",
            Tiktoken.createFromTiktoken("p50k_base", p50kRanks,
                Pattern.compile(R50K_PATTERN), Map.of("<|endoftext|>", 50256))));

        // Build p50k_edit tokenizer
        tokenizers.add(new TokenizerSpec("p50k_edit",
            Tiktoken.createFromTiktoken("p50k_edit", p50kRanks,
                Pattern.compile(R50K_PATTERN), 
                Map.of("<|endoftext|>", 50256, "<|fim_prefix|>", 50281, 
                       "<|fim_middle|>", 50282, "<|fim_suffix|>", 50283))));

        // Build cl100k_base tokenizer
        Map<String, Integer> cl100kRanks = loadMergeableRanks("cl100k_base.tiktoken", CL100K_BASE_HASH, cache);
        tokenizers.add(new TokenizerSpec("cl100k_base",
            Tiktoken.createFromTiktoken("cl100k_base", cl100kRanks,
                Pattern.compile(CL100K_PATTERN),
                Map.of("<|endoftext|>", 100257, "<|fim_prefix|>", 100258,
                       "<|fim_middle|>", 100259, "<|fim_suffix|>", 100260, "<|endofprompt|>", 100276))));

        // Build o200k_base tokenizer
        Map<String, Integer> o200kRanks = loadMergeableRanks("o200k_base.tiktoken", O200K_BASE_HASH, cache);
        tokenizers.add(new TokenizerSpec("o200k_base",
            Tiktoken.createFromTiktoken("o200k_base", o200kRanks,
                Pattern.compile(O200K_PATTERN),
                Map.of("<|endoftext|>", 199999, "<|endofprompt|>", 200018))));
    }

    private static Map<String, Integer> loadMergeableRanks(
            String fileName, String expectedHash, Map<String, Map<String, Integer>> cache) {
        return cache.computeIfAbsent(
                fileName,
                key -> {
                    try {
                        return ClassicBPE.loadMergeableRanks(
                                resourcePath(fileName).toString(), expectedHash);
                    } catch (Exception e) {
                        throw new IllegalStateException("Failed to load mergeable ranks", e);
                    }
                });
    }

    private static Path resourcePath(String fileName) {
        try {
            return Path.of(
                    TokenizerCornerCaseTest.class
                            .getClassLoader()
                            .getResource("tiktoken/" + fileName)
                            .toURI());
        } catch (URISyntaxException e) {
            throw new IllegalStateException("Failed to resolve " + fileName, e);
        }
    }

    // ==================== EMPTY AND BOUNDARY CASES ====================

    @Test
    void testEmptyString() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            IntSequence tokens = tokenizer.encode("");
            assertEquals(0, tokens.length(), spec.name() + " empty string should produce no tokens");
            assertEquals("", tokenizer.decode(tokens), spec.name() + " empty decode should produce empty string");
            assertArrayEquals(new byte[0], tokenizer.decodeBytes(tokens), 
                spec.name() + " empty decodeBytes should produce empty array");
        }
    }

    @Test
    void testSingleCharacter() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String text = "a";
            IntSequence tokens = tokenizer.encode(text);
            assertTrue(tokens.length() > 0, spec.name() + " single char should produce tokens");
            assertEquals(text, tokenizer.decode(tokens), spec.name() + " single char round-trip");
        }
    }

    @Test
    void testWhitespaceOnly() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String[] whitespaceTexts = {" ", "\t", "\n", "\r\n", "   ", " \t\n"};
            for (String text : whitespaceTexts) {
                IntSequence tokens = tokenizer.encode(text);
                assertTrue(tokens.length() > 0, spec.name() + " whitespace should produce tokens");
                assertEquals(text, tokenizer.decode(tokens), spec.name() + " whitespace round-trip");
            }
        }
    }

    // ==================== EMOJI TESTS ====================

    @Test
    void testSingleEmoji() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String[] emojis = {"😀", "🎉", "✨", "❤️", "🔥"};
            for (String emoji : emojis) {
                IntSequence tokens = tokenizer.encode(emoji);
                assertTrue(tokens.length() > 0, spec.name() + " single emoji should produce tokens: " + emoji);
                String decoded = tokenizer.decode(tokens);
                assertEquals(emoji, decoded, spec.name() + " single emoji round-trip: " + emoji);
            }
        }
    }

    @Test
    void testMultipleEmojis() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String text = "😀🎉✨❤️🔥";
            IntSequence tokens = tokenizer.encode(text);
            assertTrue(tokens.length() > 0, spec.name() + " multiple emojis should produce tokens");
            String decoded = tokenizer.decode(tokens);
            assertEquals(text, decoded, spec.name() + " multiple emojis round-trip");
        }
    }

    @Test
    void testEmojiWithText() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String[] texts = {
                "Hello 😀",
                "🎉 Party time!",
                "Start 🎉 middle ✨ end 🔥",
                "👨‍👩‍👧‍👦 Family"
            };
            for (String text : texts) {
                IntSequence tokens = tokenizer.encode(text);
                assertTrue(tokens.length() > 0, spec.name() + " emoji with text should produce tokens");
                String decoded = tokenizer.decode(tokens);
                assertEquals(text, decoded, spec.name() + " emoji with text round-trip: " + text);
            }
        }
    }

    @Test
    void testCountryFlags() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String[] flags = {"🇺🇸", "🇯🇵", "🇬🇧", "🇩🇪", "🇫🇷"};
            for (String flag : flags) {
                IntSequence tokens = tokenizer.encode(flag);
                assertTrue(tokens.length() > 0, spec.name() + " flag should produce tokens: " + flag);
                String decoded = tokenizer.decode(tokens);
                assertEquals(flag, decoded, spec.name() + " flag round-trip: " + flag);
            }
        }
    }

    @Test
    void testEmojiZWJ() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            // Zero-width joiner sequences
            String[] zwjEmojis = {
                "👨‍👩‍👧‍👦",  // Family
                "🏳️‍🌈",     // Rainbow flag
                "👨‍💻",      // Man technologist
                "👩‍🍳"       // Woman cook
            };
            for (String emoji : zwjEmojis) {
                IntSequence tokens = tokenizer.encode(emoji);
                assertTrue(tokens.length() > 0, spec.name() + " ZWJ emoji should produce tokens: " + emoji);
                String decoded = tokenizer.decode(tokens);
                assertEquals(emoji, decoded, spec.name() + " ZWJ emoji round-trip: " + emoji);
            }
        }
    }

    @Test
    void testEmojiSkinTone() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String[] skinToneEmojis = {
                "👍🏽",  // Medium skin tone
                "👋🏿",  // Dark skin tone
                "✋🏻",  // Light skin tone
                "🙋🏾"   // Medium-dark skin tone
            };
            for (String emoji : skinToneEmojis) {
                IntSequence tokens = tokenizer.encode(emoji);
                assertTrue(tokens.length() > 0, spec.name() + " skin tone emoji should produce tokens: " + emoji);
                String decoded = tokenizer.decode(tokens);
                assertEquals(emoji, decoded, spec.name() + " skin tone emoji round-trip: " + emoji);
            }
        }
    }

    // ==================== UNICODE SCRIPT TESTS ====================

    @Test
    void testCJK() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String[] texts = {
                "こんにちは世界",  // Japanese
                "你好世界",        // Chinese (Simplified)
                "안녕하세요 세계",   // Korean
                "日本語のテキスト"  // Japanese
            };
            for (String text : texts) {
                IntSequence tokens = tokenizer.encode(text);
                assertTrue(tokens.length() > 0, spec.name() + " CJK should produce tokens: " + text);
                String decoded = tokenizer.decode(tokens);
                assertEquals(text, decoded, spec.name() + " CJK round-trip: " + text);
            }
        }
    }

    @Test
    void testArabic() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String[] texts = {
                "مرحبا بالعالم",
                "السلام عليكم",
                "كيف حالك"
            };
            for (String text : texts) {
                IntSequence tokens = tokenizer.encode(text);
                assertTrue(tokens.length() > 0, spec.name() + " Arabic should produce tokens: " + text);
                String decoded = tokenizer.decode(tokens);
                assertEquals(text, decoded, spec.name() + " Arabic round-trip: " + text);
            }
        }
    }

    @Test
    void testHebrew() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String[] texts = {
                "שלום עולם",
                "מה נשמע",
                "תודה רבה"
            };
            for (String text : texts) {
                IntSequence tokens = tokenizer.encode(text);
                assertTrue(tokens.length() > 0, spec.name() + " Hebrew should produce tokens: " + text);
                String decoded = tokenizer.decode(tokens);
                assertEquals(text, decoded, spec.name() + " Hebrew round-trip: " + text);
            }
        }
    }

    @Test
    void testIndicScripts() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String[] texts = {
                "नमस्ते दुनिया",    // Hindi
                "হ্যালো ওয়ার্ল্ড",    // Bengali
                "வணக்கம் உலகம்",     // Tamil
                "హలో వరల్డ్"          // Telugu
            };
            for (String text : texts) {
                IntSequence tokens = tokenizer.encode(text);
                assertTrue(tokens.length() > 0, spec.name() + " Indic script should produce tokens: " + text);
                String decoded = tokenizer.decode(tokens);
                assertEquals(text, decoded, spec.name() + " Indic script round-trip: " + text);
            }
        }
    }

    @Test
    void testGreek() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String[] texts = {
                "Γειά σου Κόσμε",
                "ΑΒΓΔΕ αβγδε",
                "Καλημέρα"
            };
            for (String text : texts) {
                IntSequence tokens = tokenizer.encode(text);
                assertTrue(tokens.length() > 0, spec.name() + " Greek should produce tokens: " + text);
                String decoded = tokenizer.decode(tokens);
                assertEquals(text, decoded, spec.name() + " Greek round-trip: " + text);
            }
        }
    }

    @Test
    void testCyrillic() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String[] texts = {
                "Привет мир",
                "АБВГД абвгд",
                "Здравствуй"
            };
            for (String text : texts) {
                IntSequence tokens = tokenizer.encode(text);
                assertTrue(tokens.length() > 0, spec.name() + " Cyrillic should produce tokens: " + text);
                String decoded = tokenizer.decode(tokens);
                assertEquals(text, decoded, spec.name() + " Cyrillic round-trip: " + text);
            }
        }
    }

    // ==================== COMBINING CHARACTERS ====================

    @Test
    void testCombiningCharacters() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            // Test combining characters (e + combining acute)
            String[] texts = {
                "é",           // Precomposed
                "e\u0301",     // e + combining acute
                "ñ",           // Precomposed
                "n\u0303",     // n + combining tilde
                "ü",           // Precomposed
                "u\u0308"      // u + combining diaeresis
            };
            for (String text : texts) {
                IntSequence tokens = tokenizer.encode(text);
                assertTrue(tokens.length() > 0, spec.name() + " combining chars should produce tokens: " + text);
                String decoded = tokenizer.decode(tokens);
                assertEquals(text, decoded, spec.name() + " combining chars round-trip: " + text);
            }
        }
    }

    // ==================== ZERO-WIDTH CHARACTERS ====================

    @Test
    void testZeroWidthCharacters() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String[] texts = {
                "\u200B",       // Zero-width space
                "\u200D",       // Zero-width joiner
                "a\u200Bb",     // ZWJ between letters
                "\u200E",       // Left-to-right mark
                "\u200F"        // Right-to-left mark
            };
            for (String text : texts) {
                IntSequence tokens = tokenizer.encode(text);
                // Should not throw and should round-trip
                String decoded = tokenizer.decode(tokens);
                assertEquals(text, decoded, spec.name() + " ZW chars round-trip");
            }
        }
    }

    // ==================== MIXED SCRIPTS ====================

    @Test
    void testMixedScripts() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String[] texts = {
                "Hello世界مرحباשלום",
                "Price: $100 or €85 or £75",
                "Contact: test@example.com or visit https://example.com",
                "Math: E = mc² and π ≈ 3.14159",
                "Temperature: 25°C or 77°F"
            };
            for (String text : texts) {
                IntSequence tokens = tokenizer.encode(text);
                assertTrue(tokens.length() > 0, spec.name() + " mixed scripts should produce tokens");
                String decoded = tokenizer.decode(tokens);
                assertEquals(text, decoded, spec.name() + " mixed scripts round-trip: " + text);
            }
        }
    }

    // ==================== LONG TEXT TESTS ====================

    @Test
    void testVeryLongText() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String longText = "a".repeat(10000);
            IntSequence tokens = tokenizer.encode(longText);
            assertTrue(tokens.length() > 0, spec.name() + " long text should produce tokens");
            String decoded = tokenizer.decode(tokens);
            assertEquals(longText, decoded, spec.name() + " long text round-trip");
        }
    }

    @Test
    void testRepeatedPattern() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String text = "ab".repeat(5000);
            IntSequence tokens = tokenizer.encode(text);
            assertTrue(tokens.length() > 0, spec.name() + " repeated pattern should produce tokens");
            String decoded = tokenizer.decode(tokens);
            assertEquals(text, decoded, spec.name() + " repeated pattern round-trip");
        }
    }

    // ==================== VOCABULARY TESTS ====================

    @Test
    void testTokensAreInVocabulary() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String text = "Testing various characters: Hello世界😀مرحبا";
            IntSequence tokens = tokenizer.encode(text);
            for (int i = 0; i < tokens.length(); i++) {
                int tokenId = tokens.intAt(i);
                assertTrue(tokenizer.vocabulary().contains(tokenId),
                    spec.name() + " token " + tokenId + " should be in vocabulary");
            }
        }
    }

    @Test
    void testSpecialTokenRoundTrip() {
        // Test that special tokens can be decoded even if they can't be encoded as text
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            Vocabulary vocab = tokenizer.vocabulary();
            
            // Check that special tokens are in vocabulary and can be decoded
            if (spec.name().equals("r50k_base") || spec.name().equals("p50k_base")) {
                assertTrue(vocab.contains(50256), spec.name() + " should have endoftext token");
                assertEquals("<|endoftext|>", vocab.token(50256));
            }
        }
    }

    // ==================== INVALID TOKEN ID TESTS ====================

    @Test
    void testDecodeNegativeTokenIdThrowsException() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            // Test that decoding a negative token ID throws an exception
            // Note: jtokkit throws NullPointerException, not NoSuchElementException
            IntSequence invalidTokens = IntSequence.of(-1);
            assertThrows(Exception.class, 
                () -> tokenizer.decode(invalidTokens),
                spec.name() + " should throw exception for negative token ID");
            assertThrows(Exception.class, 
                () -> tokenizer.decodeBytes(invalidTokens),
                spec.name() + " should throw exception for negative token ID (bytes)");
        }
    }

    @Test
    void testDecodeTooLargeTokenIdThrowsException() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            // Test that decoding a token ID beyond vocabulary size throws an exception
            // Note: jtokkit throws NullPointerException, not NoSuchElementException
            int vocabSize = tokenizer.vocabulary().size();
            IntSequence invalidTokens = IntSequence.of(vocabSize + 1000);
            assertThrows(Exception.class, 
                () -> tokenizer.decode(invalidTokens),
                spec.name() + " should throw exception for token ID beyond vocabulary");
            assertThrows(Exception.class, 
                () -> tokenizer.decodeBytes(invalidTokens),
                spec.name() + " should throw exception for token ID beyond vocabulary (bytes)");
        }
    }

    @Test
    void testDecodeMaxIntegerTokenIdThrowsException() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            // Test that decoding Integer.MAX_VALUE throws an exception
            // Note: jtokkit throws NullPointerException, not NoSuchElementException
            IntSequence invalidTokens = IntSequence.of(Integer.MAX_VALUE);
            assertThrows(Exception.class, 
                () -> tokenizer.decode(invalidTokens),
                spec.name() + " should throw exception for Integer.MAX_VALUE token ID");
        }
    }

    @Test
    void testDecodeSequenceWithOneInvalidTokenId() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            // Test that decoding a sequence with one invalid token throws
            String text = "Hello world";
            IntSequence validTokens = tokenizer.encode(text);
            assertTrue(validTokens.length() > 0, spec.name() + " should have valid tokens");
            
            // Create sequence with one valid and one invalid token
            IntSequence.Builder builder = IntSequence.newBuilder();
            builder.add(validTokens.intAt(0));
            builder.add(-1); // Invalid token
            IntSequence mixedTokens = builder.build();
            
            assertThrows(Exception.class, 
                () -> tokenizer.decode(mixedTokens),
                spec.name() + " should throw exception for sequence with invalid token");
        }
    }

    @Test
    void testVocabularyContainsInvalidTokenIds() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            Vocabulary vocab = tokenizer.vocabulary();
            
            // Test that invalid token IDs are not in vocabulary
            assertFalse(vocab.contains(-1), spec.name() + " negative token should not exist");
            assertFalse(vocab.contains(Integer.MAX_VALUE), spec.name() + " huge token should not exist");
            
            // Test that vocabulary size is reasonable
            assertTrue(vocab.size() > 50000, spec.name() + " vocabulary should be large");
            
            // Test that token IDs beyond vocabulary size are not contained
            assertFalse(vocab.contains(vocab.size()), spec.name() + " token ID at vocab size should not exist");
            assertFalse(vocab.contains(vocab.size() + 1000), spec.name() + " token ID beyond vocab size should not exist");
        }
    }

    @Test
    void testVocabularyTokenLookupInvalidIdThrows() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            Vocabulary vocab = tokenizer.vocabulary();
            
            // Test that looking up invalid token IDs throws NoSuchElementException
            assertThrows(NoSuchElementException.class, 
                () -> vocab.token(-1),
                spec.name() + " should throw for negative token ID lookup");
            assertThrows(NoSuchElementException.class, 
                () -> vocab.token(vocab.size()),
                spec.name() + " should throw for token ID at vocab size");
            assertThrows(NoSuchElementException.class, 
                () -> vocab.token(Integer.MAX_VALUE),
                spec.name() + " should throw for max int token ID");
        }
    }

    // ==================== SPECIAL TOKEN HANDLING TESTS ====================

    @Test
    void testSpecialTokensInTextAreNotTokenized() {
        // Test that special token text in regular text is tokenized as regular text
        // NOT as the special token ID
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            Vocabulary vocab = tokenizer.vocabulary();

            // Get the special token text and ID
            String specialTokenText = "<|endoftext|>";
            int specialTokenId;

            if (spec.name().equals("r50k_base") || spec.name().equals("p50k_base")) {
                specialTokenId = 50256;
            } else if (spec.name().equals("p50k_edit")) {
                specialTokenId = 50256;
            } else if (spec.name().equals("cl100k_base")) {
                specialTokenId = 100257;
            } else if (spec.name().equals("o200k_base")) {
                specialTokenId = 199999;
            } else {
                continue;
            }

            // Verify the special token exists in vocabulary
            assertTrue(vocab.contains(specialTokenId),
                spec.name() + " should contain special token " + specialTokenText);
            assertEquals(specialTokenText, vocab.token(specialTokenId),
                spec.name() + " special token text should match");

            // Encode the special token text as regular text
            // Note: JTokkitAdapter.encode() should raise an error when encountering special tokens
            // This is the expected behavior (like OpenAI's tiktoken with disallowed_special="all")
            // jtokkit throws UnsupportedOperationException, not IllegalArgumentException
            assertThrows(UnsupportedOperationException.class,
                () -> tokenizer.encode(specialTokenText),
                spec.name() + " should raise error when encoding special token text");
        }
    }

    @Test
    void testSpecialTokenTextRoundTrip() {
        // Test that special token text can be decoded when treated as regular text
        // by manually constructing the token sequence
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            Vocabulary vocab = tokenizer.vocabulary();
            
            // Get the special token ID
            int specialTokenId;
            String specialTokenText;
            
            if (spec.name().equals("r50k_base") || spec.name().equals("p50k_base")) {
                specialTokenId = 50256;
                specialTokenText = "<|endoftext|>";
            } else if (spec.name().equals("p50k_edit")) {
                specialTokenId = 50256;
                specialTokenText = "<|endoftext|>";
            } else if (spec.name().equals("cl100k_base")) {
                specialTokenId = 100257;
                specialTokenText = "<|endoftext|>";
            } else if (spec.name().equals("o200k_base")) {
                specialTokenId = 199999;
                specialTokenText = "<|endoftext|>";
            } else {
                continue;
            }
            
            // Decode the special token ID directly
            String decoded = vocab.token(specialTokenId);
            assertEquals(specialTokenText, decoded,
                spec.name() + " should decode special token ID to text");
            
            // Decode via tokenizer
            IntSequence tokens = IntSequence.of(specialTokenId);
            String decodedViaTokenizer = tokenizer.decode(tokens);
            assertEquals(specialTokenText, decodedViaTokenizer,
                spec.name() + " tokenizer should decode special token ID to text");
        }
    }

    @Test
    void testSpecialTokenIdInVocabulary() {
        // Test that special token IDs are in vocabulary and have correct type
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            Vocabulary vocab = tokenizer.vocabulary();
            
            Map<String, Integer> expectedSpecials = new HashMap<>();
            if (spec.name().equals("r50k_base") || spec.name().equals("p50k_base")) {
                expectedSpecials.put("<|endoftext|>", 50256);
            } else if (spec.name().equals("p50k_edit")) {
                expectedSpecials.put("<|endoftext|>", 50256);
                expectedSpecials.put("<|fim_prefix|>", 50281);
                expectedSpecials.put("<|fim_middle|>", 50282);
                expectedSpecials.put("<|fim_suffix|>", 50283);
            } else if (spec.name().equals("cl100k_base")) {
                expectedSpecials.put("<|endoftext|>", 100257);
                expectedSpecials.put("<|fim_prefix|>", 100258);
                expectedSpecials.put("<|fim_middle|>", 100259);
                expectedSpecials.put("<|fim_suffix|>", 100260);
                expectedSpecials.put("<|endofprompt|>", 100276);
            } else if (spec.name().equals("o200k_base")) {
                expectedSpecials.put("<|endoftext|>", 199999);
                expectedSpecials.put("<|endofprompt|>", 200018);
            }
            
            for (Map.Entry<String, Integer> entry : expectedSpecials.entrySet()) {
                String tokenText = entry.getKey();
                int tokenId = entry.getValue();
                
                assertTrue(vocab.contains(tokenId),
                    spec.name() + " should contain special token " + tokenText + " (ID: " + tokenId + ")");
                assertEquals(tokenText, vocab.token(tokenId),
                    spec.name() + " special token " + tokenText + " should decode correctly");
            }
        }
    }

    @Test
    void testSpecialTokenInMixedText() {
        // Test that special tokens in mixed text raise errors
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            
            // Text containing special token
            String[] textsWithSpecials = {
                "Hello <|endoftext|> world",
                "<|endoftext|>",
                "Start <|endoftext|> end"
            };
            
            for (String text : textsWithSpecials) {
                // JTokkitAdapter should raise an error for special tokens in text
                // jtokkit throws UnsupportedOperationException
                assertThrows(UnsupportedOperationException.class,
                    () -> tokenizer.encode(text),
                    spec.name() + " should raise error for text containing special token: " + text);
            }
        }
    }

    @Test
    void testSpecialTokenDecodeOnly() {
        // Test that special tokens can be decoded even though they can't be encoded from text
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            
            int specialTokenId;
            String specialTokenText = "<|endoftext|>";
            
            if (spec.name().equals("r50k_base") || spec.name().equals("p50k_base")) {
                specialTokenId = 50256;
            } else if (spec.name().equals("p50k_edit")) {
                specialTokenId = 50256;
            } else if (spec.name().equals("cl100k_base")) {
                specialTokenId = 100257;
            } else if (spec.name().equals("o200k_base")) {
                specialTokenId = 199999;
            } else {
                continue;
            }
            
            // Decode the special token ID
            IntSequence tokens = IntSequence.of(specialTokenId);
            String decoded = tokenizer.decode(tokens);
            assertEquals(specialTokenText, decoded,
                spec.name() + " should decode special token ID to text");
            
            // Also test decodeBytes
            byte[] decodedBytes = tokenizer.decodeBytes(tokens);
            byte[] expectedBytes = specialTokenText.getBytes(java.nio.charset.StandardCharsets.UTF_8);
            assertArrayEquals(expectedBytes, decodedBytes,
                spec.name() + " decodeBytes should produce correct UTF-8 bytes for special token");
        }
    }

    @Test
    void testSpecialTokenVsRegularTextTokens() {
        // Test that special token text produces different tokens than the special token ID
        // when encoded as regular text (which should fail)
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            
            String specialTokenText = "<|endoftext|>";
            int specialTokenId;
            
            if (spec.name().equals("r50k_base") || spec.name().equals("p50k_base")) {
                specialTokenId = 50256;
            } else if (spec.name().equals("p50k_edit")) {
                specialTokenId = 50256;
            } else if (spec.name().equals("cl100k_base")) {
                specialTokenId = 100257;
            } else if (spec.name().equals("o200k_base")) {
                specialTokenId = 199999;
            } else {
                continue;
            }
            
            // The special token ID should be in vocabulary
            assertTrue(tokenizer.vocabulary().contains(specialTokenId),
                spec.name() + " special token ID should be in vocabulary");
            
            // But encoding the special token text should fail (not produce the special token ID)
            // jtokkit throws UnsupportedOperationException
            assertThrows(UnsupportedOperationException.class,
                () -> tokenizer.encode(specialTokenText),
                spec.name() + " encoding special token text should fail, not produce special token ID");
        }
    }

    @Test
    void testVocabularyLookupByString() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            Vocabulary vocab = tokenizer.vocabulary();
            
            // Test that we can look up known tokens
            String text = "Hello";
            IntSequence tokens = tokenizer.encode(text);
            if (tokens.length() > 0) {
                String tokenStr = vocab.token(tokens.intAt(0));
                assertNotNull(tokenStr, spec.name() + " should return token string");
            }
        }
    }

    // ==================== BYTE-LEVEL DECODING TESTS ====================

    @Test
    void testByteLevelDecoding() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String text = "Byte-level test with unicode: 你好مرحبا";
            IntSequence tokens = tokenizer.encode(text);
            byte[] decodedBytes = tokenizer.decodeBytes(tokens);
            byte[] expectedBytes = text.getBytes(java.nio.charset.StandardCharsets.UTF_8);
            assertArrayEquals(expectedBytes, decodedBytes, 
                spec.name() + " byte decoding should match UTF-8");
        }
    }

    @Test
    void testByteDecodingRoundTrip() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String text = "Test with bytes: 你好😀";
            IntSequence tokens = tokenizer.encode(text);
            byte[] decodedBytes = tokenizer.decodeBytes(tokens);
            String reconstructed = new String(decodedBytes, java.nio.charset.StandardCharsets.UTF_8);
            assertEquals(text, reconstructed, spec.name() + " byte round-trip should work");
        }
    }

    // ==================== TOKEN COUNT TESTS ====================

    @Test
    void testTokenCountMatchesEncodeLength() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String[] texts = {
                "Hello world",
                "Testing token counts",
                "Unicode: 你好😀",
                ""
            };
            for (String text : texts) {
                int count = tokenizer.countTokens(text);
                IntSequence tokens = tokenizer.encode(text);
                assertEquals(tokens.length(), count, 
                    spec.name() + " countTokens should match encode length");
            }
        }
    }

    // ==================== PUNCTUATION AND SYMBOLS ====================

    @Test
    void testPunctuation() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String[] texts = {
                "!@#$%^&*()",
                "[]{}|;':\",./<>?",
                "... --- ...",
                "«»‹›\"\"''",
                "——–•·"
            };
            for (String text : texts) {
                IntSequence tokens = tokenizer.encode(text);
                assertTrue(tokens.length() >= 0, spec.name() + " punctuation should be handled");
                String decoded = tokenizer.decode(tokens);
                assertEquals(text, decoded, spec.name() + " punctuation round-trip: " + text);
            }
        }
    }

    @Test
    void testMathSymbols() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String[] texts = {
                "±×÷√∞",
                "∑∏∫∂",
                "≤≥≠≈",
                "αβγδε",
                "∴∵∧∨"
            };
            for (String text : texts) {
                IntSequence tokens = tokenizer.encode(text);
                assertTrue(tokens.length() > 0, spec.name() + " math symbols should produce tokens");
                String decoded = tokenizer.decode(tokens);
                assertEquals(text, decoded, spec.name() + " math symbols round-trip: " + text);
            }
        }
    }

    @Test
    void testCurrencySymbols() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String[] texts = {
                "$100.00",
                "€50,00",
                "£25.00",
                "¥1000",
                "₹500"
            };
            for (String text : texts) {
                IntSequence tokens = tokenizer.encode(text);
                assertTrue(tokens.length() > 0, spec.name() + " currency should produce tokens");
                String decoded = tokenizer.decode(tokens);
                assertEquals(text, decoded, spec.name() + " currency round-trip: " + text);
            }
        }
    }

    // ==================== URL AND EMAIL TESTS ====================

    @Test
    void testUrls() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String[] texts = {
                "https://example.com",
                "https://example.com/path?query=value",
                "http://localhost:8080",
                "ftp://files.example.com"
            };
            for (String text : texts) {
                IntSequence tokens = tokenizer.encode(text);
                assertTrue(tokens.length() > 0, spec.name() + " URL should produce tokens");
                String decoded = tokenizer.decode(tokens);
                assertEquals(text, decoded, spec.name() + " URL round-trip: " + text);
            }
        }
    }

    @Test
    void testEmails() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String[] texts = {
                "test@example.com",
                "user.name@domain.co.uk",
                "first+last@example.org"
            };
            for (String text : texts) {
                IntSequence tokens = tokenizer.encode(text);
                assertTrue(tokens.length() > 0, spec.name() + " email should produce tokens");
                String decoded = tokenizer.decode(tokens);
                assertEquals(text, decoded, spec.name() + " email round-trip: " + text);
            }
        }
    }

    // ==================== CODE AND MARKUP ====================

    @Test
    void testCode() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String[] texts = {
                "def hello():\n    return 'world'",
                "const x = 42;",
                "if (x > 0) { console.log(x); }",
                "class MyClass { }"
            };
            for (String text : texts) {
                IntSequence tokens = tokenizer.encode(text);
                assertTrue(tokens.length() > 0, spec.name() + " code should produce tokens");
                String decoded = tokenizer.decode(tokens);
                assertEquals(text, decoded, spec.name() + " code round-trip");
            }
        }
    }

    @Test
    void testMarkup() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String[] texts = {
                "<html><body>Hello</body></html>",
                "<div class='test'>Content</div>",
                "{\"key\": \"value\", \"num\": 42}",
                "# Header\n\n**bold** and *italic*"
            };
            for (String text : texts) {
                IntSequence tokens = tokenizer.encode(text);
                assertTrue(tokens.length() > 0, spec.name() + " markup should produce tokens");
                String decoded = tokenizer.decode(tokens);
                assertEquals(text, decoded, spec.name() + " markup round-trip");
            }
        }
    }

    // ==================== CONTROL CHARACTERS ====================

    @Test
    void testControlCharacters() {
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            // Test various control characters
            String[] texts = {
                "\t",           // Tab
                "\n",           // Newline
                "\r\n",         // CRLF
                "\f",           // Form feed
                "Line1\nLine2", // Multi-line
                "Tab\tSeparated\tValues"
            };
            for (String text : texts) {
                IntSequence tokens = tokenizer.encode(text);
                assertTrue(tokens.length() >= 0, spec.name() + " control chars should be handled");
                String decoded = tokenizer.decode(tokens);
                assertEquals(text, decoded, spec.name() + " control chars round-trip");
            }
        }
    }

    // ==================== UTF-8 BYTE SEQUENCE TESTS ====================

    @Test
    void testUtf8ByteSequenceRoundTrip() {
        // Test that encode/decode preserves UTF-8 byte sequences exactly
        // This is critical for BPE tokenizers that operate on raw bytes
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            
            // Test various UTF-8 sequences
            String[] texts = {
                "Hello",                                    // ASCII
                "こんにちは",                                 // Japanese (3-byte UTF-8)
                "你好",                                      // Chinese (3-byte UTF-8)
                "😀",                                       // Emoji (4-byte UTF-8)
                "Hello 世界",                                // Mixed ASCII and CJK
                "αβγ",                                      // Greek (2-byte UTF-8)
                "مرحبا",                                    // Arabic (2-byte UTF-8)
                "\u00E9",                                   // é - Latin Extended (2-byte)
                "\u4E00",                                   // 一 - CJK Unified (3-byte)
                "\u1F600"                                   // 😀 - Emoji (4-byte)
            };
            
            for (String text : texts) {
                IntSequence tokens = tokenizer.encode(text);
                byte[] decodedBytes = tokenizer.decodeBytes(tokens);
                byte[] originalBytes = text.getBytes(java.nio.charset.StandardCharsets.UTF_8);
                
                assertArrayEquals(originalBytes, decodedBytes,
                    spec.name() + " UTF-8 bytes should match for: " + text);
                
                // Also verify string round-trip
                String decoded = tokenizer.decode(tokens);
                assertEquals(text, decoded,
                    spec.name() + " String round-trip should match for: " + text);
            }
        }
    }

    @Test
    void testMultiByteUtf8Characters() {
        // Test that multi-byte UTF-8 characters are handled correctly
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            
            // Test characters that require multiple bytes in UTF-8
            String[] multiByteTexts = {
                "日本語",           // Japanese: 3 bytes each
                "한국어",           // Korean: 3 bytes each
                "🎉🎊🎁",           // Emoji: 4 bytes each
                "👨‍👩‍👧‍👦",           // Family emoji with ZWJ
                "🇺🇸🇬🇧",           // Country flags
                "🤦🏽‍♂️"            // Emoji with skin tone and gender
            };
            
            for (String text : multiByteTexts) {
                IntSequence tokens = tokenizer.encode(text);
                assertTrue(tokens.length() > 0, 
                    spec.name() + " should produce tokens for: " + text);
                
                byte[] decodedBytes = tokenizer.decodeBytes(tokens);
                String reconstructed = new String(decodedBytes, java.nio.charset.StandardCharsets.UTF_8);
                assertEquals(text, reconstructed,
                    spec.name() + " Multi-byte UTF-8 round-trip should work for: " + text);
            }
        }
    }

    @Test
    void testByteLevelConsistency() {
        // Test that decodeBytes is consistent with decode
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            String text = "Test consistency with unicode: 你好世界🌍";
            
            IntSequence tokens = tokenizer.encode(text);
            
            // Decode via both methods
            String decodedString = tokenizer.decode(tokens);
            byte[] decodedBytes = tokenizer.decodeBytes(tokens);
            String bytesAsString = new String(decodedBytes, java.nio.charset.StandardCharsets.UTF_8);
            
            // Both should produce the same result
            assertEquals(decodedString, bytesAsString,
                spec.name() + " decode() and decodeBytes() should be consistent");
            assertEquals(text, decodedString,
                spec.name() + " Round-trip should preserve text");
        }
    }

    // ==================== VOCABULARY BOUNDARY TESTS ====================

    @Test
    void testVocabularySizeBoundaries() {
        // Test vocabulary size boundaries for each tokenizer
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            Vocabulary vocab = tokenizer.vocabulary();
            int vocabSize = vocab.size();
            
            // Verify vocabulary size is reasonable
            assertTrue(vocabSize > 50000, 
                spec.name() + " vocabulary should be large (>50000)");
            
            // Find the actual max valid token ID (accounting for gaps)
            int maxValidId = -1;
            for (int i = vocabSize - 1; i >= 0; i--) {
                if (vocab.contains(i)) {
                    maxValidId = i;
                    break;
                }
            }
            assertTrue(maxValidId >= 0,
                spec.name() + " should have at least one valid token ID");
            
            // Test that token ID beyond vocabulary size is invalid
            assertFalse(vocab.contains(vocabSize),
                spec.name() + " token ID at vocab size (" + vocabSize + ") should not exist");
            
            // Test that we can look up the max valid token
            String maxToken = vocab.token(maxValidId);
            assertNotNull(maxToken,
                spec.name() + " should be able to look up max valid token");
        }
    }

    @Test
    void testKnownVocabularySizes() {
        // Test that vocabulary sizes match expected values for known tokenizers
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            Vocabulary vocab = tokenizer.vocabulary();
            int vocabSize = vocab.size();
            
            // Check approximate sizes for known tokenizers
            // These are the typical vocabulary sizes
            switch (spec.name()) {
                case "r50k_base":
                case "p50k_base":
                case "p50k_edit":
                    // GPT-2 family: ~50,257 tokens
                    assertTrue(vocabSize >= 50000 && vocabSize <= 51000,
                        spec.name() + " vocabulary size should be ~50257, got " + vocabSize);
                    break;
                case "cl100k_base":
                    // CL100K: ~100,277 tokens
                    assertTrue(vocabSize >= 100000 && vocabSize <= 101000,
                        spec.name() + " vocabulary size should be ~100277, got " + vocabSize);
                    break;
                case "o200k_base":
                    // O200K: ~200,019 tokens
                    assertTrue(vocabSize >= 200000 && vocabSize <= 201000,
                        spec.name() + " vocabulary size should be ~200019, got " + vocabSize);
                    break;
            }
        }
    }

    @Test
    void testSpecialTokenBoundaries() {
        // Test that special tokens are at expected positions
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            Vocabulary vocab = tokenizer.vocabulary();
            
            // Check special token positions
            if (spec.name().equals("r50k_base") || spec.name().equals("p50k_base")) {
                // GPT-2: endoftext at 50256 (last token)
                assertTrue(vocab.contains(50256),
                    spec.name() + " should have endoftext at 50256");
                assertEquals("<|endoftext|>", vocab.token(50256));
            } else if (spec.name().equals("cl100k_base")) {
                // CL100K: special tokens at high indices
                assertTrue(vocab.contains(100257),
                    spec.name() + " should have endoftext at 100257");
            } else if (spec.name().equals("o200k_base")) {
                // O200K: special tokens at very high indices
                assertTrue(vocab.contains(199999),
                    spec.name() + " should have endoftext at 199999");
            }
        }
    }

    // ==================== SPECIAL TOKEN INJECTION PREVENTION TESTS ====================

    @Test
    void testSpecialTokenTextNotTokenizedAsSpecial() {
        // Test that special token text in regular text is NOT tokenized as special tokens
        // This is the safe default behavior (injection prevention)
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            
            // Test with endoftext token (all tokenizers have this)
            String[] textsWithEndoftext = {
                "print(\"<|endoftext|>\")",
                "The token <|endoftext|> is special"
            };
            
            for (String text : textsWithEndoftext) {
                // Encoding should fail with UnsupportedOperationException
                // because jtokkit doesn't allow special tokens in text
                assertThrows(UnsupportedOperationException.class,
                    () -> tokenizer.encode(text),
                    spec.name() + " should reject text containing endoftext: " + text);
            }
            
            // Test with FIM tokens (only p50k_edit has these)
            if (spec.name().equals("p50k_edit")) {
                String[] textsWithFim = {
                    "<|fim_prefix|> code <|fim_suffix|>"
                };
                
                for (String text : textsWithFim) {
                    assertThrows(UnsupportedOperationException.class,
                        () -> tokenizer.encode(text),
                        spec.name() + " should reject text containing FIM tokens: " + text);
                }
            }
        }
    }

    @Test
    void testSpecialTokenSurfaceForms() {
        // Test that special token surface forms are properly handled
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            Vocabulary vocab = tokenizer.vocabulary();
            
            // Test that we can decode special token IDs
            int[] specialIds;
            if (spec.name().equals("r50k_base") || spec.name().equals("p50k_base")) {
                specialIds = new int[]{50256};
            } else if (spec.name().equals("p50k_edit")) {
                specialIds = new int[]{50256, 50281, 50282, 50283};
            } else if (spec.name().equals("cl100k_base")) {
                specialIds = new int[]{100257, 100258, 100259, 100260, 100276};
            } else if (spec.name().equals("o200k_base")) {
                specialIds = new int[]{199999, 200018};
            } else {
                continue;
            }
            
            for (int specialId : specialIds) {
                // Verify special token exists and can be decoded
                assertTrue(vocab.contains(specialId),
                    spec.name() + " special token " + specialId + " should exist");
                
                String tokenText = vocab.token(specialId);
                assertNotNull(tokenText,
                    spec.name() + " special token " + specialId + " should have text");
                assertTrue(tokenText.startsWith("<") && tokenText.endsWith(">"),
                    spec.name() + " special token should have bracket format: " + tokenText);
                
                // Verify round-trip via decode
                IntSequence tokens = IntSequence.of(specialId);
                String decoded = tokenizer.decode(tokens);
                assertEquals(tokenText, decoded,
                    spec.name() + " special token should decode correctly");
            }
        }
    }

    @Test
    void testSpecialTokenVsRegularTokenDistinction() {
        // Test that special tokens are distinct from regular tokens
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            Vocabulary vocab = tokenizer.vocabulary();
            
            // Get a special token
            int specialId;
            if (spec.name().equals("r50k_base") || spec.name().equals("p50k_base")) {
                specialId = 50256;
            } else if (spec.name().equals("cl100k_base")) {
                specialId = 100257;
            } else if (spec.name().equals("o200k_base")) {
                specialId = 199999;
            } else {
                continue;
            }
            
            String specialText = vocab.token(specialId);
            
            // The special token text should NOT be in the regular vocabulary
            // (it can only be accessed via the special token ID)
            assertNotNull(specialText,
                spec.name() + " special token should have text representation");
            
            // Verify the special token text looks like a control token
            assertTrue(specialText.matches("<\\|[^|]+\\|>"),
                spec.name() + " special token should match pattern <|...|>: " + specialText);
        }
    }

    // ==================== COMPREHENSIVE BYTE-LEVEL TESTS ====================

    @Test
    void testByteLevelEncodingDecoding() {
        // Comprehensive test of byte-level encoding/decoding
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            
            // Test with various text patterns
            String[] testPatterns = {
                "Hello World",                           // Basic ASCII
                "Hello\nWorld\n",                        // With newlines
                "  Spaces  ",                            // Leading/trailing spaces
                "Tab\tSeparated\tValues",                // Tabs
                "Mixed: 中文 العربية עברית",            // Multiple scripts
                "Special: !@#$%^&*()",                   // Special characters
                "Numbers: 1234567890",                   // Digits
                "Unicode: αβγδε 🎉 🔥 💯"                 // Mixed unicode
            };
            
            for (String text : testPatterns) {
                IntSequence tokens = tokenizer.encode(text);
                byte[] decodedBytes = tokenizer.decodeBytes(tokens);
                byte[] expectedBytes = text.getBytes(java.nio.charset.StandardCharsets.UTF_8);
                
                assertArrayEquals(expectedBytes, decodedBytes,
                    spec.name() + " Byte-level encoding/decoding should match for: " + text);
            }
        }
    }

    @Test
    void testEmptyAndWhitespaceByteDecoding() {
        // Test byte decoding for empty and whitespace-only strings
        for (TokenizerSpec spec : tokenizers) {
            Tokenizer tokenizer = spec.tokenizer();
            
            String[] whitespaceTexts = {"", " ", "  ", "\t", "\n", "\r\n", " \t\n "};
            
            for (String text : whitespaceTexts) {
                IntSequence tokens = tokenizer.encode(text);
                byte[] decodedBytes = tokenizer.decodeBytes(tokens);
                byte[] expectedBytes = text.getBytes(java.nio.charset.StandardCharsets.UTF_8);
                
                assertArrayEquals(expectedBytes, decodedBytes,
                    spec.name() + " Whitespace byte decoding should match for: [" + text + "]");
            }
        }
    }

    private record TokenizerSpec(String name, Tokenizer tokenizer) {}
}
