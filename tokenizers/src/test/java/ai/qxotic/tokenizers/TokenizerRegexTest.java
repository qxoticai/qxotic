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
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;
import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Regex-based tokenization behavior tests.
 * Tests that regex patterns compile correctly and produce consistent tokenization behavior
 * across different tokenizer variants.
 */
class TokenizerRegexTest {

    private static final String R50K_BASE_HASH =
            "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930";
    private static final String P50K_BASE_HASH =
            "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069";
    private static final String CL100K_BASE_HASH =
            "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7";
    private static final String O200K_BASE_HASH =
            "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d";

    // Regex patterns used by different tokenizers
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

    private static List<TokenizerVariant> variants;

    @BeforeAll
    static void setUp() {
        buildTokenizerVariants();
    }

    private static void buildTokenizerVariants() {
        variants = new ArrayList<>();
        Map<String, Map<String, Integer>> cache = new HashMap<>();

        // Build r50k_base tokenizer
        Map<String, Integer> r50kRanks = loadMergeableRanks("r50k_base.tiktoken", R50K_BASE_HASH, cache);
        variants.add(new TokenizerVariant("r50k_base", 
            Tiktoken.createFromTiktoken("r50k_base", r50kRanks, 
                Pattern.compile(R50K_PATTERN), Map.of("<|endoftext|>", 50256))));

        // Build p50k_base tokenizer
        Map<String, Integer> p50kRanks = loadMergeableRanks("p50k_base.tiktoken", P50K_BASE_HASH, cache);
        variants.add(new TokenizerVariant("p50k_base",
            Tiktoken.createFromTiktoken("p50k_base", p50kRanks,
                Pattern.compile(R50K_PATTERN), Map.of("<|endoftext|>", 50256))));

        // Build p50k_edit tokenizer
        variants.add(new TokenizerVariant("p50k_edit",
            Tiktoken.createFromTiktoken("p50k_edit", p50kRanks,
                Pattern.compile(R50K_PATTERN), 
                Map.of("<|endoftext|>", 50256, "<|fim_prefix|>", 50281, 
                       "<|fim_middle|>", 50282, "<|fim_suffix|>", 50283))));

        // Build cl100k_base tokenizer
        Map<String, Integer> cl100kRanks = loadMergeableRanks("cl100k_base.tiktoken", CL100K_BASE_HASH, cache);
        variants.add(new TokenizerVariant("cl100k_base",
            Tiktoken.createFromTiktoken("cl100k_base", cl100kRanks,
                Pattern.compile(CL100K_PATTERN),
                Map.of("<|endoftext|>", 100257, "<|fim_prefix|>", 100258,
                       "<|fim_middle|>", 100259, "<|fim_suffix|>", 100260, "<|endofprompt|>", 100276))));

        // Build o200k_base tokenizer
        Map<String, Integer> o200kRanks = loadMergeableRanks("o200k_base.tiktoken", O200K_BASE_HASH, cache);
        variants.add(new TokenizerVariant("o200k_base",
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
                    TokenizerRegexTest.class
                            .getClassLoader()
                            .getResource("tiktoken/" + fileName)
                            .toURI());
        } catch (URISyntaxException e) {
            throw new IllegalStateException("Failed to resolve " + fileName, e);
        }
    }

    // ==================== REGEX PATTERN VALIDATION TESTS ====================

    @Test
    void testR50KPatternCompiles() {
        assertDoesNotThrow(() -> Pattern.compile(R50K_PATTERN),
            "R50K pattern should compile without errors");
    }

    @Test
    void testCL100KPatternCompiles() {
        assertDoesNotThrow(() -> Pattern.compile(CL100K_PATTERN),
            "CL100K pattern should compile without errors");
    }

    @Test
    void testO200KPatternCompiles() {
        assertDoesNotThrow(() -> Pattern.compile(O200K_PATTERN),
            "O200K pattern should compile without errors");
    }

    @Test
    void testR50KPatternMatchesContractions() {
        Pattern pattern = Pattern.compile(R50K_PATTERN);
        String[] contractions = {"don't", "won't", "can't", "it's", "I'm", "you'll", "I'd"};
        for (String contraction : contractions) {
            assertTrue(pattern.matcher(contraction).find(),
                "R50K pattern should match contraction: " + contraction);
        }
    }

    @Test
    void testR50KPatternMatchesWords() {
        Pattern pattern = Pattern.compile(R50K_PATTERN);
        String[] words = {"hello", "world", "test"};
        for (String word : words) {
            assertTrue(pattern.matcher(word).find(),
                "R50K pattern should match word: " + word);
        }
    }

    @Test
    void testR50KPatternMatchesNumbers() {
        Pattern pattern = Pattern.compile(R50K_PATTERN);
        String[] numbers = {"1", "12", "123", "1234"};
        for (String number : numbers) {
            assertTrue(pattern.matcher(number).find(),
                "R50K pattern should match number: " + number);
        }
    }

    @Test
    void testCL100KPatternMatchesContractions() {
        Pattern pattern = Pattern.compile(CL100K_PATTERN);
        String[] contractions = {"don't", "won't", "can't", "it's", "I'm", "you'll", "I'd"};
        for (String contraction : contractions) {
            assertTrue(pattern.matcher(contraction).find(),
                "CL100K pattern should match contraction: " + contraction);
        }
    }

    @Test
    void testCL100KPatternMatchesNumbersWithGrouping() {
        Pattern pattern = Pattern.compile(CL100K_PATTERN);
        // CL100K groups numbers in 1-3 digit chunks
        String[] numbers = {"1", "12", "123"};
        for (String number : numbers) {
            assertTrue(pattern.matcher(number).find(),
                "CL100K pattern should match number: " + number);
        }
    }

    @Test
    void testO200KPatternMatchesCaseVariants() {
        Pattern pattern = Pattern.compile(O200K_PATTERN);
        String[] words = {"Hello", "HELLO", "hello"};
        for (String word : words) {
            assertTrue(pattern.matcher(word).find(),
                "O200K pattern should match word: " + word);
        }
    }

    // ==================== CONTRACTION HANDLING TESTS ====================

    @ParameterizedTest(name = "{0} - contraction handling")
    @MethodSource("provideTokenizerVariants")
    void testContractionsProduceTokens(TokenizerVariant variant) {
        String[] contractions = {"don't", "won't", "can't", "it's", "I'm", "you'll", "I'd", 
                                 "you're", "I've", "he'd", "she'll", "we've"};
        for (String contraction : contractions) {
            IntSequence tokens = variant.tokenizer().encode(contraction);
            assertTrue(tokens.length() > 0,
                variant.name() + " should produce tokens for contraction: " + contraction);
            
            // Verify round-trip
            String decoded = variant.tokenizer().decode(tokens);
            assertEquals(contraction, decoded,
                variant.name() + " should round-trip contraction correctly: " + contraction);
        }
    }

    @ParameterizedTest(name = "{0} - possessive vs contraction")
    @MethodSource("provideTokenizerVariants")
    void testPossessiveVsContraction(TokenizerVariant variant) {
        // Test that "John's" (possessive) and "John's" in "John's book" work
        String text1 = "John's";
        String text2 = "John's book";
        
        IntSequence tokens1 = variant.tokenizer().encode(text1);
        IntSequence tokens2 = variant.tokenizer().encode(text2);
        
        assertTrue(tokens1.length() > 0, variant.name() + " should tokenize possessive");
        assertTrue(tokens2.length() > tokens1.length(), 
            variant.name() + " should tokenize possessive with more tokens when followed by text");
    }

    // ==================== WHITESPACE HANDLING TESTS ====================

    @ParameterizedTest(name = "{0} - whitespace preservation")
    @MethodSource("provideTokenizerVariants")
    void testWhitespacePreservation(TokenizerVariant variant) {
        String[] texts = {
            "hello world",
            "hello  world",  // Multiple spaces
            "hello\tworld",  // Tab
            "hello\nworld",  // Newline
            "hello\r\nworld", // CRLF
            "  hello",       // Leading space
            "hello  "        // Trailing space
        };
        
        for (String text : texts) {
            IntSequence tokens = variant.tokenizer().encode(text);
            String decoded = variant.tokenizer().decode(tokens);
            assertEquals(text, decoded,
                variant.name() + " should preserve whitespace in: " + text);
        }
    }

    @ParameterizedTest(name = "{0} - empty and whitespace-only")
    @MethodSource("provideTokenizerVariants")
    void testEmptyAndWhitespaceOnly(TokenizerVariant variant) {
        // Empty string
        IntSequence emptyTokens = variant.tokenizer().encode("");
        assertEquals(0, emptyTokens.length(),
            variant.name() + " should produce no tokens for empty string");
        
        // Whitespace-only strings
        String[] whitespaceTexts = {" ", "  ", "\t", "\n", "\r\n", " \t\n "};
        for (String text : whitespaceTexts) {
            IntSequence tokens = variant.tokenizer().encode(text);
            assertTrue(tokens.length() >= 0,
                variant.name() + " should handle whitespace-only text");
            
            // Verify round-trip
            String decoded = variant.tokenizer().decode(tokens);
            assertEquals(text, decoded,
                variant.name() + " should round-trip whitespace-only text");
        }
    }

    // ==================== NUMBER HANDLING TESTS ====================

    @ParameterizedTest(name = "{0} - number tokenization")
    @MethodSource("provideTokenizerVariants")
    void testNumberTokenization(TokenizerVariant variant) {
        String[] numbers = {"1", "12", "123", "1234", "12345", "1234567890"};
        
        for (String number : numbers) {
            IntSequence tokens = variant.tokenizer().encode(number);
            assertTrue(tokens.length() > 0,
                variant.name() + " should tokenize number: " + number);
            
            // Verify round-trip
            String decoded = variant.tokenizer().decode(tokens);
            assertEquals(number, decoded,
                variant.name() + " should round-trip number: " + number);
        }
    }

    @ParameterizedTest(name = "{0} - decimal numbers")
    @MethodSource("provideTokenizerVariants")
    void testDecimalNumbers(TokenizerVariant variant) {
        String[] decimals = {"3.14", "0.5", "123.456"};
        
        for (String decimal : decimals) {
            IntSequence tokens = variant.tokenizer().encode(decimal);
            assertTrue(tokens.length() > 0,
                variant.name() + " should tokenize decimal: " + decimal);
            
            String decoded = variant.tokenizer().decode(tokens);
            assertEquals(decimal, decoded,
                variant.name() + " should round-trip decimal: " + decimal);
        }
    }

    // ==================== PUNCTUATION HANDLING TESTS ====================

    @ParameterizedTest(name = "{0} - punctuation attachment")
    @MethodSource("provideTokenizerVariants")
    void testPunctuationAttachment(TokenizerVariant variant) {
        String[] texts = {
            "Hello.",
            "Hello!",
            "Hello?",
            "Hello...",
            "(hello)",
            "[hello]",
            "{hello}",
            "\"hello\"",
            "'hello'"
        };
        
        for (String text : texts) {
            IntSequence tokens = variant.tokenizer().encode(text);
            assertTrue(tokens.length() > 0,
                variant.name() + " should tokenize text with punctuation: " + text);
            
            String decoded = variant.tokenizer().decode(tokens);
            assertEquals(text, decoded,
                variant.name() + " should round-trip punctuated text: " + text);
        }
    }

    @ParameterizedTest(name = "{0} - special characters")
    @MethodSource("provideTokenizerVariants")
    void testSpecialCharacters(TokenizerVariant variant) {
        String[] texts = {
            "test@example.com",
            "price: $100",
            "100%",
            "a+b=c",
            "x*y",
            "a/b"
        };
        
        for (String text : texts) {
            IntSequence tokens = variant.tokenizer().encode(text);
            assertTrue(tokens.length() > 0,
                variant.name() + " should tokenize special characters: " + text);
            
            String decoded = variant.tokenizer().decode(tokens);
            assertEquals(text, decoded,
                variant.name() + " should round-trip special characters: " + text);
        }
    }

    // ==================== UNICODE HANDLING TESTS ====================

    @ParameterizedTest(name = "{0} - Unicode scripts")
    @MethodSource("provideTokenizerVariants")
    void testUnicodeScripts(TokenizerVariant variant) {
        String[] texts = {
            "café",           // Accented Latin
            "你好世界",        // CJK
            "مرحبا",          // Arabic
            "שלום",           // Hebrew
            "Привет",         // Cyrillic
            "Γειά",           // Greek
            "Hello世界",      // Mixed scripts
            "👋",             // Emoji
        };
        
        for (String text : texts) {
            IntSequence tokens = variant.tokenizer().encode(text);
            assertTrue(tokens.length() > 0,
                variant.name() + " should tokenize Unicode text: " + text);
            
            String decoded = variant.tokenizer().decode(tokens);
            assertEquals(text, decoded,
                variant.name() + " should round-trip Unicode text: " + text);
        }
    }

    @ParameterizedTest(name = "{0} - combining characters")
    @MethodSource("provideTokenizerVariants")
    void testCombiningCharacters(TokenizerVariant variant) {
        // Test precomposed vs decomposed forms
        String precomposed = "café";  // é as single character
        String decomposed = "café";   // e + combining acute
        
        IntSequence tokens1 = variant.tokenizer().encode(precomposed);
        IntSequence tokens2 = variant.tokenizer().encode(decomposed);
        
        // Both should produce tokens
        assertTrue(tokens1.length() > 0, variant.name() + " should tokenize precomposed");
        assertTrue(tokens2.length() > 0, variant.name() + " should tokenize decomposed");
        
        // Both should round-trip correctly
        assertEquals(precomposed, variant.tokenizer().decode(tokens1),
            variant.name() + " should round-trip precomposed");
        assertEquals(decomposed, variant.tokenizer().decode(tokens2),
            variant.name() + " should round-trip decomposed");
    }

    // ==================== EDGE CASE TESTS ====================

    @ParameterizedTest(name = "{0} - single characters")
    @MethodSource("provideTokenizerVariants")
    void testSingleCharacters(TokenizerVariant variant) {
        String[] chars = {"a", "1", "!", "@", "#", "$", "%"};
        
        for (String ch : chars) {
            IntSequence tokens = variant.tokenizer().encode(ch);
            assertTrue(tokens.length() > 0,
                variant.name() + " should tokenize single character: " + ch);
            
            String decoded = variant.tokenizer().decode(tokens);
            assertEquals(ch, decoded,
                variant.name() + " should round-trip single character: " + ch);
        }
    }

    @ParameterizedTest(name = "{0} - regex special characters")
    @MethodSource("provideTokenizerVariants")
    void testRegexSpecialCharacters(TokenizerVariant variant) {
        String[] texts = {
            "a*b",      // Asterisk
            "a+b",      // Plus
            "a?b",      // Question mark
            "[test]",   // Brackets
            "(test)",   // Parentheses
            "{test}",   // Braces
            "a|b",      // Pipe
            "^test",    // Caret
            "test$",    // Dollar
            "a.b",      // Dot
        };
        
        for (String text : texts) {
            IntSequence tokens = variant.tokenizer().encode(text);
            assertTrue(tokens.length() > 0,
                variant.name() + " should tokenize regex special chars: " + text);
            
            String decoded = variant.tokenizer().decode(tokens);
            assertEquals(text, decoded,
                variant.name() + " should round-trip regex special chars: " + text);
        }
    }

    @ParameterizedTest(name = "{0} - long text")
    @MethodSource("provideTokenizerVariants")
    void testLongText(TokenizerVariant variant) {
        String longWord = "a".repeat(1000);
        IntSequence tokens = variant.tokenizer().encode(longWord);
        assertTrue(tokens.length() > 0,
            variant.name() + " should tokenize long text");
        
        String decoded = variant.tokenizer().decode(tokens);
        assertEquals(longWord, decoded,
            variant.name() + " should round-trip long text");
    }

    @ParameterizedTest(name = "{0} - repeated patterns")
    @MethodSource("provideTokenizerVariants")
    void testRepeatedPatterns(TokenizerVariant variant) {
        String[] patterns = {
            "abababab",
            "xyzxyzxyz",
            "123123123"
        };
        
        for (String pattern : patterns) {
            IntSequence tokens = variant.tokenizer().encode(pattern);
            assertTrue(tokens.length() > 0,
                variant.name() + " should tokenize repeated pattern: " + pattern);
            
            String decoded = variant.tokenizer().decode(tokens);
            assertEquals(pattern, decoded,
                variant.name() + " should round-trip repeated pattern: " + pattern);
        }
    }

    // ==================== REAL-WORLD EXAMPLE TESTS ====================

    @ParameterizedTest(name = "{0} - code snippet")
    @MethodSource("provideTokenizerVariants")
    void testCodeSnippet(TokenizerVariant variant) {
        String code = "def hello():\n    return 'world'";
        IntSequence tokens = variant.tokenizer().encode(code);
        assertTrue(tokens.length() > 0,
            variant.name() + " should tokenize code");
        
        String decoded = variant.tokenizer().decode(tokens);
        assertEquals(code, decoded,
            variant.name() + " should round-trip code");
    }

    @ParameterizedTest(name = "{0} - JSON")
    @MethodSource("provideTokenizerVariants")
    void testJson(TokenizerVariant variant) {
        String json = "{\"key\": \"value\", \"num\": 42}";
        IntSequence tokens = variant.tokenizer().encode(json);
        assertTrue(tokens.length() > 0,
            variant.name() + " should tokenize JSON");
        
        String decoded = variant.tokenizer().decode(tokens);
        assertEquals(json, decoded,
            variant.name() + " should round-trip JSON");
    }

    @ParameterizedTest(name = "{0} - URL")
    @MethodSource("provideTokenizerVariants")
    void testUrl(TokenizerVariant variant) {
        String url = "https://example.com/path?query=value&foo=bar";
        IntSequence tokens = variant.tokenizer().encode(url);
        assertTrue(tokens.length() > 0,
            variant.name() + " should tokenize URL");
        
        String decoded = variant.tokenizer().decode(tokens);
        assertEquals(url, decoded,
            variant.name() + " should round-trip URL");
    }

    @ParameterizedTest(name = "{0} - email")
    @MethodSource("provideTokenizerVariants")
    void testEmail(TokenizerVariant variant) {
        String email = "user.name+tag@example.co.uk";
        IntSequence tokens = variant.tokenizer().encode(email);
        assertTrue(tokens.length() > 0,
            variant.name() + " should tokenize email");
        
        String decoded = variant.tokenizer().decode(tokens);
        assertEquals(email, decoded,
            variant.name() + " should round-trip email");
    }

    @ParameterizedTest(name = "{0} - math expression")
    @MethodSource("provideTokenizerVariants")
    void testMathExpression(TokenizerVariant variant) {
        String math = "E = mc^2";
        IntSequence tokens = variant.tokenizer().encode(math);
        assertTrue(tokens.length() > 0,
            variant.name() + " should tokenize math expression");
        
        String decoded = variant.tokenizer().decode(tokens);
        assertEquals(math, decoded,
            variant.name() + " should round-trip math expression");
    }

    @ParameterizedTest(name = "{0} - currency")
    @MethodSource("provideTokenizerVariants")
    void testCurrency(TokenizerVariant variant) {
        String currency = "Price: $1,234.56";
        IntSequence tokens = variant.tokenizer().encode(currency);
        assertTrue(tokens.length() > 0,
            variant.name() + " should tokenize currency");
        
        String decoded = variant.tokenizer().decode(tokens);
        assertEquals(currency, decoded,
            variant.name() + " should round-trip currency");
    }

    @ParameterizedTest(name = "{0} - date")
    @MethodSource("provideTokenizerVariants")
    void testDate(TokenizerVariant variant) {
        String date = "2024-01-15";
        IntSequence tokens = variant.tokenizer().encode(date);
        assertTrue(tokens.length() > 0,
            variant.name() + " should tokenize date");
        
        String decoded = variant.tokenizer().decode(tokens);
        assertEquals(date, decoded,
            variant.name() + " should round-trip date");
    }

    @ParameterizedTest(name = "{0} - time")
    @MethodSource("provideTokenizerVariants")
    void testTime(TokenizerVariant variant) {
        String time = "14:30:00";
        IntSequence tokens = variant.tokenizer().encode(time);
        assertTrue(tokens.length() > 0,
            variant.name() + " should tokenize time");
        
        String decoded = variant.tokenizer().decode(tokens);
        assertEquals(time, decoded,
            variant.name() + " should round-trip time");
    }

    @ParameterizedTest(name = "{0} - phone number")
    @MethodSource("provideTokenizerVariants")
    void testPhoneNumber(TokenizerVariant variant) {
        String phone = "+1-234-567-8900";
        IntSequence tokens = variant.tokenizer().encode(phone);
        assertTrue(tokens.length() > 0,
            variant.name() + " should tokenize phone number");
        
        String decoded = variant.tokenizer().decode(tokens);
        assertEquals(phone, decoded,
            variant.name() + " should round-trip phone number");
    }

    // ==================== TOKENIZER VARIANT BEHAVIOR COMPARISON ====================

    @Test
    void testAllVariantsProduceConsistentResults() {
        String text = "Hello world 123";
        
        for (TokenizerVariant variant : variants) {
            IntSequence tokens = variant.tokenizer().encode(text);
            String decoded = variant.tokenizer().decode(tokens);
            assertEquals(text, decoded,
                variant.name() + " should round-trip consistently");
        }
    }

    @Test
    void testAllVariantsHandleUnicode() {
        String text = "Hello 世界 👋";
        
        for (TokenizerVariant variant : variants) {
            IntSequence tokens = variant.tokenizer().encode(text);
            String decoded = variant.tokenizer().decode(tokens);
            assertEquals(text, decoded,
                variant.name() + " should handle Unicode");
        }
    }

    @Test
    void testAllVariantsHandleContractions() {
        String text = "I'm don't you're";
        
        for (TokenizerVariant variant : variants) {
            IntSequence tokens = variant.tokenizer().encode(text);
            String decoded = variant.tokenizer().decode(tokens);
            assertEquals(text, decoded,
                variant.name() + " should handle contractions");
        }
    }

    // ==================== HELPER METHODS ====================

    private static Stream<Arguments> provideTokenizerVariants() {
        return variants.stream().map(Arguments::of);
    }

    private record TokenizerVariant(String name, Tokenizer tokenizer) {}
}
