package ai.qxotic.tokenizers;

import ai.qxotic.tokenizers.impl.ClassicBPE;
import ai.qxotic.tokenizers.impl.Tiktoken;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

class TokenizerContractTest {
    private static final String ENDOFTEXT = "<|endoftext|>";
    private static final String FIM_PREFIX = "<|fim_prefix|>";
    private static final String FIM_MIDDLE = "<|fim_middle|>";
    private static final String FIM_SUFFIX = "<|fim_suffix|>";
    private static final String ENDOFPROMPT = "<|endofprompt|>";

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

    private static Path resourcePath(String fileName) {
        try {
            return Path.of(
                    TokenizerContractTest.class
                            .getClassLoader()
                            .getResource("tiktoken/" + fileName)
                            .toURI());
        } catch (URISyntaxException e) {
            throw new IllegalStateException("Failed to resolve " + fileName, e);
        }
    }

    private static final List<EncodingSpec> TIKTOKEN_ENCODINGS =
            List.of(
                    new EncodingSpec(
                            "r50k_base",
                            "r50k_base.tiktoken",
                            R50K_BASE_HASH,
                            R50K_PATTERN,
                            Map.of(ENDOFTEXT, 50256)),
                    new EncodingSpec(
                            "p50k_base",
                            "p50k_base.tiktoken",
                            P50K_BASE_HASH,
                            R50K_PATTERN,
                            Map.of(ENDOFTEXT, 50256)),
                    new EncodingSpec(
                            "p50k_edit",
                            "p50k_base.tiktoken",
                            P50K_BASE_HASH,
                            R50K_PATTERN,
                            Map.of(
                                    ENDOFTEXT, 50256,
                                    FIM_PREFIX, 50281,
                                    FIM_MIDDLE, 50282,
                                    FIM_SUFFIX, 50283)),
                    new EncodingSpec(
                            "cl100k_base",
                            "cl100k_base.tiktoken",
                            CL100K_BASE_HASH,
                            CL100K_PATTERN,
                            Map.of(
                                    ENDOFTEXT, 100257,
                                    FIM_PREFIX, 100258,
                                    FIM_MIDDLE, 100259,
                                    FIM_SUFFIX, 100260,
                                    ENDOFPROMPT, 100276)),
                    new EncodingSpec(
                            "o200k_base",
                            "o200k_base.tiktoken",
                            O200K_BASE_HASH,
                            O200K_PATTERN,
                            Map.of(ENDOFTEXT, 199999, ENDOFPROMPT, 200018)));

    private static final List<TokenizerSpec> TOKENIZERS = buildTokenizers();

    private static List<TokenizerSpec> buildTokenizers() {
        List<TokenizerSpec> specs = new ArrayList<>();
        Map<String, Map<String, Integer>> cache = new HashMap<>();

        Map<String, Integer> r50kRanks =
                loadMergeableRanks("r50k_base.tiktoken", R50K_BASE_HASH, cache);
        Map<String, Integer> classicSpecials = Map.of(ENDOFTEXT, 50256);
        specs.add(
                new TokenizerSpec(
                        "classic-bpe",
                        ClassicBPE.classicFromTiktoken(
                                r50kRanks, classicSpecials, Normalizer.IDENTITY, TextSplitter.IDENTITY),
                        true,
                        classicSpecials));

        for (EncodingSpec encoding : TIKTOKEN_ENCODINGS) {
            Map<String, Integer> ranks =
                    loadMergeableRanks(encoding.fileName(), encoding.hash(), cache);
            Pattern pattern = Pattern.compile(encoding.pattern());
            specs.add(
                    new TokenizerSpec(
                            "jtokkit-" + encoding.name(),
                            Tiktoken.createFromTiktoken(
                                    encoding.name(), ranks, pattern, encoding.specialTokens()),
                            false,
                            encoding.specialTokens()));
        }
        return List.copyOf(specs);
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

    private static Stream<TokenizerSpec> tokenizerSpecs() {
        return TOKENIZERS.stream();
    }

    private static Stream<String> roundTripTexts() {
        return Stream.of(
                "",
                "Hello world",
                "Iñtërnâtiônàlizætiøn",
                "こんにちは世界",
                "Whitespace\tand\nnewlines",
                "Symbols: !@#$%^&*()",
                "Emoji: 😀🎉✨",
                "Flags: 🇺🇸🇯🇵",
                "Family emoji: 👨‍👩‍👧‍👦");
    }

    private static Stream<Arguments> roundTripCases() {
        return tokenizerSpecs()
                .flatMap(spec -> roundTripTexts().map(text -> Arguments.of(spec, text)));
    }

    @ParameterizedTest(name = "roundTrip {0}")
    @MethodSource("roundTripCases")
    void encodeDecodeRoundTrip(TokenizerSpec spec, String text) {
        Tokenizer tokenizer = spec.tokenizer();
        IntSequence tokens = tokenizer.encode(text);
        String decoded = tokenizer.decode(tokens);
        Assertions.assertEquals(text, decoded, spec.name());
    }

    @ParameterizedTest(name = "countTokens {0}")
    @MethodSource("tokenizerSpecs")
    void countTokensMatchesEncodeLength(TokenizerSpec spec) {
        String text = "Token count check";
        Tokenizer tokenizer = spec.tokenizer();
        Assertions.assertEquals(
                tokenizer.encode(text).length(), tokenizer.countTokens(text), spec.name());
    }

    @ParameterizedTest(name = "tokensInVocab {0}")
    @MethodSource("tokenizerSpecs")
    void tokensAreInVocabulary(TokenizerSpec spec) {
        String text = "Tokens should exist";
        Tokenizer tokenizer = spec.tokenizer();
        IntSequence tokens = tokenizer.encode(text);
        for (int i = 0; i < tokens.length(); i++) {
            Assertions.assertTrue(
                    tokenizer.vocabulary().contains(tokens.intAt(i)), spec.name());
        }
    }

    @ParameterizedTest(name = "charSequence {0}")
    @MethodSource("tokenizerSpecs")
    void acceptsCharSequence(TokenizerSpec spec) {
        String text = "StringBuilder input";
        Tokenizer tokenizer = spec.tokenizer();
        IntSequence tokens = tokenizer.encode(text);
        Assertions.assertFalse(tokens.isEmpty(), spec.name());
    }

    @ParameterizedTest(name = "specialTokens {0}")
    @MethodSource("tokenizerSpecs")
    void specialTokenHandling(TokenizerSpec spec) {
        Tokenizer tokenizer = spec.tokenizer();
        String specialToken = spec.specialTokens().keySet().iterator().next();
        
        // Verify special token exists in vocabulary
        int specialId = spec.specialTokens().get(specialToken);
        Assertions.assertTrue(tokenizer.vocabulary().contains(specialId), 
                spec.name() + " should contain special token: " + specialToken);
        Assertions.assertEquals(specialToken, tokenizer.vocabulary().token(specialId),
                spec.name() + " special token should match");
    }

    @ParameterizedTest(name = "decodeBytes {0}")
    @MethodSource("tokenizerSpecs")
    void decodeBytesMatchUtf8(TokenizerSpec spec) {
        String text = "Byte-level check";
        Tokenizer tokenizer = spec.tokenizer();
        IntSequence tokens = tokenizer.encode(text);
        byte[] decodedBytes = tokenizer.decodeBytes(tokens);
        Assertions.assertArrayEquals(
                text.getBytes(java.nio.charset.StandardCharsets.UTF_8), decodedBytes, spec.name());
    }

    private record TokenizerSpec(
            String name,
            Tokenizer tokenizer,
            boolean supportsSpecialTokens,
            Map<String, Integer> specialTokens) {}

    private record EncodingSpec(
            String name,
            String fileName,
            String hash,
            String pattern,
            Map<String, Integer> specialTokens) {}
}
