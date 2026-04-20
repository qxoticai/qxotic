package com.qxotic.toknroll.testkit;

import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.TokenizationModel;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Tokenizers;
import com.qxotic.toknroll.loaders.TiktokenLoaders;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

public final class TiktokenFixtures {

    private TiktokenFixtures() {}

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
            "'(?:[sdmt]|ll|ve|re)| ?\\p{L}++| ?\\p{N}++|"
                    + " ?[^\\s\\p{L}\\p{N}]++|\\s++$|\\s+(?!\\S)|\\s";
    private static final String CL100K_PATTERN =
            "'(?i:[sdmt]|ll|ve|re)|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?+\\p{L}++|\\p{N}{1,3}+| ?[^\\s\\p{L}\\p{N}]++[\\r"
                    + "\\n"
                    + "]*+|\\s++$|\\s*[\\r"
                    + "\\n"
                    + "]|\\s+(?!\\S)|\\s";
    private static final String O200K_PATTERN =
            String.join(
                    "|",
                    "[^\\r"
                        + "\\n"
                        + "\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
                    "[^\\r"
                        + "\\n"
                        + "\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
                    "\\p{N}{1,3}",
                    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*",
                    "\\s*[\\r\\n]+",
                    "\\s+(?!\\S)",
                    "\\s+");

    public record EncodingFixture(
            String name,
            String fileName,
            String hash,
            String pattern,
            Map<String, Integer> specialTokens) {}

    public record NamedTokenizer(String name, Tokenizer tokenizer) {}

    private static final List<EncodingFixture> ENCODINGS =
            List.of(
                    new EncodingFixture(
                            "r50k_base",
                            "r50k_base.tiktoken",
                            R50K_BASE_HASH,
                            R50K_PATTERN,
                            Map.of(ENDOFTEXT, 50256)),
                    new EncodingFixture(
                            "p50k_base",
                            "p50k_base.tiktoken",
                            P50K_BASE_HASH,
                            R50K_PATTERN,
                            Map.of(ENDOFTEXT, 50256)),
                    new EncodingFixture(
                            "p50k_edit",
                            "p50k_base.tiktoken",
                            P50K_BASE_HASH,
                            R50K_PATTERN,
                            Map.of(
                                    ENDOFTEXT, 50256,
                                    FIM_PREFIX, 50281,
                                    FIM_MIDDLE, 50282,
                                    FIM_SUFFIX, 50283)),
                    new EncodingFixture(
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
                    new EncodingFixture(
                            "o200k_base",
                            "o200k_base.tiktoken",
                            O200K_BASE_HASH,
                            O200K_PATTERN,
                            Map.of(ENDOFTEXT, 199999, ENDOFPROMPT, 200018)));

    private static final Map<String, Map<String, Integer>> MERGEABLE_RANKS_CACHE = new HashMap<>();

    // Cache for tokenizer instances to avoid recreating them for each test
    private static final Map<String, Tokenizer> TOKENIZER_CACHE = new HashMap<>();

    public static List<EncodingFixture> encodings() {
        return ENCODINGS;
    }

    public static EncodingFixture encoding(String name) {
        return ENCODINGS.stream().filter(e -> e.name().equals(name)).findFirst().orElseThrow();
    }

    public static Tokenizer createJtokkitTokenizer(String encodingName) {
        return TOKENIZER_CACHE.computeIfAbsent(
                encodingName, TiktokenFixtures::createTikTokenTokenizer);
    }

    private static Tokenizer BPE_R50K_TOKENIZER = null;
    private static final Map<String, Tokenizer> BPE_TOKENIZER_CACHE = new HashMap<>();

    public static Tokenizer createBpeR50kTokenizer() {
        if (BPE_R50K_TOKENIZER == null) {
            BPE_R50K_TOKENIZER = createBpeTokenizer("r50k_base");
        }
        return BPE_R50K_TOKENIZER;
    }

    public static Tokenizer createBpeTokenizer(String encodingName) {
        return BPE_TOKENIZER_CACHE.computeIfAbsent(
                encodingName, TiktokenFixtures::createTikTokenTokenizer);
    }

    public static Tokenizer createTikTokenTokenizer(String encodingName, Splitter splitter) {
        EncodingFixture fixture = encoding(encodingName);
        return createTikTokenTokenizer(
                mergeableRanks(encodingName), fixture.specialTokens(), splitter);
    }

    public static Tokenizer createTikTokenTokenizer(String encodingName) {
        EncodingFixture fixture = encoding(encodingName);
        return createTikTokenTokenizer(
                mergeableRanks(encodingName),
                fixture.specialTokens(),
                Splitter.regex(
                        Pattern.compile(fixture.pattern(), Pattern.UNICODE_CHARACTER_CLASS)));
    }

    public static Tokenizer createTikTokenTokenizer(
            Map<String, Integer> ranks, Map<String, Integer> specials, Splitter splitter) {
        TokenizationModel model =
                Tokenizers.tikTokenModel(
                        TiktokenLoaders.vocabulary(ranks, specials),
                        TiktokenLoaders.mergeRules(ranks));
        return Tokenizers.pipeline(model).splitter(splitter).build();
    }

    private static List<NamedTokenizer> ALL_JTOKKIT_TOKENIZERS = null;

    public static List<NamedTokenizer> createAllJtokkitTokenizers() {
        if (ALL_JTOKKIT_TOKENIZERS == null) {
            ALL_JTOKKIT_TOKENIZERS =
                    ENCODINGS.stream()
                            .map(
                                    e ->
                                            new NamedTokenizer(
                                                    "jtokkit-" + e.name(),
                                                    createJtokkitTokenizer(e.name())))
                            .toList();
        }
        return ALL_JTOKKIT_TOKENIZERS;
    }

    private static Map<String, Integer> loadMergeableRanks(String fileName, String expectedHash) {
        return MERGEABLE_RANKS_CACHE.computeIfAbsent(
                fileName,
                key -> {
                    try {
                        return TiktokenLoaders.loadMergeableRanks(
                                resourcePath(fileName).toString(), expectedHash);
                    } catch (Exception e) {
                        throw new IllegalStateException(
                                "Failed to load mergeable ranks for " + fileName, e);
                    }
                });
    }

    public static Map<String, Integer> mergeableRanks(String encodingName) {
        EncodingFixture fixture = encoding(encodingName);
        return Collections.unmodifiableMap(loadMergeableRanks(fixture.fileName(), fixture.hash()));
    }

    public static Map<String, Integer> specialTokens(String encodingName) {
        return Collections.unmodifiableMap(encoding(encodingName).specialTokens());
    }

    public static Pattern splitPattern(String encodingName) {
        return Pattern.compile(encoding(encodingName).pattern(), Pattern.UNICODE_CHARACTER_CLASS);
    }

    private static Path resourcePath(String fileName) {
        try {
            return Path.of(
                    TiktokenFixtures.class
                            .getClassLoader()
                            .getResource("tiktoken/" + fileName)
                            .toURI());
        } catch (URISyntaxException e) {
            throw new IllegalStateException("Failed to resolve " + fileName, e);
        }
    }
}
