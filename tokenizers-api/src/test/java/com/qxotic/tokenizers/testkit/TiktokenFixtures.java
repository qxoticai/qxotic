package com.qxotic.tokenizers.testkit;

import com.qxotic.tokenizers.Tokenizer;
import com.qxotic.tokenizers.advanced.Normalizer;
import com.qxotic.tokenizers.impl.ClassicBPE;
import com.qxotic.tokenizers.impl.RegexSplitter;
import com.qxotic.tokenizers.impl.Tiktoken;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
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

    public record EncodingFixture(
            String name,
            String fileName,
            String hash,
            String pattern,
            java.util.Map<String, Integer> specialTokens) {}

    public record NamedTokenizer(String name, Tokenizer tokenizer) {}

    private static final List<EncodingFixture> ENCODINGS =
            List.of(
                    new EncodingFixture(
                            "r50k_base",
                            "r50k_base.tiktoken",
                            R50K_BASE_HASH,
                            R50K_PATTERN,
                            java.util.Map.of(ENDOFTEXT, 50256)),
                    new EncodingFixture(
                            "p50k_base",
                            "p50k_base.tiktoken",
                            P50K_BASE_HASH,
                            R50K_PATTERN,
                            java.util.Map.of(ENDOFTEXT, 50256)),
                    new EncodingFixture(
                            "p50k_edit",
                            "p50k_base.tiktoken",
                            P50K_BASE_HASH,
                            R50K_PATTERN,
                            java.util.Map.of(
                                    ENDOFTEXT, 50256,
                                    FIM_PREFIX, 50281,
                                    FIM_MIDDLE, 50282,
                                    FIM_SUFFIX, 50283)),
                    new EncodingFixture(
                            "cl100k_base",
                            "cl100k_base.tiktoken",
                            CL100K_BASE_HASH,
                            CL100K_PATTERN,
                            java.util.Map.of(
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
                            java.util.Map.of(ENDOFTEXT, 199999, ENDOFPROMPT, 200018)));

    private static final java.util.Map<String, java.util.Map<String, Integer>>
            MERGEABLE_RANKS_CACHE = new HashMap<>();
    
    // Cache for tokenizer instances to avoid recreating them for each test
    private static final java.util.Map<String, Tokenizer> TOKENIZER_CACHE = new HashMap<>();

    public static List<EncodingFixture> encodings() {
        return ENCODINGS;
    }

    public static EncodingFixture encoding(String name) {
        return ENCODINGS.stream().filter(e -> e.name().equals(name)).findFirst().orElseThrow();
    }

    public static Tokenizer createJtokkitTokenizer(String encodingName) {
        return TOKENIZER_CACHE.computeIfAbsent(
                encodingName,
                name -> {
                    EncodingFixture fixture = encoding(name);
                    java.util.Map<String, Integer> ranks =
                            loadMergeableRanks(fixture.fileName(), fixture.hash());
                    return Tiktoken.createFromTiktoken(
                            fixture.name(),
                            ranks,
                            Pattern.compile(fixture.pattern()),
                            fixture.specialTokens());
                });
    }

    private static Tokenizer CLASSIC_R50K_TOKENIZER = null;

    public static Tokenizer createClassicR50kTokenizer() {
        if (CLASSIC_R50K_TOKENIZER == null) {
            EncodingFixture fixture = encoding("r50k_base");
            java.util.Map<String, Integer> ranks =
                    loadMergeableRanks(fixture.fileName(), fixture.hash());
            CLASSIC_R50K_TOKENIZER =
                    ClassicBPE.classicFromTiktoken(
                            ranks,
                            fixture.specialTokens(),
                            Normalizer.IDENTITY,
                            RegexSplitter.create(fixture.pattern()));
        }
        return CLASSIC_R50K_TOKENIZER;
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

    private static java.util.Map<String, Integer> loadMergeableRanks(
            String fileName, String expectedHash) {
        return MERGEABLE_RANKS_CACHE.computeIfAbsent(
                fileName,
                key -> {
                    try {
                        return ClassicBPE.loadMergeableRanks(
                                resourcePath(fileName).toString(), expectedHash);
                    } catch (Exception e) {
                        throw new IllegalStateException(
                                "Failed to load mergeable ranks for " + fileName, e);
                    }
                });
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
