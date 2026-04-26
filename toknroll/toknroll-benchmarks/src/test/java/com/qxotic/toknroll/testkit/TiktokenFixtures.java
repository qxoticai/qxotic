package com.qxotic.toknroll.testkit;

import com.knuddels.jtokkit.Encodings;
import com.knuddels.jtokkit.api.Encoding;
import com.knuddels.jtokkit.api.EncodingRegistry;
import com.knuddels.jtokkit.api.GptBytePairEncodingParams;
import com.knuddels.jtokkit.api.IntArrayList;
import com.qxotic.toknroll.ByteLevel;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Tokenizers;
import com.qxotic.toknroll.Vocabulary;
import com.qxotic.toknroll.loaders.TiktokenLoaders;
import java.net.URISyntaxException;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
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

    private static final Map<String, EncodingFixture> ENCODING_BY_NAME = buildEncodingMap();

    private static final Map<String, Map<String, Integer>> MERGEABLE_RANKS_CACHE = new HashMap<>();
    private static final Map<String, Tokenizer> TOKENIZER_CACHE = new HashMap<>();
    private static final Map<String, Tokenizer> JTOKKIT_CACHE = new HashMap<>();

    public static Tokenizer createJtokkitTokenizer(String encodingName) {
        return JTOKKIT_CACHE.computeIfAbsent(
                encodingName,
                name -> {
                    EncodingFixture fixture = encoding(name);
                    return createJtokkitTokenizerInternal(
                            name,
                            mergeableRanks(name),
                            splitPattern(name),
                            fixture.specialTokens());
                });
    }

    public static Tokenizer createTiktokenTokenizer(String encodingName) {
        return TOKENIZER_CACHE.computeIfAbsent(
                encodingName,
                name -> {
                    EncodingFixture fixture = encoding(name);
                    return createTiktokenTokenizer(
                            mergeableRanks(name),
                            fixture.specialTokens(),
                            Splitter.regex(
                                    Pattern.compile(
                                            fixture.pattern(), Pattern.UNICODE_CHARACTER_CLASS)));
                });
    }

    public static Tokenizer createTiktokenTokenizer(String encodingName, Splitter splitter) {
        EncodingFixture fixture = encoding(encodingName);
        return createTiktokenTokenizer(
                mergeableRanks(encodingName), fixture.specialTokens(), splitter);
    }

    public static Tokenizer createTiktokenTokenizer(
            Map<String, Integer> ranks, Map<String, Integer> specials, Splitter splitter) {
        return Tokenizers.pipeline(
                        Tokenizers.tiktokenModel(
                                TiktokenLoaders.vocabulary(ranks, specials),
                                TiktokenLoaders.mergeRules(ranks)))
                .splitter(splitter)
                .build();
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

    private static EncodingFixture encoding(String name) {
        EncodingFixture fixture = ENCODING_BY_NAME.get(name);
        if (fixture == null) {
            throw new IllegalArgumentException("Unsupported encoding: " + name);
        }
        return fixture;
    }

    private static Tokenizer createJtokkitTokenizerInternal(
            String name,
            Map<String, Integer> mergeableRanks,
            Pattern splitPattern,
            Map<String, Integer> specialTokens) {
        try {
            EncodingRegistry registry = Encodings.newLazyEncodingRegistry();
            Map<byte[], Integer> rawRanks = new HashMap<>();
            for (Map.Entry<String, Integer> e : mergeableRanks.entrySet()) {
                rawRanks.put(ByteLevel.decode(e.getKey()), e.getValue());
            }
            GptBytePairEncodingParams params =
                    new GptBytePairEncodingParams(name, splitPattern, rawRanks, specialTokens);
            registry.registerGptBytePairEncoding(params);
            Encoding encoding = registry.getEncoding(name).orElseThrow();
            Vocabulary vocabulary = TiktokenLoaders.vocabulary(mergeableRanks, specialTokens);
            return new JTokkitAdapter(encoding, vocabulary);
        } catch (Exception e) {
            throw new IllegalStateException("Failed to create JTokkit tokenizer for " + name, e);
        }
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

    private static Path resourcePath(String fileName) {
        java.net.URL resource =
                TiktokenFixtures.class.getClassLoader().getResource("tiktoken/" + fileName);
        if (resource == null) {
            throw new IllegalStateException("Missing tiktoken fixture: " + fileName);
        }
        try {
            return Path.of(resource.toURI());
        } catch (URISyntaxException e) {
            throw new IllegalStateException("Failed to resolve " + fileName, e);
        }
    }

    private static Map<String, EncodingFixture> buildEncodingMap() {
        Map<String, EncodingFixture> map = new HashMap<>();
        for (EncodingFixture fixture : ENCODINGS) {
            map.put(fixture.name(), fixture);
        }
        return Map.copyOf(map);
    }

    private static final class JTokkitAdapter implements Tokenizer {
        private final Encoding encoding;
        private final Vocabulary vocabulary;

        JTokkitAdapter(Encoding encoding, Vocabulary vocabulary) {
            this.encoding = Objects.requireNonNull(encoding, "encoding");
            this.vocabulary = Objects.requireNonNull(vocabulary, "vocabulary");
        }

        @Override
        public Vocabulary vocabulary() {
            return vocabulary;
        }

        @Override
        public int countTokens(CharSequence text, int startInclusive, int endExclusive) {
            return encoding.countTokensOrdinary(
                    text.subSequence(startInclusive, endExclusive).toString());
        }

        @Override
        public void encodeInto(
                CharSequence text, int startInclusive, int endExclusive, IntSequence.Builder out) {
            String slice = text.subSequence(startInclusive, endExclusive).toString();
            IntArrayList encoded = encoding.encodeOrdinary(slice);
            int size = encoded.size();
            out.ensureCapacity(out.size() + size);
            for (int i = 0; i < size; i++) {
                out.add(encoded.get(i));
            }
        }

        @Override
        public int countBytes(IntSequence tokens) {
            IntArrayList list = new IntArrayList(tokens.length());
            for (int i = 0; i < tokens.length(); i++) {
                list.add(tokens.intAt(i));
            }
            return encoding.decodeBytes(list).length;
        }

        @Override
        public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
            int length = tokens.length();
            if (tokenStartIndex == length) {
                return 0;
            }
            int consumed = 0;
            IntArrayList oneToken = new IntArrayList(1);
            for (int i = tokenStartIndex; i < length; i++) {
                oneToken.clear();
                oneToken.add(tokens.intAt(i));
                byte[] chunk = encoding.decodeBytes(oneToken);
                if (chunk.length > out.remaining()) {
                    if (consumed == 0) {
                        throw new IllegalArgumentException(
                                "Not enough output space for token at index " + i);
                    }
                    break;
                }
                out.put(chunk);
                consumed++;
            }
            return consumed;
        }
    }
}
