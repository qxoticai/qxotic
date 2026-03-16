package ai.llm4j.test;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.model.llm.llama.Timer;
import com.qxotic.tokenizers.IntSequence;
import com.qxotic.tokenizers.Tokenizer;
import com.qxotic.tokenizers.advanced.Normalizer;
import com.qxotic.tokenizers.impl.ClassicBPE;
import com.qxotic.tokenizers.impl.RegexSplitter;
import com.qxotic.tokenizers.impl.Tiktoken;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.FieldSource;
import org.junit.jupiter.params.provider.MethodSource;

@Disabled
public class GPT4TokenizerTest {

    static final String ENDOFTEXT = "<|endoftext|>";
    static final String FIM_PREFIX = "<|fim_prefix|>";
    static final String FIM_MIDDLE = "<|fim_middle|>";
    static final String FIM_SUFFIX = "<|fim_suffix|>";
    static final String ENDOFPROMPT = "<|endofprompt|>";

    /**
     * Regex pattern for GPT-2 tokenizer, optimized version. Original was: "'s|'t|'re|'ve|'m|'ll|'d|
     * ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+" This version uses possessive
     * quantifiers (++) for better performance
     */
    public static final String R50K_PATTERN =
            String.join(
                    "|",
                    // Contractions
                    "'(?:[sdmt]|ll|ve|re)",

                    // Letters with optional leading space
                    " ?\\p{L}++",

                    // Numbers with optional leading space
                    " ?\\p{N}++",

                    // Non-alphanumeric with optional leading space
                    " ?[^\\s\\p{L}\\p{N}]++",

                    // End of string whitespace
                    "\\s++$",

                    // Whitespace not followed by non-whitespace
                    "\\s+(?!\\S)",

                    // Single whitespace
                    "\\s");

    static Tokenizer R50K_BASE() throws IOException, InterruptedException {
        Map<String, Integer> mergeableRanks =
                ClassicBPE.loadMergeableRanks(
                        "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
                        "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930");
        return ClassicBPE.classicFromTiktoken(
                mergeableRanks,
                Map.of(ENDOFTEXT, 50256),
                Normalizer.IDENTITY,
                RegexSplitter.create(R50K_PATTERN));
    }

    static Tokenizer P50K_BASE() throws IOException, InterruptedException {
        Map<String, Integer> mergeableRanks =
                ClassicBPE.loadMergeableRanks(
                        "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
                        "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069");
        Tokenizer tokenizer =
                ClassicBPE.classicFromTiktoken(
                        mergeableRanks,
                        Map.of(ENDOFTEXT, 50256),
                        Normalizer.IDENTITY,
                        RegexSplitter.create(R50K_PATTERN));
        assert 50281 == tokenizer.vocabulary().size();
        return tokenizer;
    }

    static Tokenizer P50K_EDIT() throws IOException, InterruptedException {
        Map<String, Integer> mergeableRanks =
                ClassicBPE.loadMergeableRanks(
                        "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
                        "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069");
        return ClassicBPE.classicFromTiktoken(
                mergeableRanks,
                Map.of(
                        ENDOFTEXT, 50256,
                        FIM_PREFIX, 50281,
                        FIM_MIDDLE, 50282,
                        FIM_SUFFIX, 50283),
                Normalizer.IDENTITY,
                RegexSplitter.create(R50K_PATTERN));
    }

    /** Enhanced tokenizer pattern with more specific matching and possessive quantifiers */
    public static final String CL100K_PATTERN =
            String.join(
                    "|",
                    // Case-insensitive contractions
                    "'(?i:[sdmt]|ll|ve|re)",

                    // Letters with optional non-alphanumeric prefix
                    "[^\\r\\n\\p{L}\\p{N}]?+\\p{L}++",

                    // 1-3 digit numbers
                    "\\p{N}{1,3}+",

                    // Non-alphanumeric with optional space and newlines
                    " ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*+",

                    // End of string whitespace
                    "\\s++$",

                    // Newlines with optional whitespace
                    "\\s*[\\r\\n]",

                    // Whitespace not followed by non-whitespace
                    "\\s+(?!\\S)",

                    // Single whitespace
                    "\\s");

    public static final String O200K_PATTERN =
            String.join(
                    "|",
                    // Words starting with lowercase or modifier after optional non-alphanumeric
                    "[^\\r"
                        + "\\n"
                        + "\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",

                    // Words starting with uppercase or title case after optional non-alphanumeric
                    "[^\\r"
                        + "\\n"
                        + "\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",

                    // Numbers 1-3 digits
                    "\\p{N}{1,3}",

                    // Non-alphanumeric sequences with optional leading space and trailing newlines
                    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*",

                    // Whitespace followed by newlines
                    "\\s*[\\r\\n]+",

                    // Whitespace not followed by non-whitespace
                    "\\s+(?!\\S)",

                    // Other whitespace sequences
                    "\\s+");

    static Map<String, Integer> loadMergeableRanksQuiet(String blobPath, String expectedHash) {
        try {
            return ClassicBPE.loadMergeableRanks(blobPath, expectedHash);
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    static final Tokenizer CLASSIC_O200K_BASE;
    static final Tokenizer TIKTOKEN_O200K_BASE;

    static {
        Map<String, Integer> mergeableRanks =
                loadMergeableRanksQuiet(
                        "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
                        "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d");
        Map<String, Integer> specialTokens =
                Map.of(
                        ENDOFTEXT, 199999,
                        ENDOFPROMPT, 200018);
        TIKTOKEN_O200K_BASE =
                Tiktoken.createFromTiktoken(
                        "o200k_base",
                        mergeableRanks,
                        Pattern.compile(O200K_PATTERN),
                        specialTokens);
        CLASSIC_O200K_BASE =
                ClassicBPE.classicFromTiktoken(
                        mergeableRanks,
                        specialTokens,
                        Normalizer.IDENTITY,
                        RegexSplitter.create(O200K_PATTERN));
    }

    private static final Tokenizer CLASSIC_CL100K_BASE;
    private static final Tokenizer TIKTOKEN_CL100K_BASE;

    static {
        Map<String, Integer> mergeableRanks =
                loadMergeableRanksQuiet(
                        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
                        "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7");

        Map<String, Integer> specialTokens =
                Map.of(
                        ENDOFTEXT, 100257,
                        FIM_PREFIX, 100258,
                        FIM_MIDDLE, 100259,
                        FIM_SUFFIX, 100260,
                        ENDOFPROMPT, 100276);

        CLASSIC_CL100K_BASE =
                ClassicBPE.classicFromTiktoken(
                        mergeableRanks,
                        specialTokens,
                        Normalizer.IDENTITY,
                        RegexSplitter.create(CL100K_PATTERN));
        TIKTOKEN_CL100K_BASE =
                Tiktoken.createFromTiktoken(
                        "cl100k_base",
                        mergeableRanks,
                        Pattern.compile(CL100K_PATTERN),
                        specialTokens);
    }

    static Stream<Arguments> tokenizerFileCountHash() {
        return Stream.of(CLASSIC_CL100K_BASE, TIKTOKEN_CL100K_BASE)
                .flatMap(
                        tokenizer ->
                                Stream.of(
                                        Arguments.of(
                                                tokenizer, "shakespeare.txt", 22853, 2118913072),
                                        Arguments.of(
                                                tokenizer,
                                                "shakespeare_full.txt",
                                                1468752,
                                                -1230282261),
                                        Arguments.of(tokenizer, "enwik8", 25793088, 760361529)));
    }

    static final Path TEST_DATA = Path.of("/home/mukel/Desktop/playground/text/");

    @ParameterizedTest
    @MethodSource("tokenizerFileCountHash")
    void testGPT4(Tokenizer tokenizer, String file, int expectedTokenCount, int expectedHashCode)
            throws IOException, InterruptedException {
        String text = Files.readString(TEST_DATA.resolve(file));
        assertEquals(expectedTokenCount, tokenizer.countTokens(text));
        assertEquals(expectedHashCode, tokenizer.encode(text).hashCode());
    }

    static List<Tokenizer> O200K_BASE_TOKENIZERS = List.of(CLASSIC_O200K_BASE, TIKTOKEN_O200K_BASE);

    @ParameterizedTest
    @FieldSource("O200K_BASE_TOKENIZERS")
    void testGPT4o(Tokenizer tokenizer) {
        String text =
                "Many words map to one token, but some don't: indivisible.\n"
                    + "\n"
                    + "Unicode characters like emojis may be split into many tokens containing the"
                    + " underlying bytes: \uD83E\uDD1A\uD83C\uDFFE\n"
                    + "\n"
                    + "Sequences of characters commonly found next to each other may be grouped"
                    + " together: 1234567890";
        IntSequence expected =
                IntSequence.of(
                        12488, 6391, 4014, 316, 1001, 6602, 11, 889, 1236, 4128, 25, 3862, 181386,
                        364, 61064, 9862, 1299, 166700, 1340, 413, 12648, 1511, 1991, 20290, 15683,
                        290, 27899, 11643, 25, 93643, 248, 52622, 122, 279, 168191, 328, 9862,
                        22378, 2491, 2613, 316, 2454, 1273, 1340, 413, 73263, 4717, 25, 220, 7633,
                        19354, 29338, 15);
        IntSequence tokens = tokenizer.encode(text);
        //        expected = IntSequence.of(8607, 4339, 2472, 311, 832, 4037, 11, 719, 1063, 1541,
        // 956, 25, 3687, 23936, 382, 35020, 5885, 1093, 100166, 1253, 387, 6859, 1139, 1690, 11460,
        // 8649, 279, 16940, 5943, 25, 11410, 97, 248, 9468, 237, 122, 271, 1542, 45045, 315, 5885,
        // 17037, 1766, 1828, 311, 1855, 1023, 1253, 387, 41141, 3871, 25, 220, 4513, 10961, 16474,
        // 15);
        assertEquals(expected, tokens);
    }

    @Test
    void testSpeed() throws IOException {
        String text = Files.readString(Paths.get("/home/mukel/Downloads/enwik9"));
        try (var timer = Timer.log("encode")) {
            TIKTOKEN_O200K_BASE.encode(text);
        }
    }
}
