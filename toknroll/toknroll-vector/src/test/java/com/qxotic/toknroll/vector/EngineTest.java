package com.qxotic.toknroll.vector;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Toknroll;
import com.qxotic.toknroll.impl.ImplAccessor;
import com.qxotic.toknroll.testkit.GitRepoPaths;
import com.qxotic.toknroll.testkit.TiktokenFiles;
import com.qxotic.toknroll.testkit.corpus.Enwik8Corpus;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * The engine against core's own encoder, through the public API: every text gets the same ids from
 * both, from a {@code String} and from its UTF-8.
 */
class EngineTest {

    // as the tiktoken fixtures spell them, which core's scanner knows
    private static final String R50K =
            "'(?:[sdmt]|ll|ve|re)| ?\\p{L}++| ?\\p{N}++|"
                    + " ?[^\\s\\p{L}\\p{N}]++|\\s++$|\\s+(?!\\S)|\\s";
    private static final String CL100K =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}|"
                    + " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    private static final String O200K =
            String.join(
                    "|",
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
                            + "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
                    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+"
                            + "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
                    "\\p{N}{1,3}",
                    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*",
                    "\\s*[\\r\\n]+",
                    "\\s+(?!\\S)",
                    "\\s+");

    // the GGUF loader's patterns for the families core scans in stages, and Bailing's
    private static final String GPT2 =
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
    private static final String ASCII_BOTH =
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])";
    private static final String BODY =
            "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|%s| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
                    + "|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    private static final String QWEN2 = ASCII_BOTH + String.format(BODY, "\\p{N}");
    private static final String MINICPM5_MAIN = ASCII_BOTH + String.format(BODY, "\\p{N}+");
    private static final String BAILING =
            "'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}"
                    + "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+";

    /** One encoding as core encodes it and as the engine does. */
    record Pair(String name, Tokenizer reference, Tokenizer engine) {
        @Override
        public String toString() {
            return name;
        }
    }

    private static final Pair[] PAIRS = {
        pair("r50k_base", R50K),
        pair("cl100k_base", CL100K),
        pair("o200k_base", O200K),
        pair("bailingmoe", "cl100k_base", BAILING),
        staged("mellum2", "r50k_base", "\\p{N}", GPT2),
        staged("minicpm5", "cl100k_base", "\\p{N}{1,3}", MINICPM5_MAIN),
        staged("laguna", "cl100k_base", ImplAccessor.LAGUNA_NEWLINES, QWEN2)
    };

    /**
     * Two stages: the engine behind the scanner {@link Splitter#sequence} makes of them, against
     * the stages chained by hand, which nothing recognizes, so core splits with the regexes.
     */
    private static Pair staged(String name, String vocabulary, String first, String second) {
        Splitter one = Splitter.regex(Pattern.compile(first, Pattern.UNICODE_CHARACTER_CLASS));
        Splitter two = Splitter.regex(Pattern.compile(second, Pattern.UNICODE_CHARACTER_CLASS));
        Splitter chained =
                (text, start, end, consumer) ->
                        one.splitAll(
                                text, start, end, (t, s, e) -> two.splitAll(t, s, e, consumer));
        Tokenizer engine = tokenizer(vocabulary, Splitter.sequence(one, two));
        engine.encode("x");
        return new Pair(name, tokenizer(vocabulary, chained), engine);
    }

    static Stream<Pair> pairs() {
        return Stream.of(PAIRS);
    }

    private static Pair pair(String name, String regex) {
        return pair(name, name, regex);
    }

    private static Pair pair(String name, String vocabulary, String regex) {
        Tokenizer reference = tokenizer(vocabulary, regex);
        // the engine is resolved on a model's first encode, so this one never has one
        System.setProperty("toknroll.engine", "none");
        try {
            reference.encode("x");
        } finally {
            System.clearProperty("toknroll.engine");
        }
        Tokenizer engine = tokenizer(vocabulary, regex);
        engine.encode("x");
        return new Pair(name, reference, engine);
    }

    private static Tokenizer tokenizer(String name, String regex) {
        return tokenizer(
                name, Splitter.regex(Pattern.compile(regex, Pattern.UNICODE_CHARACTER_CLASS)));
    }

    private static Tokenizer tokenizer(String name, Splitter splitter) {
        Path file =
                GitRepoPaths.findGitRoot().resolve("test-fixtures/tiktoken/" + name + ".tiktoken");
        try (var reader = Files.newBufferedReader(file)) {
            Map<String, Integer> ranks = TiktokenFiles.loadMergeableRanks(reader);
            return Toknroll.pipeline(
                    splitter,
                    Toknroll.tiktokenModel(
                            ImplAccessor.reconstructTiktokenVocabulary(ranks, Map.of(), Set.of()),
                            ImplAccessor.reconstructTiktokenMergeRules(ranks)));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("pairs")
    void theEngineEncodesAndCoreDoesNot(Pair pair) {
        assertTrue(pair.engine().toString().contains("FastEncoder"), pair.engine().toString());
        assertFalse(
                pair.reference().toString().contains("encoded by"), pair.reference().toString());
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("pairs")
    void fuzz(Pair pair) {
        Random random = new Random(0xE9617EL);
        String[] pieces = BitSplitTest.PIECES;
        for (int round = 0; round < 2000; round++) {
            StringBuilder text = new StringBuilder();
            int count = random.nextInt(8) == 0 ? 1 + random.nextInt(5000) : 1 + random.nextInt(60);
            for (int i = 0; i < count; i++) {
                String piece = pieces[random.nextInt(pieces.length)];
                text.append(
                        random.nextInt(16) == 0 ? piece.repeat(1 + random.nextInt(120)) : piece);
            }
            same(pair, text.toString());
        }
    }

    @Tag("corpus")
    @ParameterizedTest(name = "{0}")
    @MethodSource("pairs")
    void enwik8(Pair pair) {
        same(pair, new String(Enwik8Corpus.load().data(), StandardCharsets.UTF_8));
    }

    private static void same(Pair pair, String text) {
        int[] expected = pair.reference().encode(text).toArray();
        assertArrayEquals(expected, pair.engine().encode(text).toArray(), () -> shown(text));
        byte[] utf8 = Utf8Reference.encode(text, 0, text.length(), 0);
        IntSequence fromBytes = pair.engine().encodeUtf8(utf8);
        assertArrayEquals(expected, fromBytes.toArray(), () -> "UTF-8: " + shown(text));
    }

    private static String shown(String text) {
        return text.length() > 200 ? text.substring(0, 200) + "..." : text;
    }
}
