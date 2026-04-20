package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.gguf.ModelTextSplitters;
import com.qxotic.toknroll.impl.FastSplitters;
import com.qxotic.toknroll.loaders.ModelSplitters;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Regression tests for Unicode whitespace handling in splitters.
 *
 * <p>Java's default {@code \\s} only matches {@code [ \\t\\n\\x0B\\f\\r]} (ASCII whitespace).
 * Python's {@code regex} module (used by tiktoken) treats Unicode whitespace like U+0085 (NEL),
 * U+00A0 (NBSP), U+200B (ZWSP) as {@code \\s}. Without {@code Pattern.UNICODE_CHARACTER_CLASS},
 * Java splitters produce different chunk boundaries than tiktoken, leading to different BPE
 * outputs.
 *
 * <p>See: r50k_base/case_075 golden test — input {@code " \\u00850"} was split as {@code ["
 * \\u0085", "0"]} (2 chunks, 3 tokens) instead of tiktoken's {@code [" ", "\\u0085", "0"]} (3
 * chunks, 4 tokens).
 */
class SplitterUnicodeWhitespaceRegressionTest {

    // NEL (U+0085) — Next Line, Unicode whitespace missed by Java's default \s
    private static final char NEL = '\u0085';

    // Other Unicode whitespace characters that Java's default \s misses
    private static final char NBSP = '\u00A0';
    private static final char ZWSP = '\u200B';
    private static final char BOM = '\uFEFF';
    private static final char FIG_SPACE = '\u2007';
    private static final char NARROW_NBSP = '\u202F';
    private static final char IDEO_SPACE = '\u3000';

    private static final String[] UNICODE_WHITESPACE_SAMPLES = {
        " " + NEL + "0",
        "hello" + NEL + "world",
        NEL + "leading",
        "trailing" + NEL,
        " " + NBSP + "x" + NBSP + "y",
        "a" + ZWSP + "b" + ZWSP + "c",
        "x" + BOM + "y",
        "col" + FIG_SPACE + "val",
        "a" + NARROW_NBSP + "b",
        "hello" + IDEO_SPACE + "world",
        "" + NEL + NEL + NEL,
        " " + NEL + " " + NEL,
        "\t" + NEL + "\n" + NBSP,
    };

    @ParameterizedTest(name = "{0}")
    @MethodSource("splitterCases")
    void fastSplitterMatchesRegexOnUnicodeWhitespace(String name, Splitter fast, Splitter regex) {
        for (String sample : UNICODE_WHITESPACE_SAMPLES) {
            assertEquals(
                    tokens(regex, sample), tokens(fast, sample), name + " :: " + escape(sample));
        }
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("modelSplitterCases")
    void modelSplittersHandleUnicodeWhitespace(String name, Splitter splitter) {
        for (String sample : UNICODE_WHITESPACE_SAMPLES) {
            List<String> chunks = tokens(splitter, sample);
            assertFullCoverage(sample, chunks, name);
        }
    }

    static Stream<Arguments> splitterCases() {
        return Stream.of(
                Arguments.of("r50k", FastSplitters.r50k(), r50kRegex()),
                Arguments.of("cl100k", FastSplitters.cl100k(), cl100kRegex()),
                Arguments.of("o200k", FastSplitters.o200k(), o200kRegex()),
                Arguments.of("llama3", FastSplitters.llama3(), llama3Regex()),
                Arguments.of("qwen35", FastSplitters.qwen35(), qwen35Regex()),
                Arguments.of("tekken", FastSplitters.tekken(), tekkenRegex()));
    }

    static Stream<Arguments> modelSplitterCases() {
        return Stream.of(
                Arguments.of("ModelSplitters.LLAMA3", ModelSplitters.LLAMA3),
                Arguments.of("ModelSplitters.QWEN2", ModelSplitters.QWEN2),
                Arguments.of("ModelSplitters.QWEN35", ModelSplitters.QWEN35));
    }

    private static Splitter r50kRegex() {
        return Splitter.regex(
                Pattern.compile(
                        "'(?:[sdmt]|ll|ve|re)| ?\\p{L}++| ?\\p{N}++|"
                                + " ?[^\\s\\p{L}\\p{N}]++|\\s++$|\\s+(?!\\S)|\\s",
                        Pattern.UNICODE_CHARACTER_CLASS));
    }

    private static Splitter cl100kRegex() {
        return Splitter.regex(
                Pattern.compile(
                        "'(?i:[sdmt]|ll|ve|re)|[^\\r"
                                + "\\n"
                                + "\\p{L}\\p{N}]?+\\p{L}++|\\p{N}{1,3}+| ?[^\\s\\p{L}\\p{N}]++[\\r"
                                + "\\n"
                                + "]*+|\\s++$|\\s*[\\r"
                                + "\\n"
                                + "]|\\s+(?!\\S)|\\s",
                        Pattern.UNICODE_CHARACTER_CLASS));
    }

    private static Splitter o200kRegex() {
        return Splitter.regex(
                Pattern.compile(
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
                                "\\s+"),
                        Pattern.UNICODE_CHARACTER_CLASS));
    }

    private static Splitter llama3Regex() {
        return Splitter.regex(
                Pattern.compile(
                        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
                                + "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}"
                                + "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
                                + "|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                        Pattern.UNICODE_CHARACTER_CLASS));
    }

    private static Splitter qwen35Regex() {
        return Splitter.regex(
                Pattern.compile(
                        ModelTextSplitters.QWEN35_PATTERN, Pattern.UNICODE_CHARACTER_CLASS));
    }

    private static Splitter tekkenRegex() {
        return Splitter.regex(
                Pattern.compile(
                        "[^\\r"
                            + "\\n"
                            + "\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r"
                            + "\\n"
                            + "\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}|"
                            + " ?[^\\s\\p{L}\\p{N}]+[\\r"
                            + "\\n"
                            + "/]*|\\s*[\\r"
                            + "\\n"
                            + "]+|\\s+(?!\\S)|\\s+",
                        Pattern.UNICODE_CHARACTER_CLASS));
    }

    private static List<String> tokens(Splitter splitter, String text) {
        List<String> out = new ArrayList<>();
        splitter.splitAll(
                text, (source, start, end) -> out.add(source.subSequence(start, end).toString()));
        return out;
    }

    private static void assertFullCoverage(String input, List<String> chunks, String name) {
        assertEquals(
                input,
                String.join("", chunks),
                name + " :: chunks must cover full input: " + escape(input));
    }

    private static String escape(String s) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c < 0x20
                    || (c >= 0x7f && c <= 0xa0)
                    || c == 0x2007
                    || c == 0x202F
                    || c == 0x3000
                    || c == 0x200B
                    || c == 0xFEFF) {
                sb.append(String.format("\\u%04X", (int) c));
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }
}
