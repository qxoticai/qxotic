package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.loaders.ModelSplitters;
import java.util.ArrayList;
import java.util.List;
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
 * <p>See: r50k_base/case_075 golden test - input {@code " \\u00850"} was split as {@code ["
 * \\u0085", "0"]} (2 chunks, 3 tokens) instead of tiktoken's {@code [" ", "\\u0085", "0"]} (3
 * chunks, 4 tokens).
 */
class SplitterUnicodeWhitespaceRegressionTest {

    // NEL (U+0085) - Next Line, Unicode whitespace missed by Java's default \s
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
    @MethodSource("modelSplitterCases")
    void modelSplittersHandleUnicodeWhitespace(String name, Splitter splitter) {
        for (String sample : UNICODE_WHITESPACE_SAMPLES) {
            List<String> chunks = tokens(splitter, sample);
            assertFullCoverage(sample, chunks, name);
        }
    }

    static Stream<Arguments> modelSplitterCases() {
        return Stream.of(
                Arguments.of("ModelSplitters.LLAMA3", ModelSplitters.LLAMA3),
                Arguments.of("ModelSplitters.QWEN2", ModelSplitters.QWEN2),
                Arguments.of("ModelSplitters.QWEN35", ModelSplitters.QWEN35));
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
