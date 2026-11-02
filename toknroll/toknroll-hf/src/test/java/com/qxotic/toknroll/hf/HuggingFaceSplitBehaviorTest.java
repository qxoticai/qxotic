package com.qxotic.toknroll.hf;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.Splitter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * The {@code Split} pre-tokenizer's pieces, for every behavior, as Hugging Face tokenizers 0.23.2
 * cuts them (generated from {@code pre_tokenizers.Split(...).pre_tokenize_str}). A delimiter with
 * no neighbor to merge into is a piece of its own: text ending in Laguna's newline run once lost
 * it.
 */
class HuggingFaceSplitBehaviorTest {

    static Stream<Arguments> cases() {
        return Stream.of(
                hf("String", " ", "MergedWithPrevious", false, "a  b", "a ", " ", "b"),
                hf("String", " ", "MergedWithPrevious", false, " a", " ", "a"),
                hf("String", " ", "MergedWithPrevious", false, "a ", "a "),
                hf("String", " ", "MergedWithPrevious", false, "  ", " ", " "),
                hf("String", " ", "MergedWithPrevious", false, "a", "a"),
                hf("String", " ", "MergedWithPrevious", false, " a b  ", " ", "a ", "b ", " "),
                hf("String", " ", "MergedWithPrevious", false, "a\n", "a\n"),
                hf("String", " ", "MergedWithPrevious", false, "a\nb\n", "a\nb\n"),
                hf("String", " ", "MergedWithPrevious", false, "\n\na\r\nb", "\n\na\r\nb"),
                hf("String", " ", "MergedWithPrevious", false, "\n", "\n"),
                hf("String", " ", "MergedWithPrevious", false, "a\n\n", "a\n\n"),
                hf("String", " ", "MergedWithPrevious", false, "x \n", "x ", "\n"),
                hf("String", " ", "MergedWithPrevious", true, "a  b", "a", " ", " b"),
                hf("String", " ", "MergedWithPrevious", true, " a", " a"),
                hf("String", " ", "MergedWithPrevious", true, "a ", "a", " "),
                hf("String", " ", "MergedWithPrevious", true, "  ", " ", " "),
                hf("String", " ", "MergedWithPrevious", true, "a", "a"),
                hf("String", " ", "MergedWithPrevious", true, " a b  ", " a", " b", " ", " "),
                hf("String", " ", "MergedWithPrevious", true, "a\n", "a\n"),
                hf("String", " ", "MergedWithPrevious", true, "a\nb\n", "a\nb\n"),
                hf("String", " ", "MergedWithPrevious", true, "\n\na\r\nb", "\n\na\r\nb"),
                hf("String", " ", "MergedWithPrevious", true, "\n", "\n"),
                hf("String", " ", "MergedWithPrevious", true, "a\n\n", "a\n\n"),
                hf("String", " ", "MergedWithPrevious", true, "x \n", "x", " \n"),
                hf("String", " ", "MergedWithNext", false, "a  b", "a", " ", " b"),
                hf("String", " ", "MergedWithNext", false, " a", " a"),
                hf("String", " ", "MergedWithNext", false, "a ", "a", " "),
                hf("String", " ", "MergedWithNext", false, "  ", " ", " "),
                hf("String", " ", "MergedWithNext", false, "a", "a"),
                hf("String", " ", "MergedWithNext", false, " a b  ", " a", " b", " ", " "),
                hf("String", " ", "MergedWithNext", false, "a\n", "a\n"),
                hf("String", " ", "MergedWithNext", false, "a\nb\n", "a\nb\n"),
                hf("String", " ", "MergedWithNext", false, "\n\na\r\nb", "\n\na\r\nb"),
                hf("String", " ", "MergedWithNext", false, "\n", "\n"),
                hf("String", " ", "MergedWithNext", false, "a\n\n", "a\n\n"),
                hf("String", " ", "MergedWithNext", false, "x \n", "x", " \n"),
                hf("String", " ", "MergedWithNext", true, "a  b", "a ", " ", "b"),
                hf("String", " ", "MergedWithNext", true, " a", " ", "a"),
                hf("String", " ", "MergedWithNext", true, "a ", "a "),
                hf("String", " ", "MergedWithNext", true, "  ", " ", " "),
                hf("String", " ", "MergedWithNext", true, "a", "a"),
                hf("String", " ", "MergedWithNext", true, " a b  ", " ", "a ", "b ", " "),
                hf("String", " ", "MergedWithNext", true, "a\n", "a\n"),
                hf("String", " ", "MergedWithNext", true, "a\nb\n", "a\nb\n"),
                hf("String", " ", "MergedWithNext", true, "\n\na\r\nb", "\n\na\r\nb"),
                hf("String", " ", "MergedWithNext", true, "\n", "\n"),
                hf("String", " ", "MergedWithNext", true, "a\n\n", "a\n\n"),
                hf("String", " ", "MergedWithNext", true, "x \n", "x ", "\n"),
                hf("String", " ", "Isolated", false, "a  b", "a", " ", " ", "b"),
                hf("String", " ", "Isolated", false, " a", " ", "a"),
                hf("String", " ", "Isolated", false, "a ", "a", " "),
                hf("String", " ", "Isolated", false, "  ", " ", " "),
                hf("String", " ", "Isolated", false, "a", "a"),
                hf("String", " ", "Isolated", false, " a b  ", " ", "a", " ", "b", " ", " "),
                hf("String", " ", "Isolated", false, "a\n", "a\n"),
                hf("String", " ", "Isolated", false, "a\nb\n", "a\nb\n"),
                hf("String", " ", "Isolated", false, "\n\na\r\nb", "\n\na\r\nb"),
                hf("String", " ", "Isolated", false, "\n", "\n"),
                hf("String", " ", "Isolated", false, "a\n\n", "a\n\n"),
                hf("String", " ", "Isolated", false, "x \n", "x", " ", "\n"),
                hf("String", " ", "Isolated", true, "a  b", "a", " ", " ", "b"),
                hf("String", " ", "Isolated", true, " a", " ", "a"),
                hf("String", " ", "Isolated", true, "a ", "a", " "),
                hf("String", " ", "Isolated", true, "  ", " ", " "),
                hf("String", " ", "Isolated", true, "a", "a"),
                hf("String", " ", "Isolated", true, " a b  ", " ", "a", " ", "b", " ", " "),
                hf("String", " ", "Isolated", true, "a\n", "a\n"),
                hf("String", " ", "Isolated", true, "a\nb\n", "a\nb\n"),
                hf("String", " ", "Isolated", true, "\n\na\r\nb", "\n\na\r\nb"),
                hf("String", " ", "Isolated", true, "\n", "\n"),
                hf("String", " ", "Isolated", true, "a\n\n", "a\n\n"),
                hf("String", " ", "Isolated", true, "x \n", "x", " ", "\n"),
                hf("String", " ", "Removed", false, "a  b", "a", "b"),
                hf("String", " ", "Removed", false, " a", "a"),
                hf("String", " ", "Removed", false, "a ", "a"),
                hf("String", " ", "Removed", false, "  "),
                hf("String", " ", "Removed", false, "a", "a"),
                hf("String", " ", "Removed", false, " a b  ", "a", "b"),
                hf("String", " ", "Removed", false, "a\n", "a\n"),
                hf("String", " ", "Removed", false, "a\nb\n", "a\nb\n"),
                hf("String", " ", "Removed", false, "\n\na\r\nb", "\n\na\r\nb"),
                hf("String", " ", "Removed", false, "\n", "\n"),
                hf("String", " ", "Removed", false, "a\n\n", "a\n\n"),
                hf("String", " ", "Removed", false, "x \n", "x", "\n"),
                hf("String", " ", "Removed", true, "a  b", " ", " "),
                hf("String", " ", "Removed", true, " a", " "),
                hf("String", " ", "Removed", true, "a ", " "),
                hf("String", " ", "Removed", true, "  ", " ", " "),
                hf("String", " ", "Removed", true, "a"),
                hf("String", " ", "Removed", true, " a b  ", " ", " ", " ", " "),
                hf("String", " ", "Removed", true, "a\n"),
                hf("String", " ", "Removed", true, "a\nb\n"),
                hf("String", " ", "Removed", true, "\n\na\r\nb"),
                hf("String", " ", "Removed", true, "\n"),
                hf("String", " ", "Removed", true, "a\n\n"),
                hf("String", " ", "Removed", true, "x \n", " "),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithPrevious", false, "a  b", "a  b"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithPrevious", false, " a", " a"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithPrevious", false, "a ", "a "),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithPrevious", false, "  ", "  "),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithPrevious", false, "a", "a"),
                hf(
                        "Regex",
                        "(?:\\r?\\n)+(?!\\r?\\n)",
                        "MergedWithPrevious",
                        false,
                        " a b  ",
                        " a b  "),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithPrevious", false, "a\n", "a\n"),
                hf(
                        "Regex",
                        "(?:\\r?\\n)+(?!\\r?\\n)",
                        "MergedWithPrevious",
                        false,
                        "a\nb\n",
                        "a\n",
                        "b\n"),
                hf(
                        "Regex",
                        "(?:\\r?\\n)+(?!\\r?\\n)",
                        "MergedWithPrevious",
                        false,
                        "\n\na\r\nb",
                        "\n\n",
                        "a\r\n",
                        "b"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithPrevious", false, "\n", "\n"),
                hf(
                        "Regex",
                        "(?:\\r?\\n)+(?!\\r?\\n)",
                        "MergedWithPrevious",
                        false,
                        "a\n\n",
                        "a\n\n"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithPrevious", false, "x \n", "x \n"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithPrevious", true, "a  b", "a  b"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithPrevious", true, " a", " a"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithPrevious", true, "a ", "a "),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithPrevious", true, "  ", "  "),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithPrevious", true, "a", "a"),
                hf(
                        "Regex",
                        "(?:\\r?\\n)+(?!\\r?\\n)",
                        "MergedWithPrevious",
                        true,
                        " a b  ",
                        " a b  "),
                hf(
                        "Regex",
                        "(?:\\r?\\n)+(?!\\r?\\n)",
                        "MergedWithPrevious",
                        true,
                        "a\n",
                        "a",
                        "\n"),
                hf(
                        "Regex",
                        "(?:\\r?\\n)+(?!\\r?\\n)",
                        "MergedWithPrevious",
                        true,
                        "a\nb\n",
                        "a",
                        "\nb",
                        "\n"),
                hf(
                        "Regex",
                        "(?:\\r?\\n)+(?!\\r?\\n)",
                        "MergedWithPrevious",
                        true,
                        "\n\na\r\nb",
                        "\n\na",
                        "\r\nb"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithPrevious", true, "\n", "\n"),
                hf(
                        "Regex",
                        "(?:\\r?\\n)+(?!\\r?\\n)",
                        "MergedWithPrevious",
                        true,
                        "a\n\n",
                        "a",
                        "\n\n"),
                hf(
                        "Regex",
                        "(?:\\r?\\n)+(?!\\r?\\n)",
                        "MergedWithPrevious",
                        true,
                        "x \n",
                        "x ",
                        "\n"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithNext", false, "a  b", "a  b"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithNext", false, " a", " a"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithNext", false, "a ", "a "),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithNext", false, "  ", "  "),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithNext", false, "a", "a"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithNext", false, " a b  ", " a b  "),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithNext", false, "a\n", "a", "\n"),
                hf(
                        "Regex",
                        "(?:\\r?\\n)+(?!\\r?\\n)",
                        "MergedWithNext",
                        false,
                        "a\nb\n",
                        "a",
                        "\nb",
                        "\n"),
                hf(
                        "Regex",
                        "(?:\\r?\\n)+(?!\\r?\\n)",
                        "MergedWithNext",
                        false,
                        "\n\na\r\nb",
                        "\n\na",
                        "\r\nb"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithNext", false, "\n", "\n"),
                hf(
                        "Regex",
                        "(?:\\r?\\n)+(?!\\r?\\n)",
                        "MergedWithNext",
                        false,
                        "a\n\n",
                        "a",
                        "\n\n"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithNext", false, "x \n", "x ", "\n"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithNext", true, "a  b", "a  b"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithNext", true, " a", " a"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithNext", true, "a ", "a "),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithNext", true, "  ", "  "),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithNext", true, "a", "a"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithNext", true, " a b  ", " a b  "),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithNext", true, "a\n", "a\n"),
                hf(
                        "Regex",
                        "(?:\\r?\\n)+(?!\\r?\\n)",
                        "MergedWithNext",
                        true,
                        "a\nb\n",
                        "a\n",
                        "b\n"),
                hf(
                        "Regex",
                        "(?:\\r?\\n)+(?!\\r?\\n)",
                        "MergedWithNext",
                        true,
                        "\n\na\r\nb",
                        "\n\n",
                        "a\r\n",
                        "b"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithNext", true, "\n", "\n"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithNext", true, "a\n\n", "a\n\n"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "MergedWithNext", true, "x \n", "x \n"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Isolated", false, "a  b", "a  b"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Isolated", false, " a", " a"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Isolated", false, "a ", "a "),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Isolated", false, "  ", "  "),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Isolated", false, "a", "a"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Isolated", false, " a b  ", " a b  "),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Isolated", false, "a\n", "a", "\n"),
                hf(
                        "Regex",
                        "(?:\\r?\\n)+(?!\\r?\\n)",
                        "Isolated",
                        false,
                        "a\nb\n",
                        "a",
                        "\n",
                        "b",
                        "\n"),
                hf(
                        "Regex",
                        "(?:\\r?\\n)+(?!\\r?\\n)",
                        "Isolated",
                        false,
                        "\n\na\r\nb",
                        "\n\n",
                        "a",
                        "\r\n",
                        "b"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Isolated", false, "\n", "\n"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Isolated", false, "a\n\n", "a", "\n\n"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Isolated", false, "x \n", "x ", "\n"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Isolated", true, "a  b", "a  b"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Isolated", true, " a", " a"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Isolated", true, "a ", "a "),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Isolated", true, "  ", "  "),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Isolated", true, "a", "a"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Isolated", true, " a b  ", " a b  "),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Isolated", true, "a\n", "a", "\n"),
                hf(
                        "Regex",
                        "(?:\\r?\\n)+(?!\\r?\\n)",
                        "Isolated",
                        true,
                        "a\nb\n",
                        "a",
                        "\n",
                        "b",
                        "\n"),
                hf(
                        "Regex",
                        "(?:\\r?\\n)+(?!\\r?\\n)",
                        "Isolated",
                        true,
                        "\n\na\r\nb",
                        "\n\n",
                        "a",
                        "\r\n",
                        "b"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Isolated", true, "\n", "\n"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Isolated", true, "a\n\n", "a", "\n\n"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Isolated", true, "x \n", "x ", "\n"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Removed", false, "a  b", "a  b"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Removed", false, " a", " a"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Removed", false, "a ", "a "),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Removed", false, "  ", "  "),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Removed", false, "a", "a"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Removed", false, " a b  ", " a b  "),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Removed", false, "a\n", "a"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Removed", false, "a\nb\n", "a", "b"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Removed", false, "\n\na\r\nb", "a", "b"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Removed", false, "\n"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Removed", false, "a\n\n", "a"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Removed", false, "x \n", "x "),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Removed", true, "a  b"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Removed", true, " a"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Removed", true, "a "),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Removed", true, "  "),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Removed", true, "a"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Removed", true, " a b  "),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Removed", true, "a\n", "\n"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Removed", true, "a\nb\n", "\n", "\n"),
                hf(
                        "Regex",
                        "(?:\\r?\\n)+(?!\\r?\\n)",
                        "Removed",
                        true,
                        "\n\na\r\nb",
                        "\n\n",
                        "\r\n"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Removed", true, "\n", "\n"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Removed", true, "a\n\n", "\n\n"),
                hf("Regex", "(?:\\r?\\n)+(?!\\r?\\n)", "Removed", true, "x \n", "\n"));
    }

    private static Arguments hf(
            String kind,
            String pattern,
            String behavior,
            boolean invert,
            String text,
            String... pieces) {
        return Arguments.of(kind, pattern, behavior, invert, text, List.of(pieces));
    }

    @ParameterizedTest(name = "{0} {1} {2} invert={3} {4}")
    @MethodSource("cases")
    void piecesAreHuggingFaces(
            String kind,
            String pattern,
            String behavior,
            boolean invert,
            String text,
            List<String> pieces) {
        assertEquals(pieces, pieces(split(kind, pattern, behavior, invert), text));
    }

    private static final String LAGUNA_NEWLINES = "(?:\\r?\\n)+(?!\\r?\\n)";

    /**
     * Laguna's newline stage takes a possessive regex of the same pieces (java.util.regex recursed
     * per newline on the file's own); every text of up to 8 chars over 'a', CR and LF cuts as
     * Hugging Face's MergedWithNext cuts at the file's regex.
     */
    @Test
    void lagunaNewlineStageCutsAsMergedWithNext() {
        Splitter split = split("Regex", LAGUNA_NEWLINES, "MergedWithNext", false);
        char[] alphabet = {'a', '\r', '\n'};
        for (int length = 0; length <= 8; length++) {
            for (int n = 0; n < Math.pow(alphabet.length, length); n++) {
                StringBuilder text = new StringBuilder();
                for (int i = 0, m = n; i < length; i++, m /= alphabet.length) {
                    text.append(alphabet[m % alphabet.length]);
                }
                assertEquals(
                        mergedWithNext(text.toString()),
                        pieces(split, text.toString()),
                        text.toString().replace("\r", "\\r").replace("\n", "\\n"));
            }
        }
    }

    /**
     * Laguna's two stages are one scanner, and a long newline run no longer overflows the stack.
     */
    @Test
    void lagunaSplitsALongNewlineRun() {
        Splitter laguna =
                HuggingFaceTokenizerLoader.parsePreTokenizer(
                        Map.of(
                                "type",
                                "Sequence",
                                "pretokenizers",
                                List.of(
                                        Map.of(
                                                "type",
                                                "Split",
                                                "pattern",
                                                Map.of("Regex", LAGUNA_NEWLINES),
                                                "behavior",
                                                "MergedWithNext",
                                                "invert",
                                                false),
                                        Map.of(
                                                "type",
                                                "Split",
                                                "pattern",
                                                Map.of(
                                                        "Regex",
                                                        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r"
                                                                + "\\n"
                                                                + "\\p{L}\\p{N}]?\\p{L}+|\\p{N}|"
                                                                + " ?[^\\s\\p{L}\\p{N}]+[\\r"
                                                                + "\\n"
                                                                + "]*|\\s*[\\r"
                                                                + "\\n"
                                                                + "]+|\\s+(?!\\S)|\\s+"),
                                                "behavior",
                                                "Isolated",
                                                "invert",
                                                false))));
        assertEquals("TiktokenSplitter", laguna.getClass().getSimpleName());
        String text = "x" + "\n".repeat(200_000) + "y";
        assertEquals(text, String.join("", pieces(laguna, text)));
    }

    /** Hugging Face's MergedWithNext: each match joins the text after it, up to the next match. */
    private static List<String> mergedWithNext(String text) {
        List<String> pieces = new ArrayList<>();
        Matcher m = Pattern.compile(LAGUNA_NEWLINES).matcher(text);
        int start = 0;
        while (m.find()) {
            if (m.start() > start) {
                pieces.add(text.substring(start, m.start()));
            }
            start = m.start();
        }
        if (text.length() > start) {
            pieces.add(text.substring(start));
        }
        return pieces;
    }

    private static Splitter split(String kind, String pattern, String behavior, boolean invert) {
        return HuggingFaceTokenizerLoader.parsePreTokenizer(
                Map.of(
                        "type",
                        "Split",
                        "pattern",
                        Map.of(kind, pattern),
                        "behavior",
                        behavior,
                        "invert",
                        invert));
    }

    private static List<String> pieces(Splitter split, String text) {
        List<String> ours = new ArrayList<>();
        split.splitAll(text, (t, s, e) -> ours.add(t.subSequence(s, e).toString()));
        ours.removeIf(String::isEmpty); // an empty piece encodes to nothing
        return ours;
    }
}
