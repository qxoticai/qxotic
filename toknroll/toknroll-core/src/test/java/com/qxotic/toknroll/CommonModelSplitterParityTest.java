package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertNotNull;

import com.qxotic.toknroll.gguf.ModelTextSplitters;
import com.qxotic.toknroll.loaders.ModelSplitters;
import com.qxotic.toknroll.testkit.SplitterParityHarness;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

class CommonModelSplitterParityTest {

    private static final String LLAMA3_PATTERN =
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r"
                    + "\\n"
                    + "]*|\\s*[\\r"
                    + "\\n"
                    + "]+|\\s+(?!\\S)|\\s+";

    private static final String QWEN35_PATTERN = ModelTextSplitters.QWEN35_PATTERN;

    private static final String[] SMOLLM3_PATTERNS = {
        "\\p{N}", "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"
    };

    private static final String[] TEKKEN_PATTERNS = {
        "[^\\r"
            + "\\n"
            + "\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r"
            + "\\n"
            + "\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}|"
            + " ?[^\\s\\p{L}\\p{N}]+[\\r"
            + "\\n"
            + "/]*|\\s*[\\r"
            + "\\n"
            + "]+|\\s+(?!\\S)|\\s+"
    };

    private static final String[] REFACT_PATTERNS = {
        "\\p{N}", "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"
    };

    @ParameterizedTest(name = "{0}")
    @MethodSource("cases")
    void commonModelSplittersMatchRegexSourceOfTruth(Case testCase) {
        Splitter splitter = ModelSplitters.forModel(testCase.model);
        assertNotNull(splitter, "Model splitter not found for: " + testCase.model);
        SplitterParityHarness.runFullBattery(testCase.model, splitter, testCase.sourceOfTruth);
    }

    private static Stream<Case> cases() {
        return Stream.of(
                new Case(
                        "llama3",
                        Splitter.regex(
                                Pattern.compile(LLAMA3_PATTERN, Pattern.UNICODE_CHARACTER_CLASS))),
                new Case(
                        "qwen3.5",
                        Splitter.regex(
                                Pattern.compile(QWEN35_PATTERN, Pattern.UNICODE_CHARACTER_CLASS))),
                new Case(
                        "deepseek-r1",
                        Splitter.regex(
                                Pattern.compile(QWEN35_PATTERN, Pattern.UNICODE_CHARACTER_CLASS))),
                new Case(
                        "kimi-2.5",
                        Splitter.regex(
                                Pattern.compile(QWEN35_PATTERN, Pattern.UNICODE_CHARACTER_CLASS))),
                new Case("gemma4", Splitter.identity()),
                new Case(
                        "mistral-tekken",
                        Splitter.sequence(
                                Stream.of(TEKKEN_PATTERNS)
                                        .map(
                                                p ->
                                                        Splitter.regex(
                                                                Pattern.compile(
                                                                        p,
                                                                        Pattern
                                                                                .UNICODE_CHARACTER_CLASS)))
                                        .toArray(Splitter[]::new))),
                new Case(
                        "granite4",
                        Splitter.sequence(
                                Stream.of(REFACT_PATTERNS)
                                        .map(
                                                p ->
                                                        Splitter.regex(
                                                                Pattern.compile(
                                                                        p,
                                                                        Pattern
                                                                                .UNICODE_CHARACTER_CLASS)))
                                        .toArray(Splitter[]::new))),
                new Case(
                        "smollm3",
                        Splitter.sequence(
                                Stream.of(SMOLLM3_PATTERNS)
                                        .map(
                                                p ->
                                                        Splitter.regex(
                                                                Pattern.compile(
                                                                        p,
                                                                        Pattern
                                                                                .UNICODE_CHARACTER_CLASS)))
                                        .toArray(Splitter[]::new))));
    }

    private static final class Case {
        final String model;
        final Splitter sourceOfTruth;

        Case(String model, Splitter sourceOfTruth) {
            this.model = model;
            this.sourceOfTruth = sourceOfTruth;
        }

        @Override
        public String toString() {
            return model;
        }
    }
}
