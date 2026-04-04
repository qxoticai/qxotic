package com.qxotic.toknroll;

import com.qxotic.toknroll.advanced.Splitter;
import com.qxotic.toknroll.impl.FastCl100kSplitter;
import com.qxotic.toknroll.impl.FastLlama3Splitter;
import com.qxotic.toknroll.impl.FastO200kSplitter;
import com.qxotic.toknroll.impl.FastQwen35Splitter;
import com.qxotic.toknroll.impl.FastR50kSplitter;
import com.qxotic.toknroll.testkit.SplitterParityHarness;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

class SplitterRegexParityComprehensiveTest {

    private static final String R50K_PATTERN =
            "'(?:[sdmt]|ll|ve|re)| ?\\p{L}++| ?\\p{N}++|"
                    + " ?[^\\s\\p{L}\\p{N}]++|\\s++$|\\s+(?!\\S)|\\s";

    private static final String LLAMA3_PATTERN =
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r"
                    + "\\n"
                    + "]*|\\s*[\\r"
                    + "\\n"
                    + "]+|\\s+(?!\\S)|\\s+";

    private static final String CL100K_PATTERN =
            "'(?i:[sdmt]|ll|ve|re)|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?+\\p{L}++|\\p{N}{1,3}+| ?[^\\s\\p{L}\\p{N}]++[\\r"
                    + "\\n"
                    + "]*+|\\s++$|\\s*[\\r"
                    + "\\n"
                    + "]|\\s+(?!\\S)|\\s";

    private static final String QWEN35_PATTERN =
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r"
                    + "\\n"
                    + "]*|\\s*[\\r"
                    + "\\n"
                    + "]+|\\s+(?!\\S)|\\s+";

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

    @ParameterizedTest(name = "{0}")
    @MethodSource("splitterCases")
    void fastSplitterMatchesRegexBattery(SplitterCase splitterCase) {
        SplitterParityHarness.runFullBattery(
                splitterCase.name, splitterCase.fast, splitterCase.sourceOfTruth);
    }

    private static Stream<SplitterCase> splitterCases() {
        return Stream.of(
                new SplitterCase("r50k", FastR50kSplitter.INSTANCE, Splitter.regex(R50K_PATTERN)),
                new SplitterCase(
                        "llama3", FastLlama3Splitter.INSTANCE, Splitter.regex(LLAMA3_PATTERN)),
                new SplitterCase(
                        "cl100k",
                        FastCl100kSplitter.INSTANCE,
                        Splitter.regex(
                                Pattern.compile(CL100K_PATTERN, Pattern.UNICODE_CHARACTER_CLASS))),
                new SplitterCase(
                        "qwen35", FastQwen35Splitter.INSTANCE, Splitter.regex(QWEN35_PATTERN)),
                new SplitterCase(
                        "o200k", FastO200kSplitter.INSTANCE, Splitter.regex(O200K_PATTERN)));
    }

    private static final class SplitterCase {
        final String name;
        final Splitter fast;
        final Splitter sourceOfTruth;

        SplitterCase(String name, Splitter fast, Splitter sourceOfTruth) {
            this.name = name;
            this.fast = fast;
            this.sourceOfTruth = sourceOfTruth;
        }

        @Override
        public String toString() {
            return name;
        }
    }
}
