package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.impl.FastCl100kSplitter;
import com.qxotic.toknroll.impl.FastO200kSplitter;
import com.qxotic.toknroll.impl.FastR50kSplitter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

class FastSplitterExhaustiveParityTest {

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

    @ParameterizedTest(name = "{0} exhaustive edge ranges")
    @MethodSource("splitterCases")
    void matchesRegexForAllRangesOnCuratedEdgeCases(SplitterCase splitterCase) {
        for (String sample : edgeSamples()) {
            assertAllRangesMatch(splitterCase, sample);
        }
    }

    @ParameterizedTest(name = "{0} deterministic random short ranges")
    @MethodSource("splitterCases")
    void matchesRegexForDeterministicRandomShortRanges(SplitterCase splitterCase) {
        Random rnd = new Random(20260415L);
        for (int i = 0; i < 600; i++) {
            String sample = randomShortMixedString(rnd, 24);
            assertAllRangesMatch(splitterCase, sample);
        }
    }

    private static void assertAllRangesMatch(SplitterCase splitterCase, String sample) {
        int len = sample.length();
        for (int start = 0; start <= len; start++) {
            for (int end = start; end <= len; end++) {
                assertEquals(
                        tokens(splitterCase.sourceOfTruth, sample, start, end),
                        tokens(splitterCase.fast, sample, start, end),
                        splitterCase.name + " range " + start + ".." + end + " :: " + sample);
            }
        }
    }

    private static List<String> tokens(Splitter splitter, String text, int start, int end) {
        List<String> out = new ArrayList<>();
        splitter.splitAll(
                text, start, end, (source, s, e) -> out.add(source.subSequence(s, e).toString()));
        return out;
    }

    private static List<String> edgeSamples() {
        return List.of(
                "",
                " ",
                "\r\n",
                "\n\n\n",
                "'s 't 're 've 'm 'll 'd",
                "'S 'T 'Re 'vE 'M 'Ll 'D",
                "a1234b 12 123 1234 12345",
                "AaAAa aAaA AAaa",
                "HTTPServer's READY and i'm done",
                " ?word",
                "!word",
                " symbols!!!\r\n",
                " /path/to/v1.2.3\r\n",
                "emoji 👩‍💻🚀 and accents café e\u0301",
                "Arabic مرحبا بالعالم",
                "CJK 你好 世界",
                "mixed: abc😀def\n12345\tEND",
                "zero-width a\u200Bb\u200Cc\u200Dd",
                "regional indicators 🇺🇸🇨🇦🇲",
                "line separators \u2028x\u2029y");
    }

    private static String randomShortMixedString(Random rnd, int maxLen) {
        int len = rnd.nextInt(maxLen + 1);
        StringBuilder sb = new StringBuilder(len);
        for (int i = 0; i < len; i++) {
            int bucket = rnd.nextInt(9);
            int cp;
            switch (bucket) {
                case 0:
                case 1:
                case 2:
                    cp = 32 + rnd.nextInt(95);
                    break;
                case 3:
                    cp = 0x00C0 + rnd.nextInt(0x017F - 0x00C0 + 1);
                    break;
                case 4:
                    cp = 0x0400 + rnd.nextInt(0x04FF - 0x0400 + 1);
                    break;
                case 5:
                    cp = 0x0600 + rnd.nextInt(0x06FF - 0x0600 + 1);
                    break;
                case 6:
                    cp = 0x4E00 + rnd.nextInt(0x9FFF - 0x4E00 + 1);
                    break;
                case 7:
                    cp = 0x1F300 + rnd.nextInt(0x1FAFF - 0x1F300 + 1);
                    break;
                default:
                    cp = rnd.nextBoolean() ? '\n' : '\r';
                    break;
            }
            sb.appendCodePoint(cp);
            if (rnd.nextInt(16) == 0) {
                sb.appendCodePoint(0x0300 + rnd.nextInt(0x036F - 0x0300 + 1));
            }
        }
        return sb.toString();
    }

    private static Stream<SplitterCase> splitterCases() {
        return Stream.of(
                new SplitterCase("r50k", FastR50kSplitter.INSTANCE, Splitter.regex(R50K_PATTERN)),
                new SplitterCase(
                        "cl100k",
                        FastCl100kSplitter.INSTANCE,
                        Splitter.regex(
                                Pattern.compile(CL100K_PATTERN, Pattern.UNICODE_CHARACTER_CLASS))),
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
