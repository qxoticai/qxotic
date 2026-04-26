package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;

import com.qxotic.toknroll.testkit.corpus.Enwik8Corpus;
import com.qxotic.toknroll.impl.FastSplitters;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("slow")
class O200kCorpusSpanParityTest {

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

    @Test
    void matchesRegexOnRandomEnwik8Slices() {
        byte[] bytes = Enwik8Corpus.load().data();
        String text = new String(bytes, StandardCharsets.UTF_8);

        Splitter fast = FastSplitters.o200k();
        Splitter regex =
                Splitter.regex(Pattern.compile(O200K_PATTERN, Pattern.UNICODE_CHARACTER_CLASS));

        Random rnd = new Random(20260421L);
        for (int i = 0; i < 500; i++) {
            int start = rnd.nextInt(text.length());
            int maxLen = Math.min(16_384, text.length() - start);
            int len = 1 + rnd.nextInt(maxLen);
            int end = start + len;

            // stay aligned to char boundaries for valid subrange checks around potential surrogates
            while (start > 0 && Character.isLowSurrogate(text.charAt(start))) {
                start--;
            }
            while (end < text.length() && Character.isLowSurrogate(text.charAt(end))) {
                end++;
            }

            List<Range> expected = ranges(regex, text, start, end);
            List<Range> actual = ranges(fast, text, start, end);
            if (!expected.equals(actual)) {
                int mismatch = firstMismatch(expected, actual);
                String expectedDbg = rangeDebug(expected, mismatch, text);
                String actualDbg = rangeDebug(actual, mismatch, text);
                fail(
                        "slice "
                                + i
                                + " range "
                                + start
                                + ".."
                                + end
                                + " mismatch index "
                                + mismatch
                                + "\nexpected="
                                + expectedDbg
                                + "\nactual="
                                + actualDbg);
            }
            assertEquals(expected, actual);
        }
    }

    private static List<Range> ranges(Splitter splitter, String text, int start, int end) {
        List<Range> out = new ArrayList<Range>();
        splitter.splitAll(text, start, end, (source, s, e) -> out.add(new Range(s, e)));
        return out;
    }

    private static int firstMismatch(List<Range> expected, List<Range> actual) {
        int common = Math.min(expected.size(), actual.size());
        for (int i = 0; i < common; i++) {
            if (!expected.get(i).equals(actual.get(i))) {
                return i;
            }
        }
        return expected.size() == actual.size() ? -1 : common;
    }

    private static String rangeDebug(List<Range> ranges, int index, String text) {
        if (index < 0 || index >= ranges.size()) {
            return "<missing>";
        }
        Range r = ranges.get(index);
        String token = text.substring(r.start, r.end).replace("\n", "\\n").replace("\r", "\\r");
        return r.start + ".." + r.end + " '" + token + "'";
    }

    private static final class Range {
        final int start;
        final int end;

        Range(int start, int end) {
            this.start = start;
            this.end = end;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) {
                return true;
            }
            if (!(obj instanceof Range)) {
                return false;
            }
            Range other = (Range) obj;
            return start == other.start && end == other.end;
        }

        @Override
        public int hashCode() {
            return 31 * start + end;
        }
    }
}
