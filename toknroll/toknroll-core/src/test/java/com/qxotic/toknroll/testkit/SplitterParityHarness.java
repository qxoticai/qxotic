package com.qxotic.toknroll.testkit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;

import com.qxotic.toknroll.Splitter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/** Shared splitter parity harness against a source-of-truth splitter. */
public final class SplitterParityHarness {

    private static final List<String> REPRESENTATIVE_SAMPLES = TestCorpora.SPLITTER_PARITY_SAMPLES;

    private SplitterParityHarness() {}

    public static void runFullBattery(String name, Splitter splitter, Splitter sourceOfTruth) {
        compareRepresentative(name, splitter, sourceOfTruth);
        compareSubRanges(name, splitter, sourceOfTruth);
        compareRandomAscii(name, splitter, sourceOfTruth, 20260403L, 2_000);
        compareRandomUnicode(name, splitter, sourceOfTruth, 20260404L, 800);
    }

    private static void compareRepresentative(
            String name, Splitter splitter, Splitter sourceOfTruth) {
        for (String sample : REPRESENTATIVE_SAMPLES) {
            assertRangesMatch(name + " representative", splitter, sourceOfTruth, sample, 0, sample.length());
        }
    }

    private static void compareSubRanges(String name, Splitter splitter, Splitter sourceOfTruth) {
        for (String sample : REPRESENTATIVE_SAMPLES) {
            int len = sample.length();
            int[][] ranges = {
                {0, len},
                {0, Math.max(0, len / 2)},
                {Math.max(0, len / 3), len},
                {Math.max(0, len / 4), Math.max(0, (len * 3) / 4)}
            };
            for (int[] range : ranges) {
                int start = Math.min(range[0], len);
                int end = Math.min(Math.max(start, range[1]), len);
                assertRangesMatch(name + " range", splitter, sourceOfTruth, sample, start, end);
            }
        }
    }

    private static void compareRandomAscii(
            String name, Splitter splitter, Splitter sourceOfTruth, long seed, int iterations) {
        Random rnd = new Random(seed);
        for (int n = 0; n < iterations; n++) {
            int len = 1 + rnd.nextInt(128);
            StringBuilder sb = new StringBuilder(len);
            for (int i = 0; i < len; i++) {
                sb.append((char) (9 + rnd.nextInt(118)));
            }
            String sample = sb.toString();
            assertRangesMatch(name + " random-ascii", splitter, sourceOfTruth, sample, 0, sample.length());
        }
    }

    private static void compareRandomUnicode(
            String name, Splitter splitter, Splitter sourceOfTruth, long seed, int iterations) {
        Random rnd = new Random(seed);
        for (int n = 0; n < iterations; n++) {
            String sample = randomUnicodeString(rnd, 80);
            assertRangesMatch(name + " random-unicode", splitter, sourceOfTruth, sample, 0, sample.length());
        }
    }

    private static void assertRangesMatch(
            String context,
            Splitter splitter,
            Splitter sourceOfTruth,
            String text,
            int start,
            int end) {
        List<Range> expected = ranges(sourceOfTruth, text, start, end);
        List<Range> actual = ranges(splitter, text, start, end);
        if (!expected.equals(actual)) {
            int mismatch = firstMismatch(expected, actual);
            StringBuilder sb = new StringBuilder(256);
            sb.append(context)
                    .append(" mismatch at range ")
                    .append(start)
                    .append("..")
                    .append(end)
                    .append(" in text: ")
                    .append(escape(text));
            if (mismatch >= 0) {
                sb.append("\nexpected[")
                        .append(mismatch)
                        .append("]=")
                        .append(rangeDebug(expected, mismatch, text));
                sb.append("\nactual[")
                        .append(mismatch)
                        .append("]=")
                        .append(rangeDebug(actual, mismatch, text));
            }
            fail(sb.toString());
        }
        assertEquals(expected, actual);
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
        return r.start + ".." + r.end + " '" + escape(text.substring(r.start, r.end)) + "'";
    }

    private static String escape(String text) {
        return text.replace("\\", "\\\\")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }

    private static String randomUnicodeString(Random rnd, int maxCodePoints) {
        int count = 1 + rnd.nextInt(maxCodePoints);
        StringBuilder sb = new StringBuilder(count);
        for (int i = 0; i < count; i++) {
            int bucket = rnd.nextInt(10);
            int cp;
            if (bucket <= 3) {
                cp = 32 + rnd.nextInt(95);
            } else if (bucket == 4) {
                cp = 0x00C0 + rnd.nextInt(0x017F - 0x00C0 + 1);
            } else if (bucket == 5) {
                cp = 0x0370 + rnd.nextInt(0x03FF - 0x0370 + 1);
            } else if (bucket == 6) {
                cp = 0x0400 + rnd.nextInt(0x04FF - 0x0400 + 1);
            } else if (bucket == 7) {
                cp = 0x4E00 + rnd.nextInt(0x9FFF - 0x4E00 + 1);
            } else if (bucket == 8) {
                cp = 0x0600 + rnd.nextInt(0x06FF - 0x0600 + 1);
            } else {
                cp = 0x1F300 + rnd.nextInt(0x1FAFF - 0x1F300 + 1);
            }
            if (Character.isSurrogate((char) cp)) {
                cp = 'x';
            }
            sb.appendCodePoint(cp);
            if (rnd.nextInt(16) == 0) {
                sb.appendCodePoint(0x0300 + rnd.nextInt(0x036F - 0x0300 + 1));
            }
        }
        return sb.toString();
    }

    private static List<String> tokens(Splitter splitter, String text) {
        return tokens(splitter, text, 0, text.length());
    }

    private static List<String> tokens(Splitter splitter, String text, int start, int end) {
        List<String> out = new ArrayList<>();
        splitter.splitAll(
                text, start, end, (source, s, e) -> out.add(source.subSequence(s, e).toString()));
        return out;
    }

    private static List<Range> ranges(Splitter splitter, String text, int start, int end) {
        List<Range> out = new ArrayList<>();
        splitter.splitAll(text, start, end, (source, s, e) -> out.add(new Range(s, e)));
        return out;
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

        @Override
        public String toString() {
            return start + ".." + end;
        }
    }
}
