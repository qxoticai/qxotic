package com.qxotic.toknroll.testkit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.advanced.Splitter;
import java.util.ArrayList;
import java.util.List;

/** Contract assertions for splitter range emissions. */
public final class SplitterContractHarness {

    private SplitterContractHarness() {}

    public static void assertConformsOnRange(
            String name,
            Splitter splitter,
            CharSequence text,
            int startInclusive,
            int endExclusive) {
        List<Range> ranges = new ArrayList<>();
        splitter.splitAll(
                text,
                startInclusive,
                endExclusive,
                (source, start, end) -> {
                    assertSame(text, source, name + " must emit original source");
                    ranges.add(new Range(start, end));
                });

        if (startInclusive == endExclusive) {
            if (ranges.isEmpty()) {
                return;
            }
            throw new AssertionError(name + " must emit nothing for empty ranges");
        }

        assertTrue(!ranges.isEmpty(), name + " must emit at least one range");
        int cursor = startInclusive;
        for (Range r : ranges) {
            assertTrue(r.startInclusive >= startInclusive, name + " range starts before parent");
            assertTrue(r.endExclusive <= endExclusive, name + " range ends after parent");
            assertTrue(r.startInclusive < r.endExclusive, name + " emitted empty range");
            assertEquals(
                    cursor, r.startInclusive, name + " emitted hole/overlap/out-of-order range");
            cursor = r.endExclusive;
        }
        assertEquals(endExclusive, cursor, name + " did not fully cover parent range");
    }

    public static void assertConformsOnText(String name, Splitter splitter, CharSequence text) {
        int len = text.length();
        assertConformsOnRange(name, splitter, text, 0, len);
        assertConformsOnRange(name, splitter, text, 0, 0);
        assertConformsOnRange(name, splitter, text, len, len);
        if (len >= 2) {
            assertConformsOnRange(name, splitter, text, 0, len / 2);
            assertConformsOnRange(name, splitter, text, len / 3, len);
            assertConformsOnRange(name, splitter, text, len / 4, (len * 3) / 4);
        }
    }

    private static final class Range {
        final int startInclusive;
        final int endExclusive;

        Range(int startInclusive, int endExclusive) {
            this.startInclusive = startInclusive;
            this.endExclusive = endExclusive;
        }
    }
}
