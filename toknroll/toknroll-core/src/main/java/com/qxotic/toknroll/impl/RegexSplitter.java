package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.Splitter;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

final class RegexSplitter implements Splitter {

    private final Pattern pattern;
    private final boolean emitGaps;

    private RegexSplitter(Pattern pattern, boolean emitGaps) {
        this.pattern = pattern;
        this.emitGaps = emitGaps;
    }

    public static RegexSplitter create(Pattern pattern) {
        return create(pattern, true);
    }

    /** With {@code emitGaps} false, text between two matches is dropped instead of emitted. */
    public static RegexSplitter create(Pattern pattern, boolean emitGaps) {
        return new RegexSplitter(Objects.requireNonNull(pattern, "pattern"), emitGaps);
    }

    @Override
    public void splitAll(
            CharSequence text,
            int startInclusive,
            int endExclusive,
            Splitter.SplitConsumer consumer) {
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(consumer, "consumer");
        if (startInclusive < 0 || endExclusive < startInclusive || endExclusive > text.length()) {
            throw new IndexOutOfBoundsException(
                    "Invalid range ["
                            + startInclusive
                            + ", "
                            + endExclusive
                            + ") for text length "
                            + text.length());
        }
        Matcher matcher = pattern.matcher(text);
        matcher.region(startInclusive, endExclusive);
        int lastEnd = startInclusive;
        while (matcher.find()) {
            int matchStart = matcher.start();
            if (emitGaps && matchStart > lastEnd) {
                consumer.accept(text, lastEnd, matchStart);
            }
            int matchEnd = matcher.end();
            consumer.accept(text, matchStart, matchEnd);
            lastEnd = matchEnd;
        }
        if (emitGaps && lastEnd < endExclusive) {
            consumer.accept(text, lastEnd, endExclusive);
        }
    }
}
