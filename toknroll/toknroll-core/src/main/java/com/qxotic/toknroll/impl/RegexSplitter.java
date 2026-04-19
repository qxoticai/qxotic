package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.Splitter;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class RegexSplitter implements Splitter {

    private final Pattern pattern;

    private RegexSplitter(Pattern pattern) {
        this.pattern = pattern;
    }

    public static RegexSplitter create(Pattern pattern) {
        return new RegexSplitter(Objects.requireNonNull(pattern, "pattern"));
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
            if (matchStart > lastEnd) {
                consumer.accept(text, lastEnd, matchStart);
            }
            int matchEnd = matcher.end();
            consumer.accept(text, matchStart, matchEnd);
            lastEnd = matchEnd;
        }
        if (lastEnd < endExclusive) {
            consumer.accept(text, lastEnd, endExclusive);
        }
    }
}
