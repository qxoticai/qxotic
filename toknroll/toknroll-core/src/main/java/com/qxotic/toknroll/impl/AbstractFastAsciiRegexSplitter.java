package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.advanced.Splitter;
import java.util.Objects;
import java.util.regex.Pattern;

abstract class AbstractFastAsciiRegexSplitter implements Splitter {

    private final Pattern fallbackPattern;

    protected AbstractFastAsciiRegexSplitter(Pattern fallbackPattern) {
        this.fallbackPattern = Objects.requireNonNull(fallbackPattern, "fallbackPattern");
    }

    @Override
    public final void splitAll(
            CharSequence text,
            int startInclusive,
            int endExclusive,
            Splitter.SplitConsumer consumer) {
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(consumer, "consumer");
        SplitterSupport.validateRange(text, startInclusive, endExclusive);
        if (startInclusive == endExclusive) {
            return;
        }

        if (SplitterSupport.containsNonAscii(text, startInclusive, endExclusive)) {
            SplitterSupport.splitRegex(
                    fallbackPattern, text, startInclusive, endExclusive, consumer);
            return;
        }

        splitAscii(text, startInclusive, endExclusive, consumer);
    }

    protected abstract void splitAscii(
            CharSequence text,
            int startInclusive,
            int endExclusive,
            Splitter.SplitConsumer consumer);
}
