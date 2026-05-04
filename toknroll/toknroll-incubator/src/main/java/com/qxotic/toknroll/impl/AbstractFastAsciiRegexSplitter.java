package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.Splitter;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

abstract class AbstractFastAsciiRegexSplitter implements Splitter {

    private final Pattern fallbackPattern;
    private static final int NO_UNICODE = -1;
    private static final long NO_UNICODE_SCAN = 0xFFFF_FFFFL;
    private static final int FLAG_NON_ASCII = 1;
    private static final int FLAG_CRLF = 2;

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

        int flags = scanFlags(text, startInclusive, endExclusive);
        if ((flags & FLAG_NON_ASCII) == 0) {
            splitAscii(text, startInclusive, endExclusive, consumer);
            return;
        }
        if ((flags & FLAG_CRLF) != 0) {
            SplitterSupport.splitRegex(
                    fallbackPattern, text, startInclusive, endExclusive, consumer);
            return;
        }

        Matcher matcher = null;
        int cursor = startInclusive;
        while (cursor < endExclusive) {
            long scan = scanForUnicodeAndLineBoundary(text, cursor, endExclusive);
            int unicodeAt = unpackLow(scan);
            if (unicodeAt == NO_UNICODE) {
                splitAscii(text, cursor, endExclusive, consumer);
                return;
            }

            int islandStart = unpackHigh(scan);
            if (islandStart > cursor) {
                splitAscii(text, cursor, islandStart, consumer);
            }

            int islandEnd = nextLineBoundary(text, unicodeAt, endExclusive);
            if (matcher == null) {
                matcher = fallbackPattern.matcher(text);
            }
            splitRegex(text, islandStart, islandEnd, consumer, matcher);
            cursor = islandEnd;
        }
    }

    protected boolean includeSlashInBoundary() {
        return false;
    }

    protected boolean mergeNewlineRunsInBoundary() {
        return false;
    }

    private static long scanForUnicodeAndLineBoundary(
            CharSequence text, int startInclusive, int endExclusive) {
        int lineBoundary = startInclusive;
        for (int i = startInclusive; i < endExclusive; i++) {
            char ch = text.charAt(i);
            if (ch > 0x7F) {
                return (((long) lineBoundary) << 32) | (i & 0xFFFF_FFFFL);
            }
            if (ch == '\n' || ch == '\r') {
                lineBoundary = i + 1;
            }
        }
        return NO_UNICODE_SCAN;
    }

    private static int scanFlags(CharSequence text, int startInclusive, int endExclusive) {
        int flags = 0;
        for (int i = startInclusive; i < endExclusive; i++) {
            char ch = text.charAt(i);
            if (ch > 0x7F) {
                flags |= FLAG_NON_ASCII;
            }
            if (ch == '\n' || ch == '\r') {
                flags |= FLAG_CRLF;
            }
            if (flags == (FLAG_NON_ASCII | FLAG_CRLF)) {
                return flags;
            }
        }
        return flags;
    }

    private int nextLineBoundary(CharSequence text, int index, int endExclusive) {
        int i = index;
        while (i < endExclusive) {
            char ch = text.charAt(i);
            if (ch == '\n' || ch == '\r') {
                if (!mergeNewlineRunsInBoundary()) {
                    return i + 1;
                }
                int wsEnd = i;
                int lastNewlineEnd = i + 1;
                while (wsEnd < endExclusive) {
                    char n = text.charAt(wsEnd);
                    if (!SplitterSupport.isAsciiWhitespace(n)) {
                        break;
                    }
                    if (n == '\n' || n == '\r') {
                        lastNewlineEnd = wsEnd + 1;
                    }
                    wsEnd++;
                }

                int boundary = lastNewlineEnd;
                while (boundary < endExclusive) {
                    char n = text.charAt(boundary);
                    if (n == '\n' || n == '\r' || (includeSlashInBoundary() && n == '/')) {
                        boundary++;
                        continue;
                    }
                    break;
                }
                return boundary;
            }
            i++;
        }
        return endExclusive;
    }

    private static int unpackHigh(long packed) {
        return (int) (packed >>> 32);
    }

    private static int unpackLow(long packed) {
        return (int) packed;
    }

    private static void splitRegex(
            CharSequence text,
            int startInclusive,
            int endExclusive,
            Splitter.SplitConsumer consumer,
            Matcher matcher) {
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

    protected abstract void splitAscii(
            CharSequence text,
            int startInclusive,
            int endExclusive,
            Splitter.SplitConsumer consumer);
}
