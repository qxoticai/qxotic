package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.Splitter;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Fast ASCII-oriented splitter for o200k regex semantics. */
final class FastO200kSplitter implements Splitter {

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

    private static final Pattern O200K_COMPILED =
            Pattern.compile(O200K_PATTERN, Pattern.UNICODE_CHARACTER_CLASS);
    private static final int NO_UNICODE = -1;
    private static final long NO_UNICODE_SCAN = 0xFFFF_FFFFL;

    public static final FastO200kSplitter INSTANCE = new FastO200kSplitter();

    private FastO200kSplitter() {}

    @Override
    public void splitAll(
            CharSequence text,
            int startInclusive,
            int endExclusive,
            Splitter.SplitConsumer consumer) {
        SplitterSupport.validateRange(text, startInclusive, endExclusive);
        if (startInclusive == endExclusive) {
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
                matcher = O200K_COMPILED.matcher(text);
            }
            splitRegex(text, islandStart, islandEnd, consumer, matcher);
            cursor = islandEnd;
        }
    }

    private void splitAscii(
            CharSequence text,
            int startInclusive,
            int endExclusive,
            Splitter.SplitConsumer consumer) {
        int i = startInclusive;
        while (i < endExclusive) {
            char c = text.charAt(i);

            if (SplitterSupport.isAsciiDigit(c)) {
                int end = i + 1;
                int maxEnd = Math.min(i + 3, endExclusive);
                while (end < maxEnd && SplitterSupport.isAsciiDigit(text.charAt(end))) {
                    end++;
                }
                consumer.accept(text, i, end);
                i = end;
                continue;
            }

            if (SplitterSupport.isAsciiLetter(c)) {
                int end = scanAsciiWordWithSuffix(text, i, endExclusive);
                consumer.accept(text, i, end);
                i = end;
                continue;
            }

            if (c != '\r'
                    && c != '\n'
                    && !SplitterSupport.isAsciiDigit(c)
                    && i + 1 < endExclusive
                    && SplitterSupport.isAsciiLetter(text.charAt(i + 1))) {
                int end = scanAsciiWordWithSuffix(text, i + 1, endExclusive);
                consumer.accept(text, i, end);
                i = end;
                continue;
            }

            if (c == ' '
                    && i + 1 < endExclusive
                    && SplitterSupport.isAsciiSymbolNoSpace(text.charAt(i + 1))) {
                int end = SplitterSupport.scanAsciiSymbolsNoSpace(text, i + 2, endExclusive);
                end = SplitterSupport.scanTrailingCrLfSlash(text, end, endExclusive);
                consumer.accept(text, i, end);
                i = end;
                continue;
            }

            if (SplitterSupport.isAsciiSymbolNoSpace(c)) {
                int end = SplitterSupport.scanAsciiSymbolsNoSpace(text, i + 1, endExclusive);
                end = SplitterSupport.scanTrailingCrLfSlash(text, end, endExclusive);
                consumer.accept(text, i, end);
                i = end;
                continue;
            }

            if (SplitterSupport.isAsciiWhitespace(c)) {
                int end = scanAsciiWhitespaceToken(text, i, endExclusive);
                consumer.accept(text, i, end);
                i = end;
                continue;
            }

            consumer.accept(text, i, i + 1);
            i++;
        }
    }

    private static int scanAsciiWhitespaceToken(
            CharSequence text, int startInclusive, int endExclusive) {
        int runEnd = SplitterSupport.scanAsciiWhitespace(text, startInclusive, endExclusive);
        int lastNewline = -1;
        for (int i = startInclusive; i < runEnd; i++) {
            char ch = text.charAt(i);
            if (ch == '\n' || ch == '\r') {
                lastNewline = i;
            }
        }
        if (lastNewline >= 0) {
            return lastNewline + 1;
        }
        if (runEnd == endExclusive) {
            return runEnd;
        }
        return (runEnd - startInclusive) > 1 ? (runEnd - 1) : runEnd;
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

    private static int unpackHigh(long packed) {
        return (int) (packed >>> 32);
    }

    private static int unpackLow(long packed) {
        return (int) packed;
    }

    private static int nextLineBoundary(CharSequence text, int index, int endExclusive) {
        int i = index;
        while (i < endExclusive) {
            char ch = text.charAt(i);
            if (ch == '\n' || ch == '\r') {
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
                    if (n != '/' && n != '\n' && n != '\r') {
                        break;
                    }
                    boundary++;
                }
                return boundary;
            }
            i++;
        }
        return endExclusive;
    }

    private static int scanAsciiWordWithSuffix(CharSequence text, int start, int endExclusive) {
        int end = start;
        boolean seenLower = false;
        while (end < endExclusive) {
            char c = text.charAt(end);
            if (!SplitterSupport.isAsciiLetter(c)) {
                break;
            }
            boolean isLower = c >= 'a' && c <= 'z';
            if (!isLower && seenLower) {
                break;
            }
            seenLower |= isLower;
            end++;
        }
        int suffix = apostropheSuffixLength(text, end, endExclusive);
        return end + suffix;
    }

    private static int apostropheSuffixLength(CharSequence text, int i, int endExclusive) {
        if (i + 1 >= endExclusive || text.charAt(i) != '\'') {
            return 0;
        }
        char c1 = SplitterSupport.lowerAscii(text.charAt(i + 1));
        if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
            return 2;
        }
        if (i + 2 >= endExclusive) {
            return 0;
        }
        char c2 = SplitterSupport.lowerAscii(text.charAt(i + 2));
        if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
            return 3;
        }
        return 0;
    }
}
