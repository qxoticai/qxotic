package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.Splitter;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

final class SplitterSupport {

    private static final byte[] ASCII_LETTER = new byte[128];
    private static final byte[] ASCII_DIGIT = new byte[128];
    private static final byte[] ASCII_WHITESPACE = new byte[128];
    private static final byte[] ASCII_SYMBOL = new byte[128];
    private static final byte[] ASCII_SYMBOL_NO_SPACE = new byte[128];

    static {
        for (int c = 'a'; c <= 'z'; c++) {
            ASCII_LETTER[c] = 1;
        }
        for (int c = 'A'; c <= 'Z'; c++) {
            ASCII_LETTER[c] = 1;
        }
        for (int c = '0'; c <= '9'; c++) {
            ASCII_DIGIT[c] = 1;
        }

        ASCII_WHITESPACE[' '] = 1;
        ASCII_WHITESPACE['\t'] = 1;
        ASCII_WHITESPACE['\n'] = 1;
        ASCII_WHITESPACE['\r'] = 1;
        ASCII_WHITESPACE['\f'] = 1;
        ASCII_WHITESPACE[0x0B] = 1;

        for (int c = 0; c < 128; c++) {
            if (ASCII_WHITESPACE[c] == 0 && ASCII_LETTER[c] == 0 && ASCII_DIGIT[c] == 0) {
                ASCII_SYMBOL[c] = 1;
            }
            if (c != ' '
                    && ASCII_WHITESPACE[c] == 0
                    && ASCII_LETTER[c] == 0
                    && ASCII_DIGIT[c] == 0) {
                ASCII_SYMBOL_NO_SPACE[c] = 1;
            }
        }
    }

    private SplitterSupport() {}

    static void validateRange(CharSequence text, int startInclusive, int endExclusive) {
        if (startInclusive < 0 || endExclusive < startInclusive || endExclusive > text.length()) {
            throw new IndexOutOfBoundsException(
                    "Invalid range ["
                            + startInclusive
                            + ", "
                            + endExclusive
                            + ") for text length "
                            + text.length());
        }
    }

    static boolean containsNonAscii(CharSequence text, int startInclusive, int endExclusive) {
        for (int i = startInclusive; i < endExclusive; i++) {
            if (text.charAt(i) > 0x7F) {
                return true;
            }
        }
        return false;
    }

    static void splitRegex(
            Pattern pattern,
            CharSequence text,
            int startInclusive,
            int endExclusive,
            Splitter.SplitConsumer consumer) {
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

    static char lowerAscii(char c) {
        return (c >= 'A' && c <= 'Z') ? (char) (c + ('a' - 'A')) : c;
    }

    static boolean isAsciiLetter(char c) {
        return c < 128 && ASCII_LETTER[c] != 0;
    }

    static boolean isAsciiDigit(char c) {
        return c < 128 && ASCII_DIGIT[c] != 0;
    }

    static boolean isAsciiWhitespace(char c) {
        return c < 128 && ASCII_WHITESPACE[c] != 0;
    }

    static boolean isAsciiSymbol(char c) {
        return c < 128 && ASCII_SYMBOL[c] != 0;
    }

    static boolean isAsciiSymbolNoSpace(char c) {
        return c < 128 && ASCII_SYMBOL_NO_SPACE[c] != 0;
    }

    static int scanAsciiLetters(CharSequence text, int startInclusive, int endExclusive) {
        int end = startInclusive;
        while (end < endExclusive && isAsciiLetter(text.charAt(end))) {
            end++;
        }
        return end;
    }

    static int scanAsciiDigits(CharSequence text, int startInclusive, int endExclusive) {
        int end = startInclusive;
        while (end < endExclusive && isAsciiDigit(text.charAt(end))) {
            end++;
        }
        return end;
    }

    static int scanAsciiSymbols(CharSequence text, int startInclusive, int endExclusive) {
        int end = startInclusive;
        while (end < endExclusive && isAsciiSymbol(text.charAt(end))) {
            end++;
        }
        return end;
    }

    static int scanAsciiSymbolsNoSpace(CharSequence text, int startInclusive, int endExclusive) {
        int end = startInclusive;
        while (end < endExclusive && isAsciiSymbolNoSpace(text.charAt(end))) {
            end++;
        }
        return end;
    }

    static int scanAsciiWhitespace(CharSequence text, int startInclusive, int endExclusive) {
        int end = startInclusive;
        while (end < endExclusive && isAsciiWhitespace(text.charAt(end))) {
            end++;
        }
        return end;
    }

    static int scanTrailingCrLf(CharSequence text, int startInclusive, int endExclusive) {
        if (text instanceof String) {
            return scanTrailingCrLfInString((String) text, startInclusive, endExclusive);
        }
        int end = startInclusive;
        while (end < endExclusive) {
            char ch = text.charAt(end);
            if (ch != '\r' && ch != '\n') {
                break;
            }
            end++;
        }
        return end;
    }

    static int scanTrailingCrLfSlash(CharSequence text, int startInclusive, int endExclusive) {
        if (text instanceof String) {
            return scanTrailingCrLfSlashInString((String) text, startInclusive, endExclusive);
        }
        int end = startInclusive;
        while (end < endExclusive) {
            char ch = text.charAt(end);
            if (ch != '\r' && ch != '\n' && ch != '/') {
                break;
            }
            end++;
        }
        return end;
    }

    private static int scanTrailingCrLfInString(String text, int startInclusive, int endExclusive) {
        int firstNewline = indexOfCrLf(text, startInclusive, endExclusive);
        if (firstNewline != startInclusive) {
            return startInclusive;
        }
        int end = startInclusive + 1;
        while (end < endExclusive) {
            char ch = text.charAt(end);
            if (ch != '\r' && ch != '\n') {
                break;
            }
            end++;
        }
        return end;
    }

    private static int scanTrailingCrLfSlashInString(
            String text, int startInclusive, int endExclusive) {
        if (startInclusive >= endExclusive) {
            return startInclusive;
        }
        char first = text.charAt(startInclusive);
        if (first != '\r' && first != '\n' && first != '/') {
            return startInclusive;
        }
        int end = startInclusive + 1;
        while (end < endExclusive) {
            char ch = text.charAt(end);
            if (ch != '\r' && ch != '\n' && ch != '/') {
                break;
            }
            end++;
        }
        return end;
    }

    private static int indexOfCrLf(String text, int startInclusive, int endExclusive) {
        if (startInclusive >= endExclusive) {
            return -1;
        }
        int nl = text.indexOf('\n', startInclusive);
        int cr = text.indexOf('\r', startInclusive);
        int best;
        if (nl < 0) {
            best = cr;
        } else if (cr < 0) {
            best = nl;
        } else {
            best = nl < cr ? nl : cr;
        }
        return (best >= 0 && best < endExclusive) ? best : -1;
    }
}
