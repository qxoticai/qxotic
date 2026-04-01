package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.advanced.Splitter;
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
}
