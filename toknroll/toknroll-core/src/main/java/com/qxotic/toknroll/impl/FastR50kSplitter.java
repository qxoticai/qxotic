package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.advanced.Splitter;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * Fast ASCII-oriented splitter approximating the canonical r50k GPT-2 regex.
 *
 * <p>Falls back to regex for non-ASCII slices to preserve Unicode behavior.
 */
public final class FastR50kSplitter implements Splitter {

    private static final String R50K_PATTERN =
            "'(?:[sdmt]|ll|ve|re)| ?\\p{L}++| ?\\p{N}++|"
                    + " ?[^\\s\\p{L}\\p{N}]++|\\s++$|\\s+(?!\\S)|\\s";

    public static final FastR50kSplitter INSTANCE = new FastR50kSplitter();

    private static final Pattern R50K_COMPILED = Pattern.compile(R50K_PATTERN);

    private FastR50kSplitter() {}

    @Override
    public void splitAll(
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
            splitRegex(text, startInclusive, endExclusive, consumer);
            return;
        }

        int i = startInclusive;
        while (i < endExclusive) {
            char c = text.charAt(i);

            int apostropheChunk = apostropheChunkLength(text, i, endExclusive);
            if (apostropheChunk != 0) {
                consumer.accept(text, i, i + apostropheChunk);
                i += apostropheChunk;
                continue;
            }

            if (c == ' ' && i + 1 < endExclusive) {
                char n = text.charAt(i + 1);
                if (SplitterSupport.isAsciiLetter(n)) {
                    int end = i + 2;
                    while (end < endExclusive && SplitterSupport.isAsciiLetter(text.charAt(end))) {
                        end++;
                    }
                    consumer.accept(text, i, end);
                    i = end;
                    continue;
                }
                if (SplitterSupport.isAsciiDigit(n)) {
                    int end = i + 2;
                    while (end < endExclusive && SplitterSupport.isAsciiDigit(text.charAt(end))) {
                        end++;
                    }
                    consumer.accept(text, i, end);
                    i = end;
                    continue;
                }
                if (SplitterSupport.isAsciiSymbol(n)) {
                    int end = i + 2;
                    while (end < endExclusive && SplitterSupport.isAsciiSymbol(text.charAt(end))) {
                        end++;
                    }
                    consumer.accept(text, i, end);
                    i = end;
                    continue;
                }
            }

            if (SplitterSupport.isAsciiLetter(c)) {
                int end = i + 1;
                while (end < endExclusive && SplitterSupport.isAsciiLetter(text.charAt(end))) {
                    end++;
                }
                consumer.accept(text, i, end);
                i = end;
                continue;
            }

            if (SplitterSupport.isAsciiDigit(c)) {
                int end = i + 1;
                while (end < endExclusive && SplitterSupport.isAsciiDigit(text.charAt(end))) {
                    end++;
                }
                consumer.accept(text, i, end);
                i = end;
                continue;
            }

            if (SplitterSupport.isAsciiSymbol(c)) {
                int end = i + 1;
                while (end < endExclusive && SplitterSupport.isAsciiSymbol(text.charAt(end))) {
                    end++;
                }
                consumer.accept(text, i, end);
                i = end;
                continue;
            }

            // Whitespace handling mirrors: \s++$ | \s+(?!\S) | \s
            if (SplitterSupport.isAsciiWhitespace(c)) {
                int end = i + 1;
                while (end < endExclusive && SplitterSupport.isAsciiWhitespace(text.charAt(end))) {
                    end++;
                }
                if (end == endExclusive) {
                    consumer.accept(text, i, end);
                    i = end;
                    continue;
                }
                if (end - i >= 2) {
                    consumer.accept(text, i, end - 1);
                    i = end - 1;
                    continue;
                }
            }

            consumer.accept(text, i, i + 1);
            i++;
        }
    }

    private void splitRegex(
            CharSequence text,
            int startInclusive,
            int endExclusive,
            Splitter.SplitConsumer consumer) {
        SplitterSupport.splitRegex(R50K_COMPILED, text, startInclusive, endExclusive, consumer);
    }

    private static int apostropheChunkLength(CharSequence text, int i, int endExclusive) {
        if (text.charAt(i) != '\'') {
            return 0;
        }
        if (i + 1 >= endExclusive) {
            return 0;
        }
        char c1 = text.charAt(i + 1);
        if (c1 == 's' || c1 == 'd' || c1 == 'm' || c1 == 't') {
            return 2;
        }
        if (i + 2 >= endExclusive) {
            return 0;
        }
        char c2 = text.charAt(i + 2);
        if ((c1 == 'l' && c2 == 'l') || (c1 == 'v' && c2 == 'e') || (c1 == 'r' && c2 == 'e')) {
            return 3;
        }
        return 0;
    }
}
