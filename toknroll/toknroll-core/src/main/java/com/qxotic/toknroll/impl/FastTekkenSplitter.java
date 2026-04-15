package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.Splitter;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Fast ASCII-oriented splitter approximating the Tekken pre-tokenizer regex.
 *
 * <p>Falls back to regex for non-ASCII slices to preserve Unicode behavior.
 */
public final class FastTekkenSplitter implements Splitter {

    private static final String TEKKEN_PATTERN =
            "[^\\r"
                + "\\n"
                + "\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r"
                + "\\n"
                + "\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}|"
                + " ?[^\\s\\p{L}\\p{N}]+[\\r"
                + "\\n"
                + "/]*|\\s*[\\r"
                + "\\n"
                + "]+|\\s+(?!\\S)|\\s+";

    private static final Pattern TEKKEN_COMPILED = Pattern.compile(TEKKEN_PATTERN);
    public static final FastTekkenSplitter INSTANCE = new FastTekkenSplitter();

    private FastTekkenSplitter() {}

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
        Matcher matcher = null;
        while (i < endExclusive) {
            char c = text.charAt(i);

            // [^\r\n\p{L}\p{N}]? <word>
            if (c != '\r'
                    && c != '\n'
                    && !SplitterSupport.isAsciiLetter(c)
                    && !SplitterSupport.isAsciiDigit(c)
                    && i + 1 < endExclusive
                    && SplitterSupport.isAsciiLetter(text.charAt(i + 1))) {
                int wordEnd = consumeTekkenWord(text, i + 1, endExclusive);
                if (wordEnd > i + 1) {
                    consumer.accept(text, i, wordEnd);
                    i = wordEnd;
                    continue;
                }
            }

            if (SplitterSupport.isAsciiLetter(c)) {
                int wordEnd = consumeTekkenWord(text, i, endExclusive);
                if (wordEnd > i) {
                    consumer.accept(text, i, wordEnd);
                    i = wordEnd;
                    continue;
                }
            }

            // \p{N}
            if (SplitterSupport.isAsciiDigit(c)) {
                consumer.accept(text, i, i + 1);
                i++;
                continue;
            }

            //  ?[^\s\p{L}\p{N}]+[\r\n/]*
            if (c == ' '
                    && i + 1 < endExclusive
                    && SplitterSupport.isAsciiSymbolNoSpace(text.charAt(i + 1))) {
                int end = i + 2;
                while (end < endExclusive
                        && SplitterSupport.isAsciiSymbolNoSpace(text.charAt(end))) {
                    end++;
                }
                while (end < endExclusive) {
                    char ch = text.charAt(end);
                    if (ch != '\r' && ch != '\n' && ch != '/') {
                        break;
                    }
                    end++;
                }
                consumer.accept(text, i, end);
                i = end;
                continue;
            }
            if (SplitterSupport.isAsciiSymbolNoSpace(c)) {
                int end = i + 1;
                while (end < endExclusive
                        && SplitterSupport.isAsciiSymbolNoSpace(text.charAt(end))) {
                    end++;
                }
                while (end < endExclusive) {
                    char ch = text.charAt(end);
                    if (ch != '\r' && ch != '\n' && ch != '/') {
                        break;
                    }
                    end++;
                }
                consumer.accept(text, i, end);
                i = end;
                continue;
            }

            // \s*[\r\n]+ | \s+(?!\S) | \s+
            if (SplitterSupport.isAsciiWhitespace(c)) {
                if (matcher == null) {
                    matcher = TEKKEN_COMPILED.matcher(text);
                }
                matcher.region(i, endExclusive);
                if (matcher.find()) {
                    int matchStart = matcher.start();
                    int matchEnd = matcher.end();
                    if (matchStart > i) {
                        consumer.accept(text, i, matchStart);
                        i = matchStart;
                    } else {
                        consumer.accept(text, i, matchEnd);
                        i = matchEnd;
                    }
                    continue;
                }
                consumer.accept(text, i, i + 1);
                i++;
                continue;
            }

            if (matcher == null) {
                matcher = TEKKEN_COMPILED.matcher(text);
            }
            matcher.region(i, endExclusive);
            if (matcher.lookingAt()) {
                int end = matcher.end();
                consumer.accept(text, i, end);
                i = end;
            } else {
                consumer.accept(text, i, i + 1);
                i++;
            }
        }
    }

    private void splitRegex(
            CharSequence text,
            int startInclusive,
            int endExclusive,
            Splitter.SplitConsumer consumer) {
        SplitterSupport.splitRegex(TEKKEN_COMPILED, text, startInclusive, endExclusive, consumer);
    }

    private static int consumeTekkenWord(CharSequence text, int start, int endExclusive) {
        int i = start;
        while (i < endExclusive && isAsciiUpper(text.charAt(i))) {
            i++;
        }
        int lowerStart = i;
        while (i < endExclusive && isAsciiLower(text.charAt(i))) {
            i++;
        }

        // First branch: [Upper]*[Lower]+
        if (i > lowerStart) {
            return i;
        }

        // Second branch: [Upper]+[Lower]* (with zero lowercase)
        if (lowerStart > start) {
            return lowerStart;
        }

        return 0;
    }

    private static boolean isAsciiUpper(char c) {
        return c >= 'A' && c <= 'Z';
    }

    private static boolean isAsciiLower(char c) {
        return c >= 'a' && c <= 'z';
    }
}
