package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.Splitter;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Fast ASCII-oriented splitter approximating the LLAMA3 pre-tokenizer regex.
 *
 * <p>Falls back to regex for non-ASCII slices to preserve Unicode behavior.
 */
public final class FastLlama3Splitter implements Splitter {

    private static final String LLAMA3_PATTERN =
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r"
                    + "\\n"
                    + "]*|\\s*[\\r"
                    + "\\n"
                    + "]+|\\s+(?!\\S)|\\s+";

    private static final Pattern LLAMA3_COMPILED = Pattern.compile(LLAMA3_PATTERN);
    public static final FastLlama3Splitter INSTANCE = new FastLlama3Splitter();

    private FastLlama3Splitter() {}

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

            int apostropheChunk = apostropheChunkLength(text, i, endExclusive);
            if (apostropheChunk != 0) {
                consumer.accept(text, i, i + apostropheChunk);
                i += apostropheChunk;
                continue;
            }

            // [^\r\n\p{L}\p{N}]?\p{L}+
            if (SplitterSupport.isAsciiLetter(c)) {
                int end = i + 1;
                while (end < endExclusive && SplitterSupport.isAsciiLetter(text.charAt(end))) {
                    end++;
                }
                consumer.accept(text, i, end);
                i = end;
                continue;
            }
            if (c != '\r'
                    && c != '\n'
                    && !SplitterSupport.isAsciiDigit(c)
                    && i + 1 < endExclusive
                    && SplitterSupport.isAsciiLetter(text.charAt(i + 1))) {
                int end = i + 2;
                while (end < endExclusive && SplitterSupport.isAsciiLetter(text.charAt(end))) {
                    end++;
                }
                consumer.accept(text, i, end);
                i = end;
                continue;
            }

            // \p{N}{1,3}
            if (SplitterSupport.isAsciiDigit(c)) {
                int end = i + 1;
                if (end < endExclusive && SplitterSupport.isAsciiDigit(text.charAt(end))) {
                    end++;
                    if (end < endExclusive && SplitterSupport.isAsciiDigit(text.charAt(end))) {
                        end++;
                    }
                }
                consumer.accept(text, i, end);
                i = end;
                continue;
            }

            //  ?[^\s\p{L}\p{N}]+[\r\n]*
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
                    if (ch != '\r' && ch != '\n') {
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
                    if (ch != '\r' && ch != '\n') {
                        break;
                    }
                    end++;
                }
                consumer.accept(text, i, end);
                i = end;
                continue;
            }

            if (matcher == null) {
                matcher = LLAMA3_COMPILED.matcher(text);
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
        SplitterSupport.splitRegex(LLAMA3_COMPILED, text, startInclusive, endExclusive, consumer);
    }

    private static int apostropheChunkLength(CharSequence text, int i, int endExclusive) {
        if (text.charAt(i) != '\'') {
            return 0;
        }
        if (i + 1 >= endExclusive) {
            return 0;
        }
        char c1 = text.charAt(i + 1);
        if (c1 == 's' || c1 == 'S' || c1 == 't' || c1 == 'T' || c1 == 'm' || c1 == 'M' || c1 == 'd'
                || c1 == 'D') {
            return 2;
        }
        if (i + 2 >= endExclusive) {
            return 0;
        }
        char c2 = text.charAt(i + 2);
        if ((c1 == 'r' || c1 == 'R') && (c2 == 'e' || c2 == 'E')) {
            return 3;
        }
        if ((c1 == 'v' || c1 == 'V') && (c2 == 'e' || c2 == 'E')) {
            return 3;
        }
        if ((c1 == 'l' || c1 == 'L') && (c2 == 'l' || c2 == 'L')) {
            return 3;
        }
        return 0;
    }
}
