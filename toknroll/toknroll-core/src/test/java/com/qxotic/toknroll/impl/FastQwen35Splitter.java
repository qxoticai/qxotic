package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.Splitter;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Fast ASCII-oriented splitter for the Qwen 3.5 pre-tokenizer regex. */
final class FastQwen35Splitter implements Splitter {

    private static final String QWEN35_PATTERN =
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r"
                    + "\\n"
                    + "]*|\\s*[\\r"
                    + "\\n"
                    + "]+|\\s+(?!\\S)|\\s+";

    private static final Pattern QWEN35_COMPILED =
            Pattern.compile(QWEN35_PATTERN, Pattern.UNICODE_CHARACTER_CLASS);
    public static final FastQwen35Splitter INSTANCE = new FastQwen35Splitter();

    private FastQwen35Splitter() {}

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

            // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
            if (SplitterSupport.isAsciiLetter(c)) {
                int end = consumeAsciiLetterRun(text, i + 1, endExclusive);
                consumer.accept(text, i, end);
                i = end;
                continue;
            }
            if (c != '\r'
                    && c != '\n'
                    && !SplitterSupport.isAsciiDigit(c)
                    && i + 1 < endExclusive
                    && SplitterSupport.isAsciiLetter(text.charAt(i + 1))) {
                int end = consumeAsciiLetterRun(text, i + 2, endExclusive);
                consumer.accept(text, i, end);
                i = end;
                continue;
            }

            // \p{N}
            if (SplitterSupport.isAsciiDigit(c)) {
                consumer.accept(text, i, i + 1);
                i++;
                continue;
            }

            //  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
            if (c == ' '
                    && i + 1 < endExclusive
                    && SplitterSupport.isAsciiSymbolNoSpace(text.charAt(i + 1))) {
                int end = consumeAsciiSymbolRunAndTrailingNewlines(text, i + 2, endExclusive);
                consumer.accept(text, i, end);
                i = end;
                continue;
            }
            if (SplitterSupport.isAsciiSymbolNoSpace(c)) {
                int end = consumeAsciiSymbolRunAndTrailingNewlines(text, i + 1, endExclusive);
                consumer.accept(text, i, end);
                i = end;
                continue;
            }

            if (matcher == null) {
                matcher = QWEN35_COMPILED.matcher(text);
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
        SplitterSupport.splitRegex(QWEN35_COMPILED, text, startInclusive, endExclusive, consumer);
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

    private static int consumeAsciiLetterRun(CharSequence text, int i, int endExclusive) {
        int end = i;
        while (end < endExclusive && SplitterSupport.isAsciiLetter(text.charAt(end))) {
            end++;
        }
        return end;
    }

    private static int consumeAsciiSymbolRunAndTrailingNewlines(
            CharSequence text, int i, int endExclusive) {
        int end = i;
        while (end < endExclusive && SplitterSupport.isAsciiSymbolNoSpace(text.charAt(end))) {
            end++;
        }
        while (end < endExclusive) {
            char ch = text.charAt(end);
            if (ch != '\r' && ch != '\n') {
                break;
            }
            end++;
        }
        return end;
    }
}
