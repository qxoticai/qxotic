package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.Splitter;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Strictly compliant splitter for cl100k_base.
 *
 * <p>Uses a fast ASCII path for dominant token classes and regex matching fallback where needed.
 */
final class FastCl100kSplitter extends AbstractFastAsciiRegexSplitter {

    private static final String CL100K_PATTERN =
            "'(?i:[sdmt]|ll|ve|re)|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?+\\p{L}++|\\p{N}{1,3}+| ?[^\\s\\p{L}\\p{N}]++[\\r"
                    + "\\n"
                    + "]*+|\\s++$|\\s*[\\r"
                    + "\\n"
                    + "]|\\s+(?!\\S)|\\s";

    private static final Pattern CL100K_COMPILED =
            Pattern.compile(CL100K_PATTERN, Pattern.UNICODE_CHARACTER_CLASS);

    public static final FastCl100kSplitter INSTANCE = new FastCl100kSplitter();

    private FastCl100kSplitter() {
        super(CL100K_COMPILED);
    }

    @Override
    protected void splitAscii(
            CharSequence text,
            int startInclusive,
            int endExclusive,
            Splitter.SplitConsumer consumer) {
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

            if (SplitterSupport.isAsciiLetter(c)) {
                int end = SplitterSupport.scanAsciiLetters(text, i + 1, endExclusive);
                consumer.accept(text, i, end);
                i = end;
                continue;
            }

            if (c != '\r'
                    && c != '\n'
                    && !SplitterSupport.isAsciiDigit(c)
                    && i + 1 < endExclusive
                    && SplitterSupport.isAsciiLetter(text.charAt(i + 1))) {
                int end = SplitterSupport.scanAsciiLetters(text, i + 2, endExclusive);
                consumer.accept(text, i, end);
                i = end;
                continue;
            }

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

            if (c == ' '
                    && i + 1 < endExclusive
                    && SplitterSupport.isAsciiSymbolNoSpace(text.charAt(i + 1))) {
                int end = SplitterSupport.scanAsciiSymbolsNoSpace(text, i + 2, endExclusive);
                end = SplitterSupport.scanTrailingCrLf(text, end, endExclusive);
                consumer.accept(text, i, end);
                i = end;
                continue;
            }

            if (SplitterSupport.isAsciiSymbolNoSpace(c)) {
                int end = SplitterSupport.scanAsciiSymbolsNoSpace(text, i + 1, endExclusive);
                end = SplitterSupport.scanTrailingCrLf(text, end, endExclusive);
                consumer.accept(text, i, end);
                i = end;
                continue;
            }

            if (matcher == null) {
                matcher = CL100K_COMPILED.matcher(text);
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

    private static int apostropheChunkLength(CharSequence text, int i, int endExclusive) {
        if (text.charAt(i) != '\'' || i + 1 >= endExclusive) {
            return 0;
        }
        char c1 = SplitterSupport.lowerAscii(text.charAt(i + 1));
        if (c1 == 's' || c1 == 'd' || c1 == 'm' || c1 == 't') {
            return 2;
        }
        if (i + 2 >= endExclusive) {
            return 0;
        }
        char c2 = SplitterSupport.lowerAscii(text.charAt(i + 2));
        if ((c1 == 'l' && c2 == 'l') || (c1 == 'v' && c2 == 'e') || (c1 == 'r' && c2 == 'e')) {
            return 3;
        }
        return 0;
    }
}
