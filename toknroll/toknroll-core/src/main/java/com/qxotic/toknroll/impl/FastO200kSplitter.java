package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.advanced.Splitter;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Fast ASCII-oriented splitter approximating the o200k_base regex.
 *
 * <p>Falls back to regex for non-ASCII slices to preserve Unicode behavior.
 */
public final class FastO200kSplitter implements Splitter {

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

    private static final Pattern O200K_COMPILED = Pattern.compile(O200K_PATTERN);
    public static final FastO200kSplitter INSTANCE = new FastO200kSplitter();

    private FastO200kSplitter() {}

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

            if (matcher == null) {
                matcher = O200K_COMPILED.matcher(text);
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
        SplitterSupport.splitRegex(O200K_COMPILED, text, startInclusive, endExclusive, consumer);
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
