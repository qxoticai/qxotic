package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.Splitter;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * The pre-tokenization regexes of tiktoken (r50k, cl100k, o200k) and their descendants, scanned by
 * hand.
 *
 * <p>The family differs in a few places only: how a contraction is spelled, what a word is, how
 * many digits go together, what may trail punctuation, and whether a newline run is a piece of its
 * own. One scanner with those knobs covers it. {@link #forPattern} hands the scanner out for the
 * regex strings it knows, so every loader keeps compiling its regex and gets this instead.
 *
 * <p>The regex stays the reference. Character classes come from the same definitions the regex
 * engine uses ({@link CharClasses}), and the parity tests hold every known pattern string against
 * its own regex, corner cases included: the backtracking in o200k's word rule, the Unicode case
 * folding of a contraction, a slash after a newline.
 */
final class TiktokenSplitter implements Splitter {

    // Contraction spellings: 's|'t|'re|'ve|'m|'ll|'d as written, as [sS] classes, or under (?i),
    // where Unicode case folding also admits U+017F (long s) for the s, class or not.
    private static final int NO_CONTRACTION = 0;
    private static final int LOWER = 1;
    private static final int ASCII_BOTH = 2;
    private static final int FOLD = 3;

    // Word rules.
    private static final int SPACE_LETTERS = 0; // ' ?\p{L}+' (r50k)
    private static final int PREFIX_LETTERS = 1; // '[^\r\n\p{L}\p{N}]?\p{L}+' (cl100k)
    private static final int PREFIX_LETTERS_MARKS =
            2; // '[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+' (qwen3.5)
    private static final int CASED = 3; // o200k's upper*lower+ | upper+lower* pair

    // Number rules.
    private static final int SPACE_RUN = 0; // ' ?\p{N}+'
    private static final int UP_TO_THREE = 1; // '\p{N}{1,3}'
    private static final int ONE = 2; // '\p{N}'

    // What may trail ' ?[^\s\p{L}\p{N}]+'.
    private static final int NO_TAIL = 0;
    private static final int NEWLINES = 1; // '[\r\n]*'
    private static final int NEWLINES_SLASHES = 2; // '[\r\n/]*'

    private final int contraction;
    private final int word;
    private final boolean wordSuffix; // o200k: '(?i:'s|...)?' after the word
    private final int numbers;
    private final boolean otherExcludesMarks; // qwen3.5: '[^\s\p{L}\p{M}\p{N}]'
    private final int otherTail;
    private final boolean newlineRun; // '\s*[\r\n]+' before the whitespace rules

    private TiktokenSplitter(
            int contraction,
            int word,
            boolean wordSuffix,
            int numbers,
            boolean otherExcludesMarks,
            int otherTail,
            boolean newlineRun) {
        this.contraction = contraction;
        this.word = word;
        this.wordSuffix = wordSuffix;
        this.numbers = numbers;
        this.otherExcludesMarks = otherExcludesMarks;
        this.otherTail = otherTail;
        this.newlineRun = newlineRun;
    }

    /** The regex strings the loaders compile, each with the scanner that splits like it. */
    private static final Map<String, TiktokenSplitter> KNOWN = new HashMap<>();

    static {
        String contraction = "'s|'t|'re|'ve|'m|'ll|'d";
        String asciiBoth = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])";
        String prefix = "[^\\r\\n\\p{L}\\p{N}]?";
        String other = " ?[^\\s\\p{L}\\p{N}]+";
        String upper = "[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]";
        String lower = "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]";
        String cased = prefix + upper + "*" + lower + "+%s|" + prefix + upper + "+" + lower + "*%s";
        String tail = "|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

        // r50k: tiktoken and the ByteLevel pre-tokenizer, then the tiktoken fixture's spelling
        known(
                new TiktokenSplitter(LOWER, SPACE_LETTERS, false, SPACE_RUN, false, NO_TAIL, false),
                contraction + "| ?\\p{L}+| ?\\p{N}+|" + other + "|\\s+(?!\\S)|\\s+",
                "'(?:[sdmt]|ll|ve|re)| ?\\p{L}++| ?\\p{N}++|"
                        + " ?[^\\s\\p{L}\\p{N}]++|\\s++$|\\s+(?!\\S)|\\s");
        // cl100k: tokenizer.json (Llama 3, SmolLM3, Granite, phi-4), then llama.cpp's spelling
        known(
                new TiktokenSplitter(
                        FOLD, PREFIX_LETTERS, false, UP_TO_THREE, false, NEWLINES, true),
                "(?i:"
                        + contraction
                        + ")|"
                        + prefix
                        + "\\p{L}+|\\p{N}{1,3}|"
                        + other
                        + "[\\r\\n]*"
                        + tail);
        known(
                new TiktokenSplitter(
                        ASCII_BOTH, PREFIX_LETTERS, false, UP_TO_THREE, false, NEWLINES, true),
                asciiBoth + "|" + prefix + "\\p{L}+|\\p{N}{1,3}|" + other + "[\\r\\n]*" + tail);
        // o200k: tokenizer.json (gpt-4o, gpt-oss), llama.cpp and the fixture spell it the same
        known(
                new TiktokenSplitter(
                        NO_CONTRACTION, CASED, true, UP_TO_THREE, false, NEWLINES_SLASHES, true),
                String.format(cased, "(?i:" + contraction + ")?", "(?i:" + contraction + ")?")
                        + "|\\p{N}{1,3}|"
                        + other
                        + "[\\r\\n/]*"
                        + tail);
        // Tekken (Ministral): o200k's words without the contraction, one digit at a time
        known(
                new TiktokenSplitter(
                        NO_CONTRACTION, CASED, false, ONE, false, NEWLINES_SLASHES, true),
                String.format(cased, "", "") + "|\\p{N}|" + other + "[\\r\\n/]*" + tail);
        // Qwen2 and friends: cl100k with one digit at a time, tokenizer.json and llama.cpp
        known(
                new TiktokenSplitter(FOLD, PREFIX_LETTERS, false, ONE, false, NEWLINES, true),
                "(?i:"
                        + contraction
                        + ")|"
                        + prefix
                        + "\\p{L}+|\\p{N}|"
                        + other
                        + "[\\r\\n]*"
                        + tail);
        known(
                new TiktokenSplitter(ASCII_BOTH, PREFIX_LETTERS, false, ONE, false, NEWLINES, true),
                asciiBoth + "|" + prefix + "\\p{L}+|\\p{N}|" + other + "[\\r\\n]*" + tail);
        // Qwen3.5: marks belong to words
        known(
                new TiktokenSplitter(FOLD, PREFIX_LETTERS_MARKS, false, ONE, true, NEWLINES, true),
                "(?i:"
                        + contraction
                        + ")|"
                        + prefix
                        + "[\\p{L}\\p{M}]+|\\p{N}|"
                        + " ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*"
                        + tail);
    }

    private static void known(TiktokenSplitter splitter, String... regexes) {
        for (String regex : regexes) {
            KNOWN.put(regex, splitter);
        }
    }

    /** The scanner for a known regex, compiled with Unicode character classes; else null. */
    static Splitter forPattern(Pattern pattern) {
        int required = Pattern.UNICODE_CHARACTER_CLASS;
        if ((pattern.flags() & required) != required) {
            return null;
        }
        return KNOWN.get(pattern.pattern());
    }

    static Iterable<String> knownPatterns() {
        return KNOWN.keySet();
    }

    @Override
    public void splitAll(
            CharSequence text, int startInclusive, int endExclusive, SplitConsumer consumer) {
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(consumer, "consumer");
        if (startInclusive < 0 || endExclusive < startInclusive || endExclusive > text.length()) {
            throw new IndexOutOfBoundsException(
                    "Invalid range ["
                            + startInclusive
                            + ", "
                            + endExclusive
                            + ") for text length "
                            + text.length());
        }
        int i = startInclusive;
        while (i < endExclusive) {
            int next = match(text, i, endExclusive);
            consumer.accept(text, i, next);
            i = next;
        }
    }

    /** The end of the piece that starts at {@code i}: the alternatives in the regex's order. */
    private int match(CharSequence text, int i, int end) {
        int cp = codePointAt(text, i, end);
        int cls = CharClasses.of(cp);
        int e;
        if (contraction != NO_CONTRACTION && cp == '\'') {
            e = contraction(text, i, end, contraction);
            if (e > 0) {
                return e;
            }
        }
        e = word(text, i, end, cp, cls);
        if (e > 0) {
            return e;
        }
        e = number(text, i, end, cp, cls);
        if (e > 0) {
            return e;
        }
        e = other(text, i, end, cp, cls);
        if (e > 0) {
            return e;
        }
        // Only whitespace is left: '\s*[\r\n]+', then '\s+(?!\S)', then '\s+'.
        int run = i + Character.charCount(cp);
        int lastNewline = (cls & CharClasses.NEWLINE) != 0 ? run : -1;
        while (run < end) {
            int c = codePointAt(text, run, end);
            int k = CharClasses.of(c);
            if ((k & CharClasses.SPACE) == 0) {
                break;
            }
            run += Character.charCount(c);
            if ((k & CharClasses.NEWLINE) != 0) {
                lastNewline = run;
            }
        }
        if (newlineRun && lastNewline > 0) {
            return lastNewline;
        }
        if (run == end || run - i == 1) {
            return run;
        }
        return run - 1; // the last blank goes with what follows it
    }

    private static int contraction(CharSequence text, int i, int end, int mode) {
        int p1 = i + 1;
        if (p1 >= end) {
            return -1;
        }
        int c1 = codePointAt(text, p1, end);
        if (is(c1, 's', mode) || is(c1, 't', mode) || is(c1, 'm', mode) || is(c1, 'd', mode)) {
            return p1 + Character.charCount(c1);
        }
        int p2 = p1 + Character.charCount(c1);
        if (p2 >= end) {
            return -1;
        }
        int c2 = codePointAt(text, p2, end);
        if ((is(c1, 'r', mode) && is(c2, 'e', mode))
                || (is(c1, 'v', mode) && is(c2, 'e', mode))
                || (is(c1, 'l', mode) && is(c2, 'l', mode))) {
            return p2 + Character.charCount(c2);
        }
        return -1;
    }

    private static boolean is(int c, char lower, int mode) {
        if (c == lower) {
            return true;
        }
        switch (mode) {
            case ASCII_BOTH:
                return c == lower - ('a' - 'A');
            case FOLD:
                return Character.toLowerCase(Character.toUpperCase(c)) == lower;
            default:
                return false;
        }
    }

    private int word(CharSequence text, int i, int end, int cp, int cls) {
        int after = i + Character.charCount(cp);
        switch (word) {
            case SPACE_LETTERS:
                if ((cls & CharClasses.LETTER) != 0) {
                    return run(text, after, end, CharClasses.LETTER);
                }
                if (cp == ' ' && after < end) {
                    int c = codePointAt(text, after, end);
                    if ((CharClasses.of(c) & CharClasses.LETTER) != 0) {
                        return run(text, after + Character.charCount(c), end, CharClasses.LETTER);
                    }
                }
                return -1;
            case PREFIX_LETTERS:
            case PREFIX_LETTERS_MARKS:
                int letters =
                        word == PREFIX_LETTERS
                                ? CharClasses.LETTER
                                : CharClasses.LETTER | CharClasses.MARK;
                if ((cls & letters) != 0) {
                    return run(text, after, end, letters);
                }
                if (prefixable(cls) && after < end) {
                    int c = codePointAt(text, after, end);
                    if ((CharClasses.of(c) & letters) != 0) {
                        return run(text, after + Character.charCount(c), end, letters);
                    }
                }
                return -1;
            default:
                return cased(text, i, end, cls, after);
        }
    }

    /**
     * o200k's word: {@code P?U*L+ | P?U+L*}, tried in the regex's order, with U and L overlapping
     * (Lm, Lo and marks are in both). {@code U*L+} backtracks to the last L-class char of the U run
     * when nothing L-class follows it.
     */
    private int cased(CharSequence text, int i, int end, int cls, int after) {
        boolean prefixed = prefixable(cls);
        int e;
        if (prefixed && (e = upperThenLower(text, after, end)) > 0) {
            return suffixed(text, e, end);
        }
        if ((e = upperThenLower(text, i, end)) > 0) {
            return suffixed(text, e, end);
        }
        if (prefixed && (e = upperMaybeLower(text, after, end)) > 0) {
            return suffixed(text, e, end);
        }
        if ((e = upperMaybeLower(text, i, end)) > 0) {
            return suffixed(text, e, end);
        }
        return -1;
    }

    /** {@code U*L+} from {@code j}, or -1. */
    private static int upperThenLower(CharSequence text, int j, int end) {
        int u = j;
        int lastLower = -1;
        while (u < end) {
            int c = codePointAt(text, u, end);
            int k = CharClasses.of(c);
            if ((k & CharClasses.UPPER) == 0) {
                break;
            }
            u += Character.charCount(c);
            if ((k & CharClasses.LOWER) != 0) {
                lastLower = u;
            }
        }
        int l = run(text, u, end, CharClasses.LOWER);
        return l > u ? l : lastLower;
    }

    /** {@code U+L*} from {@code j}, or -1. */
    private static int upperMaybeLower(CharSequence text, int j, int end) {
        int u = run(text, j, end, CharClasses.UPPER);
        return u > j ? run(text, u, end, CharClasses.LOWER) : -1;
    }

    private int suffixed(CharSequence text, int e, int end) {
        if (wordSuffix && e < end && text.charAt(e) == '\'') {
            int s = contraction(text, e, end, FOLD);
            if (s > 0) {
                return s;
            }
        }
        return e;
    }

    private int number(CharSequence text, int i, int end, int cp, int cls) {
        int after = i + Character.charCount(cp);
        if ((cls & CharClasses.NUMBER) == 0) {
            if (numbers == SPACE_RUN && cp == ' ' && after < end) {
                int c = codePointAt(text, after, end);
                if ((CharClasses.of(c) & CharClasses.NUMBER) != 0) {
                    return run(text, after + Character.charCount(c), end, CharClasses.NUMBER);
                }
            }
            return -1;
        }
        switch (numbers) {
            case ONE:
                return after;
            case UP_TO_THREE:
                int e = after;
                for (int n = 1; n < 3 && e < end; n++) {
                    int c = codePointAt(text, e, end);
                    if ((CharClasses.of(c) & CharClasses.NUMBER) == 0) {
                        break;
                    }
                    e += Character.charCount(c);
                }
                return e;
            default:
                return run(text, after, end, CharClasses.NUMBER);
        }
    }

    private int other(CharSequence text, int i, int end, int cp, int cls) {
        int j = i;
        if (cp == ' ') {
            if (i + 1 >= end || !isOther(CharClasses.of(codePointAt(text, i + 1, end)))) {
                return -1;
            }
            j = i + 1;
        } else if (!isOther(cls)) {
            return -1;
        }
        int e = j;
        do {
            int c = codePointAt(text, e, end);
            if (!isOther(CharClasses.of(c))) {
                break;
            }
            e += Character.charCount(c);
        } while (e < end);
        if (otherTail != NO_TAIL) {
            while (e < end) {
                char c = text.charAt(e);
                if (c == '\r' || c == '\n' || (otherTail == NEWLINES_SLASHES && c == '/')) {
                    e++;
                } else {
                    break;
                }
            }
        }
        return e;
    }

    private boolean isOther(int cls) {
        int excluded = CharClasses.SPACE | CharClasses.LETTER | CharClasses.NUMBER;
        if (otherExcludesMarks) {
            excluded |= CharClasses.MARK;
        }
        return (cls & excluded) == 0;
    }

    /** '[^\r\n\p{L}\p{N}]': what may precede a word. */
    private static boolean prefixable(int cls) {
        return (cls & (CharClasses.NEWLINE | CharClasses.LETTER | CharClasses.NUMBER)) == 0;
    }

    /** The end of the run of code points in {@code classes} that starts at {@code j}. */
    private static int run(CharSequence text, int j, int end, int classes) {
        while (j < end) {
            int c = codePointAt(text, j, end);
            if ((CharClasses.of(c) & classes) == 0) {
                break;
            }
            j += Character.charCount(c);
        }
        return j;
    }

    /**
     * The code point at {@code i}; a surrogate without its partner before {@code end} is itself.
     */
    private static int codePointAt(CharSequence text, int i, int end) {
        char c = text.charAt(i);
        if (Character.isHighSurrogate(c) && i + 1 < end) {
            char d = text.charAt(i + 1);
            if (Character.isLowSurrogate(d)) {
                return Character.toCodePoint(c, d);
            }
        }
        return c;
    }

    /**
     * The character classes of the patterns, as java.util.regex defines them under
     * UNICODE_CHARACTER_CLASS: a bit per class, tabulated for the BMP.
     */
    static final class CharClasses {
        static final int SPACE = 1; // \s
        static final int LETTER = 2; // \p{L}
        static final int NUMBER = 4; // \p{N}
        static final int MARK = 8; // \p{M}
        static final int UPPER = 16; // [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]
        static final int LOWER = 32; // [\p{Ll}\p{Lm}\p{Lo}\p{M}]
        static final int NEWLINE = 64; // [\r\n]

        private static final byte[] BMP = new byte[0x10000];

        static {
            for (int c = 0; c < BMP.length; c++) {
                BMP[c] = (byte) compute(c);
            }
        }

        private CharClasses() {}

        static int of(int codePoint) {
            return codePoint < 0x10000 ? BMP[codePoint] : compute(codePoint);
        }

        static int compute(int cp) {
            int type = Character.getType(cp);
            int cls = 0;
            switch (type) {
                case Character.UPPERCASE_LETTER:
                case Character.TITLECASE_LETTER:
                    cls = LETTER | UPPER;
                    break;
                case Character.LOWERCASE_LETTER:
                    cls = LETTER | LOWER;
                    break;
                case Character.MODIFIER_LETTER:
                case Character.OTHER_LETTER:
                    cls = LETTER | UPPER | LOWER;
                    break;
                case Character.NON_SPACING_MARK:
                case Character.ENCLOSING_MARK:
                case Character.COMBINING_SPACING_MARK:
                    cls = MARK | UPPER | LOWER;
                    break;
                case Character.DECIMAL_DIGIT_NUMBER:
                case Character.LETTER_NUMBER:
                case Character.OTHER_NUMBER:
                    cls = NUMBER;
                    break;
                case Character.SPACE_SEPARATOR:
                case Character.LINE_SEPARATOR:
                case Character.PARAGRAPH_SEPARATOR:
                    cls = SPACE;
                    break;
                default:
                    break;
            }
            if ((cp >= 0x9 && cp <= 0xD) || cp == 0x85) {
                cls |= SPACE;
            }
            if (cp == '\r' || cp == '\n') {
                cls |= NEWLINE;
            }
            return cls;
        }
    }
}
