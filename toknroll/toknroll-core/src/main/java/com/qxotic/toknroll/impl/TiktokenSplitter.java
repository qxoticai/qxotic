package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.spi.CharClasses;
import com.qxotic.toknroll.spi.TiktokenEngine;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
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
    private static final int DEEPSEEK = 4; // DeepSeek V3's whole rule, its own match

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

    // Cuts a first stage makes before this scanner runs, each piece as if the text ended there:
    // '\p{N}{1,3}' and '\p{N}' cut around numbers, Laguna's newline stage before a newline run.
    static final int NO_CUTS = 0;
    static final int NUMBER_RUNS = 1; // around each run of numbers
    static final int EACH_NUMBER = 2; // around each number
    static final int NEWLINE_RUNS = 3; // before each '(?:\r?\n)+'
    static final int NUMBERS_CJK = 4; // '\p{N}{1,3}' apart, then each run of DeepSeek's CJK range

    private final int rules; // the TiktokenEngine rule set this scanner is, or -1
    private final int cuts;
    private final TiktokenSplitter base; // the same scanner without cuts: what an engine runs
    private final String pattern; // the regex this scanner stands for, or null

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
        this.rules = engineRules();
        this.cuts = NO_CUTS;
        this.base = this;
        this.pattern = null;
    }

    private TiktokenSplitter(TiktokenSplitter base, int cuts, String pattern) {
        this.contraction = base.contraction;
        this.word = base.word;
        this.wordSuffix = base.wordSuffix;
        this.numbers = base.numbers;
        this.otherExcludesMarks = base.otherExcludesMarks;
        this.otherTail = base.otherTail;
        this.newlineRun = base.newlineRun;
        // the engine has one rule set with cuts, r50k's around numbers; for the rest it runs the
        // scanner without them on each piece of the first stage
        this.rules =
                cuts == EACH_NUMBER && base.rules == TiktokenEngine.R50K
                        ? TiktokenEngine.R50K_NUMBERS
                        : base.rules;
        this.cuts = cuts;
        this.base = cuts == NO_CUTS ? this : base;
        this.pattern = pattern;
    }

    /** Whether core cuts the text before an engine runs: a first stage the engine's rules lack. */
    boolean coreCuts() {
        return cuts != NO_CUTS && rules != TiktokenEngine.R50K_NUMBERS;
    }

    /** This scanner without its cuts. */
    TiktokenSplitter base() {
        return base;
    }

    /** How many rule sets {@link #rules} names: {@link TiktokenEngine}'s run from 0 to the last. */
    static final int RULE_SETS = TiktokenEngine.R50K_NUMBERS + 1;

    /** Which of the {@link TiktokenEngine} rule sets this scanner is, or -1. */
    int rules() {
        return rules;
    }

    private int engineRules() {
        if (otherExcludesMarks) {
            return -1;
        }
        if (word == SPACE_LETTERS) {
            boolean r50k =
                    contraction == LOWER
                            && !wordSuffix
                            && numbers == SPACE_RUN
                            && otherTail == NO_TAIL
                            && !newlineRun;
            return r50k ? TiktokenEngine.R50K : -1;
        }
        if (word == CASED) {
            if (contraction != NO_CONTRACTION || otherTail != NEWLINES_SLASHES || !newlineRun) {
                return -1;
            }
            if (wordSuffix && numbers == UP_TO_THREE) {
                return TiktokenEngine.O200K;
            }
            return !wordSuffix && numbers == ONE ? TiktokenEngine.TEKKEN : -1;
        }
        boolean family =
                word == PREFIX_LETTERS
                        && !wordSuffix
                        && otherTail == NEWLINES
                        && newlineRun
                        && (contraction == FOLD || contraction == ASCII_BOTH);
        if (!family) {
            return -1;
        }
        boolean fold = contraction == FOLD;
        switch (numbers) {
            case UP_TO_THREE:
                return fold ? TiktokenEngine.CL100K : TiktokenEngine.CL100K_ASCII;
            case ONE:
                return fold ? TiktokenEngine.QWEN2 : TiktokenEngine.QWEN2_ASCII;
            default:
                return -1;
        }
    }

    /**
     * Text with every kind of piece of the family and the rare shapes a fast engine may handle
     * apart: a run longer than a block, a character or a contraction cut by a block edge, cased
     * runs, long words, marks, surrogates.
     */
    static String sample() {
        StringBuilder sample = new StringBuilder();
        for (int i = 0; i < 40; i++) {
            sample.append("I'd say it's 12345 pieces!! \n\n  \r\n x  \t y ")
                    .append("naïve 東京 ٣٤٥ 𝐀𝐁 'ſ 'LL'VE (?!)  \u00a0\u2028\n\n\n  z");
            sample.append(" ".repeat(i * 7)).append(i % 3 == 0 ? "\n" : "a").append('\'');
            sample.append("x".repeat(60 + i)).append("'s Q!́a !!́A ʰABC 中Qz aB McDonald /\n/#\r\n");
        }
        sample.append("\n").append(" ".repeat(5000)).append("\n x\n").append("\t".repeat(5000));
        sample.append("a".repeat(200)).append("B").append("c".repeat(200)).append("D");
        sample.append("E".repeat(200)).append("ʰ").append("F".repeat(200)).append("g");
        sample.append("١".repeat(100)).append("7".repeat(100)).append(" ").append("é".repeat(70));
        sample.append("😀".repeat(70)).append("\u0301".repeat(70)).append("\uD800\uDC00");
        sample.append(" x\uD800 \uDC00y");
        return sample.toString();
    }

    /**
     * Laguna's first stage: a newline run, CRLF included, starts a piece that runs to the next, and
     * the text before the first run is a piece. Possessive, so java.util.regex does not recurse per
     * character.
     */
    static final String LAGUNA_NEWLINES =
            "(?:\\r?\\n)++[^\\r\\n]*+(?:\\r(?!\\n)[^\\r\\n]*+)*+"
                    + "|(?=[^\\n])(?!\\r\\n)[^\\r\\n]*+(?:\\r(?!\\n)[^\\r\\n]*+)*+";

    /** The regex strings the loaders compile, each with the scanner that splits like it. */
    private static final Map<String, TiktokenSplitter> KNOWN = new HashMap<>();

    /**
     * Staged splits, the stages joined by NULs (which no regex here holds), each with the scanner
     * that splits like all of them: the last stage's scanner, cut where the stages before cut.
     */
    private static final Map<String, TiktokenSplitter> KNOWN_STAGES = new HashMap<>();

    static {
        String contraction = "'s|'t|'re|'ve|'m|'ll|'d";
        String asciiBoth = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])";
        String prefix = "[^\\r\\n\\p{L}\\p{N}]?";
        String other = " ?[^\\s\\p{L}\\p{N}]+";
        String upper = "[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]";
        String lower = "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]";
        String cased = prefix + upper + "*" + lower + "+%s|" + prefix + upper + "+" + lower + "*%s";
        String tail = "|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
        // the cl100k family with '[sS]' contractions, its number rule left open
        String asciiBody = asciiBoth + "|" + prefix + "\\p{L}+|%s|" + other + "[\\r\\n]*" + tail;

        // r50k: tiktoken and the ByteLevel pre-tokenizer, then the tiktoken fixture's spelling
        TiktokenSplitter r50k =
                new TiktokenSplitter(LOWER, SPACE_LETTERS, false, SPACE_RUN, false, NO_TAIL, false);
        String gpt2 = contraction + "| ?\\p{L}+| ?\\p{N}+|" + other + "|\\s+(?!\\S)|\\s+";
        known(
                r50k,
                gpt2,
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
        TiktokenSplitter cl100kAscii =
                new TiktokenSplitter(
                        ASCII_BOTH, PREFIX_LETTERS, false, UP_TO_THREE, false, NEWLINES, true);
        known(cl100kAscii, String.format(asciiBody, "\\p{N}{1,3}"));
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
        TiktokenSplitter qwen2Fold =
                new TiktokenSplitter(FOLD, PREFIX_LETTERS, false, ONE, false, NEWLINES, true);
        String qwen2Folded =
                "(?i:"
                        + contraction
                        + ")|"
                        + prefix
                        + "\\p{L}+|\\p{N}|"
                        + other
                        + "[\\r\\n]*"
                        + tail;
        known(qwen2Fold, qwen2Folded);
        TiktokenSplitter qwen2Ascii =
                new TiktokenSplitter(ASCII_BOTH, PREFIX_LETTERS, false, ONE, false, NEWLINES, true);
        String qwen2 = String.format(asciiBody, "\\p{N}");
        known(
                qwen2Ascii,
                qwen2,
                // Bailing (Ling, Ring): the same pieces, spelled apart. '\s*[\r\n]' ends where
                // '\s*[\r\n]+' does, at the last newline of the blank run.
                "'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|"
                        + prefix
                        + "\\p{L}+|\\p{N}|"
                        + other
                        + "[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+");
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

        // Two stages one scanner splits in a pass: the second stage's, cut where the first cuts,
        // each piece as if the text ended there.
        // SmolLM, Command R, EXAONE, Mellum 2: every digit apart, then GPT-2
        staged(r50k, EACH_NUMBER, "\\p{N}", gpt2);
        // MiniCPM5: runs of up to three numbers, then cl100k with whole runs; a piece of the first
        // stage holds three at most, so cl100k's own '\p{N}{1,3}' cuts it alike
        staged(cl100kAscii, NUMBER_RUNS, "\\p{N}{1,3}", String.format(asciiBody, "\\p{N}+"));
        // Laguna: a newline run, CRLF included, starts a piece that runs to the next; then Qwen2
        staged(qwen2Ascii, NEWLINE_RUNS, LAGUNA_NEWLINES, qwen2);
        // Laguna's tokenizer.json: the same, its contractions case-folded as Hugging Face folds
        // them
        staged(qwen2Fold, NEWLINE_RUNS, LAGUNA_NEWLINES, qwen2Folded);
        // DeepSeek V3 and V4: number groups and CJK runs apart, then its own rule; tokenizer.json
        // writes the newlines as characters, llama.cpp as escapes
        TiktokenSplitter deepSeek =
                new TiktokenSplitter(NO_CONTRACTION, DEEPSEEK, false, ONE, false, NEWLINES, true);
        for (String nl : new String[] {"\r\n", "\\r\\n"}) {
            staged(
                    deepSeek,
                    NUMBERS_CJK,
                    "\\p{N}{1,3}",
                    "[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+",
                    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^"
                            + nl
                            + "\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+["
                            + nl
                            + "]*|\\s*["
                            + nl
                            + "]+|\\s+(?!\\S)|\\s+");
        }
    }

    private static void known(TiktokenSplitter splitter, String... regexes) {
        for (String regex : regexes) {
            KNOWN.put(regex, new TiktokenSplitter(splitter, NO_CUTS, regex));
        }
    }

    private static void staged(TiktokenSplitter scanner, int cuts, String... stages) {
        KNOWN_STAGES.put(String.join("\0", stages), new TiktokenSplitter(scanner, cuts, null));
    }

    /**
     * The scanner for two known stages, each a regex with Unicode character classes only (or the
     * scanner one compiles to), the first keeping the text between its matches; else null.
     */
    static Splitter forStages(Splitter... stages) {
        if (stages.length < 2) {
            return null;
        }
        StringBuilder key = new StringBuilder();
        for (int k = 0; k < stages.length; k++) {
            boolean keepsGaps =
                    stages[k] instanceof RegexSplitter && ((RegexSplitter) stages[k]).emitsGaps();
            if (k < stages.length - 1 && !keepsGaps) {
                return null;
            }
            // a stage with no regex is "null" in the key, which no entry has
            key.append(k > 0 ? "\0" : "").append(regex(stages[k]));
        }
        return KNOWN_STAGES.get(key.toString());
    }

    private static String regex(Splitter stage) {
        if (stage instanceof TiktokenSplitter) {
            return ((TiktokenSplitter) stage).pattern;
        }
        if (stage instanceof RegexSplitter) {
            Pattern pattern = ((RegexSplitter) stage).pattern();
            return unicodeOnly(pattern) ? pattern.pattern() : null;
        }
        return null;
    }

    /**
     * Whether {@code pattern} is compiled with Unicode character classes and no flag that changes
     * what it matches, as the scanner's regexes are.
     */
    private static boolean unicodeOnly(Pattern pattern) {
        int flags = pattern.flags();
        return (flags & Pattern.UNICODE_CHARACTER_CLASS) != 0
                && (flags & ~(Pattern.UNICODE_CHARACTER_CLASS | Pattern.UNICODE_CASE)) == 0;
    }

    /** The stages {@link #forStages} knows, joined by NULs, which no regex here holds. */
    static Iterable<String> knownStages() {
        return KNOWN_STAGES.keySet();
    }

    /** The scanner for a known regex, compiled with Unicode character classes only; else null. */
    static Splitter forPattern(Pattern pattern) {
        return unicodeOnly(pattern) ? KNOWN.get(pattern.pattern()) : null;
    }

    static Iterable<String> knownPatterns() {
        return KNOWN.keySet();
    }

    /**
     * Piece ends per batch: the model consumes them from an array, not from a callback per word.
     */
    static final int BATCH = 1024;

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
        // The scanner reads UTF-8; a piece's end comes back to chars by counting its code points,
        // and the four-byte ones twice.
        byte[] utf8 = utf8(text, startInclusive, endExclusive);
        int length = utf8.length;
        int[] ends = new int[Math.min(BATCH, Math.max(1, length))];
        int cursor = 0;
        int charCursor = startInclusive;
        while (cursor < length) {
            int n = split(utf8, cursor, length, ends);
            for (int k = 0; k < n; k++) {
                int end = ends[k];
                int chars = 0;
                for (int i = cursor; i < end; i++) {
                    int b = utf8[i] & 0xFF;
                    if ((b & 0xC0) != 0x80) {
                        chars += b >= 0xF0 ? 2 : 1;
                    }
                }
                consumer.accept(text, charCursor, charCursor + chars);
                charCursor += chars;
                cursor = end;
            }
        }
    }

    /**
     * Cuts pieces from {@code text[start, end)} into {@code ends}, each entry the end of one piece,
     * until the array is full or the text ends. Returns how many. Reads no further than {@code
     * end}.
     */
    int split(byte[] text, int start, int end, int[] ends) {
        int n = 0;
        int i = start;
        while (i < end && n < ends.length) {
            int stop = cuts == NO_CUTS ? end : cut(text, i, end);
            while (i < stop && n < ends.length) {
                i = match(text, i, stop);
                ends[n++] = i;
            }
        }
        return n;
    }

    /**
     * The end of the piece of the first stage that holds {@code i}: the scanner sees it as the
     * whole text. Reads no further than {@code end}, so the text needs no padding.
     */
    int cut(byte[] text, int i, int end) {
        if (cuts == NEWLINE_RUNS) {
            int j = i;
            while (j < end) { // the newline run this piece may start with
                if (text[j] == '\n') {
                    j++;
                } else if (text[j] == '\r' && j + 1 < end && text[j + 1] == '\n') {
                    j += 2;
                } else {
                    break;
                }
            }
            while (true) { // then up to the next run
                j = nextNewline(text, j, end);
                if (j == end || text[j] == '\n' || (j + 1 < end && text[j + 1] == '\n')) {
                    return j;
                }
                j++; // a lone '\r' is text
            }
        }
        int at = at(text, i, end);
        int after = i + len(at);
        boolean deepSeek = cuts == NUMBERS_CJK;
        if ((CharClasses.of(cp(at)) & CharClasses.NUMBER) != 0) {
            if (deepSeek) {
                return twoMoreNumbers(text, after, end);
            }
            return cuts == EACH_NUMBER ? after : run(text, i, end, CharClasses.NUMBER);
        }
        if (deepSeek && cjk(cp(at))) {
            int e = after;
            for (int c; e < end && cjk(cp(c = at(text, e, end))); ) {
                e += len(c);
            }
            return e;
        }
        return nextNumber(text, after, end, deepSeek);
    }

    /** The end of '\p{N}{1,3}' whose first number ends at {@code e}. */
    private static int twoMoreNumbers(byte[] text, int e, int end) {
        for (int n = 1; n < 3 && e < end; n++) {
            int c = at(text, e, end);
            if ((CharClasses.of(cp(c)) & CharClasses.NUMBER) == 0) {
                break;
            }
            e += len(c);
        }
        return e;
    }

    /** DeepSeek's '[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]': Han, then the kana. */
    private static boolean cjk(int cp) {
        return (cp >= 0x4E00 && cp <= 0x9FA5) || (cp >= 0x3040 && cp <= 0x30FF);
    }

    /** The first number at or after {@code j}, or CJK char too if asked, or {@code end}. */
    private static int nextNumber(byte[] text, int j, int end, boolean orCjk) {
        while (true) {
            if (end - j >= 8) {
                // Eight ASCII bytes at a time; a digit or a non-ASCII byte stops the scan.
                long v = (long) LONGS.get(text, j);
                long stop = (v | ascii(v, CharClasses.NUMBER)) & HIGH_BITS;
                if (stop == 0) {
                    j += 8;
                    continue;
                }
                j += Long.numberOfTrailingZeros(stop) >>> 3;
            } else if (j >= end) {
                return end;
            }
            int c = at(text, j, end);
            if ((CharClasses.of(cp(c)) & CharClasses.NUMBER) != 0 || orCjk && cjk(cp(c))) {
                return j;
            }
            j += len(c);
        }
    }

    /** The first '\r' or '\n' at or after {@code j}, or {@code end}. */
    private static int nextNewline(byte[] text, int j, int end) {
        while (end - j >= 8) {
            long v = (long) LONGS.get(text, j);
            long stop = ascii(v, CharClasses.NEWLINE) & ~v & HIGH_BITS;
            if (stop != 0) {
                return j + (Long.numberOfTrailingZeros(stop) >>> 3);
            }
            j += 8;
        }
        while (j < end && text[j] != '\n' && text[j] != '\r') {
            j++;
        }
        return j;
    }

    /** The UTF-8 of {@code text[start, end)}; a surrogate without its partner becomes U+FFFD. */
    static byte[] utf8(CharSequence text, int start, int end) {
        int length = 0;
        for (int i = start; i < end; i++) {
            char c = text.charAt(i);
            if (c < 0x80) {
                length++;
            } else if (c < 0x800) {
                length += 2;
            } else if (Character.isHighSurrogate(c)
                    && i + 1 < end
                    && Character.isLowSurrogate(text.charAt(i + 1))) {
                length += 4;
                i++;
            } else {
                length += 3;
            }
        }
        byte[] out = new byte[length];
        int o = 0;
        for (int i = start; i < end; i++) {
            char c = text.charAt(i);
            if (c < 0x80) {
                out[o++] = (byte) c;
            } else if (c < 0x800) {
                out[o++] = (byte) (0xC0 | (c >> 6));
                out[o++] = (byte) (0x80 | (c & 0x3F));
            } else if (Character.isHighSurrogate(c)
                    && i + 1 < end
                    && Character.isLowSurrogate(text.charAt(i + 1))) {
                int cp = Character.toCodePoint(c, text.charAt(++i));
                out[o++] = (byte) (0xF0 | (cp >> 18));
                out[o++] = (byte) (0x80 | ((cp >> 12) & 0x3F));
                out[o++] = (byte) (0x80 | ((cp >> 6) & 0x3F));
                out[o++] = (byte) (0x80 | (cp & 0x3F));
            } else {
                int cp = Character.isSurrogate(c) ? 0xFFFD : c;
                out[o++] = (byte) (0xE0 | (cp >> 12));
                out[o++] = (byte) (0x80 | ((cp >> 6) & 0x3F));
                out[o++] = (byte) (0x80 | (cp & 0x3F));
            }
        }
        return out;
    }

    /** The end of the piece that starts at {@code i}: the alternatives in the regex's order. */
    private int match(byte[] text, int i, int end) {
        if (word == DEEPSEEK) {
            return deepSeek(text, i, end);
        }
        int at = at(text, i, end);
        int cp = cp(at);
        int cls = CharClasses.of(cp);
        int e;
        if (contraction != NO_CONTRACTION && cp == '\'') {
            e = contraction(text, i, end, contraction);
            if (e > 0) {
                return e;
            }
        }
        e = word(text, i, end, at, cls);
        if (e > 0) {
            return e;
        }
        e = number(text, i, end, at, cls);
        if (e > 0) {
            return e;
        }
        e = other(text, i, end, cp, cls);
        if (e > 0) {
            return e;
        }
        return blanks(text, i, end, at, cls, newlineRun);
    }

    /** Whitespace at {@code i}: '\s*[\r\n]+' if asked for, then '\s+(?!\S)', then '\s+'. */
    private static int blanks(byte[] text, int i, int end, int at, int cls, boolean newlineRun) {
        int run = i + len(at);
        int last = len(at);
        int lastNewline = (cls & CharClasses.NEWLINE) != 0 ? run : -1;
        while (run < end) {
            int c = at(text, run, end);
            int k = CharClasses.of(cp(c));
            if ((k & CharClasses.SPACE) == 0) {
                break;
            }
            last = len(c);
            run += last;
            if ((k & CharClasses.NEWLINE) != 0) {
                lastNewline = run;
            }
        }
        if (newlineRun && lastNewline > 0) {
            return lastNewline;
        }
        if (run == end || run - i == len(at)) {
            return run;
        }
        return run - last; // the last blank goes with what follows it
    }

    /**
     * DeepSeek V3's last stage, its alternatives in order: {@code [!-/:-@\[-`{-~][A-Za-z]+}, {@code
     * [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+}, {@code ?[\p{P}\p{S}]+[\r\n]*}, then the blanks. Text
     * where none of them starts is a piece of its own, as Hugging Face's Split keeps it.
     */
    private static int deepSeek(byte[] text, int i, int end) {
        int e = deepSeekMatch(text, i, end);
        if (e > 0) {
            return e;
        }
        int j = i + len(at(text, i, end));
        while (j < end && deepSeekMatch(text, j, end) < 0) { // ponytail: gaps are rare and short
            j += len(at(text, j, end));
        }
        return j;
    }

    private static int deepSeekMatch(byte[] text, int i, int end) {
        int at = at(text, i, end);
        int cp = cp(at);
        int cls = CharClasses.of(cp);
        int after = i + len(at);
        boolean ps = punctOrSymbol(cp);
        if (ps && cp < 0x80 && after < end && asciiLetter(text[after])) {
            int e = after + 1;
            while (e < end && asciiLetter(text[e])) {
                e++;
            }
            return e;
        }
        int words = CharClasses.LETTER | CharClasses.MARK;
        if ((cls & words) != 0) {
            return run(text, after, end, words);
        }
        if (!ps && (cls & CharClasses.NEWLINE) == 0 && after < end) {
            int c = at(text, after, end);
            if ((CharClasses.of(cp(c)) & words) != 0) {
                return run(text, after + len(c), end, words);
            }
        }
        boolean spaced = cp == ' ' && after < end && punctOrSymbol(cp(at(text, after, end)));
        if (ps || spaced) {
            int e = spaced ? after : i;
            for (int c; e < end && punctOrSymbol(cp(c = at(text, e, end))); ) {
                e += len(c);
            }
            while (e < end && (text[e] == '\r' || text[e] == '\n')) {
                e++;
            }
            return e;
        }
        return (cls & CharClasses.SPACE) != 0 ? blanks(text, i, end, at, cls, true) : -1;
    }

    private static boolean asciiLetter(byte b) {
        return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z');
    }

    /** '[\p{P}\p{S}]'. */
    static boolean punctOrSymbol(int cp) {
        switch (Character.getType(cp)) {
            case Character.CONNECTOR_PUNCTUATION:
            case Character.DASH_PUNCTUATION:
            case Character.START_PUNCTUATION:
            case Character.END_PUNCTUATION:
            case Character.INITIAL_QUOTE_PUNCTUATION:
            case Character.FINAL_QUOTE_PUNCTUATION:
            case Character.OTHER_PUNCTUATION:
            case Character.MATH_SYMBOL:
            case Character.CURRENCY_SYMBOL:
            case Character.MODIFIER_SYMBOL:
            case Character.OTHER_SYMBOL:
                return true;
            default:
                return false;
        }
    }

    private static int contraction(byte[] text, int i, int end, int mode) {
        int p1 = i + 1;
        if (p1 >= end) {
            return -1;
        }
        int a1 = at(text, p1, end);
        int c1 = cp(a1);
        if (is(c1, 's', mode) || is(c1, 't', mode) || is(c1, 'm', mode) || is(c1, 'd', mode)) {
            return p1 + len(a1);
        }
        int p2 = p1 + len(a1);
        if (p2 >= end) {
            return -1;
        }
        int a2 = at(text, p2, end);
        int c2 = cp(a2);
        if ((is(c1, 'r', mode) && is(c2, 'e', mode))
                || (is(c1, 'v', mode) && is(c2, 'e', mode))
                || (is(c1, 'l', mode) && is(c2, 'l', mode))) {
            return p2 + len(a2);
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

    private int word(byte[] text, int i, int end, int at, int cls) {
        int after = i + len(at);
        switch (word) {
            case SPACE_LETTERS:
                if ((cls & CharClasses.LETTER) != 0) {
                    return run(text, after, end, CharClasses.LETTER);
                }
                if (cp(at) == ' ' && after < end) {
                    int c = at(text, after, end);
                    if ((CharClasses.of(cp(c)) & CharClasses.LETTER) != 0) {
                        return run(text, after + len(c), end, CharClasses.LETTER);
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
                    int c = at(text, after, end);
                    if ((CharClasses.of(cp(c)) & letters) != 0) {
                        return run(text, after + len(c), end, letters);
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
    private int cased(byte[] text, int i, int end, int cls, int after) {
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
    private static int upperThenLower(byte[] text, int j, int end) {
        int u = j;
        int lastLower = -1;
        while (true) {
            if (end - u >= 8) {
                // Eight ASCII capitals at a time; an ASCII byte of any other class ends the run.
                long v = (long) LONGS.get(text, u);
                long stop = (v | ~ascii(v, CharClasses.UPPER)) & HIGH_BITS;
                if (stop == 0) {
                    u += 8;
                    continue;
                }
                u += Long.numberOfTrailingZeros(stop) >>> 3;
                if (text[u] >= 0) {
                    break;
                }
            } else if (u >= end) {
                break;
            }
            int c = at(text, u, end);
            int k = CharClasses.of(cp(c));
            if ((k & CharClasses.UPPER) == 0) {
                break;
            }
            u += len(c);
            if ((k & CharClasses.LOWER) != 0) {
                lastLower = u;
            }
        }
        int l = run(text, u, end, CharClasses.LOWER);
        return l > u ? l : lastLower;
    }

    /** {@code U+L*} from {@code j}, or -1. */
    private static int upperMaybeLower(byte[] text, int j, int end) {
        int u = run(text, j, end, CharClasses.UPPER);
        return u > j ? run(text, u, end, CharClasses.LOWER) : -1;
    }

    private int suffixed(byte[] text, int e, int end) {
        if (wordSuffix && e < end && text[e] == '\'') {
            int s = contraction(text, e, end, FOLD);
            if (s > 0) {
                return s;
            }
        }
        return e;
    }

    private int number(byte[] text, int i, int end, int at, int cls) {
        int after = i + len(at);
        if ((cls & CharClasses.NUMBER) == 0) {
            if (numbers == SPACE_RUN && cp(at) == ' ' && after < end) {
                int c = at(text, after, end);
                if ((CharClasses.of(cp(c)) & CharClasses.NUMBER) != 0) {
                    return run(text, after + len(c), end, CharClasses.NUMBER);
                }
            }
            return -1;
        }
        switch (numbers) {
            case ONE:
                return after;
            case UP_TO_THREE:
                return twoMoreNumbers(text, after, end);
            default:
                return run(text, after, end, CharClasses.NUMBER);
        }
    }

    private int other(byte[] text, int i, int end, int cp, int cls) {
        int j = i;
        if (cp == ' ') {
            if (i + 1 >= end || !isOther(CharClasses.of(cp(at(text, i + 1, end))))) {
                return -1;
            }
            j = i + 1;
        } else if (!isOther(cls)) {
            return -1;
        }
        int e = j;
        while (true) {
            if (end - e >= 8) {
                // Eight ASCII bytes at a time; an ASCII blank, letter or digit ends the run.
                long v = (long) LONGS.get(text, e);
                long stop =
                        (v | ascii(v, CharClasses.SPACE | CharClasses.LETTER | CharClasses.NUMBER))
                                & HIGH_BITS;
                if (stop == 0) {
                    e += 8;
                    continue;
                }
                e += Long.numberOfTrailingZeros(stop) >>> 3;
                if (text[e] >= 0) {
                    break;
                }
            } else if (e >= end) {
                break;
            }
            int c = at(text, e, end);
            if (!isOther(CharClasses.of(cp(c)))) {
                break;
            }
            e += len(c);
        }
        if (otherTail != NO_TAIL) {
            while (e < end) {
                byte c = text[e];
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
    private static int run(byte[] text, int j, int end, int classes) {
        while (true) {
            if (end - j >= 8) {
                // Eight ASCII bytes at a time; an ASCII byte outside the classes ends the run.
                long v = (long) LONGS.get(text, j);
                long stop = (v | ~ascii(v, classes)) & HIGH_BITS;
                if (stop == 0) {
                    j += 8;
                    continue;
                }
                j += Long.numberOfTrailingZeros(stop) >>> 3;
                if (text[j] >= 0) {
                    return j;
                }
            } else if (j >= end) {
                return j;
            }
            // A non-ASCII byte, or one of the last seven: the code point decides.
            int c = at(text, j, end);
            if ((CharClasses.of(cp(c)) & classes) == 0) {
                return j;
            }
            j += len(c);
        }
    }

    private static final VarHandle LONGS =
            MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);
    private static final long LOW_BYTES = 0x0101010101010101L;
    private static final long HIGH_BITS = 0x8080808080808080L;

    /**
     * Of eight bytes, which are in {@code classes} if ASCII: a flag in each byte's high bit. Bytes
     * with the high bit set are non-ASCII and get whatever the arithmetic yields; the caller
     * excludes them. Marks do not occur in ASCII.
     */
    private static long ascii(long v, int classes) {
        long a = v & ~HIGH_BITS;
        long in = 0;
        if ((classes & CharClasses.LETTER) != 0) {
            long folded = a | 0x2020202020202020L;
            in |= atLeast(folded, 'a') & ~atLeast(folded, 'z' + 1);
        }
        if ((classes & CharClasses.UPPER) != 0) {
            in |= atLeast(a, 'A') & ~atLeast(a, 'Z' + 1);
        }
        if ((classes & CharClasses.LOWER) != 0) {
            in |= atLeast(a, 'a') & ~atLeast(a, 'z' + 1);
        }
        if ((classes & CharClasses.NUMBER) != 0) {
            in |= atLeast(a, '0') & ~atLeast(a, '9' + 1);
        }
        if ((classes & CharClasses.SPACE) != 0) {
            in |= is(a, ' ') | (atLeast(a, 0x9) & ~atLeast(a, 0xE));
        }
        if ((classes & CharClasses.NEWLINE) != 0) {
            in |= is(a, '\n') | is(a, '\r');
        }
        return in & HIGH_BITS;
    }

    /** Per byte of {@code a} (all under 0x80): the high bit when the byte is at least {@code c}. */
    private static long atLeast(long a, int c) {
        return (a + (0x80 - c) * LOW_BYTES) & HIGH_BITS;
    }

    /**
     * Per byte of {@code a} (all under 0x80): the high bit when the byte equals {@code c}. Exact in
     * every byte: no borrow crosses a byte, unlike the subtract-and-mask trick.
     */
    private static long is(long a, int c) {
        long x = a ^ (c * LOW_BYTES);
        return ~((x + ~HIGH_BITS) | x) & HIGH_BITS;
    }

    // A decoded position packs the code point in the low 21 bits and its byte length above them.

    private static int cp(int at) {
        return at & 0x1FFFFF;
    }

    private static int len(int at) {
        return at >>> 24;
    }

    /**
     * The code point at {@code i}. Any ill-formed byte, which {@link #utf8} never writes, stands
     * for U+FFFD and is one byte long.
     */
    private static int at(byte[] text, int i, int end) {
        int b = text[i];
        if (b >= 0) {
            return (1 << 24) | b;
        }
        if ((b & 0xE0) == 0xC0 && i + 1 < end) {
            return (2 << 24) | ((b & 0x1F) << 6) | (text[i + 1] & 0x3F);
        }
        if ((b & 0xF0) == 0xE0 && i + 2 < end) {
            return (3 << 24)
                    | ((b & 0x0F) << 12)
                    | ((text[i + 1] & 0x3F) << 6)
                    | (text[i + 2] & 0x3F);
        }
        if ((b & 0xF8) == 0xF0 && i + 3 < end) {
            return (4 << 24)
                    | ((b & 0x07) << 18)
                    | ((text[i + 1] & 0x3F) << 12)
                    | ((text[i + 2] & 0x3F) << 6)
                    | (text[i + 3] & 0x3F);
        }
        return (1 << 24) | 0xFFFD;
    }
}
