package com.qxotic.toknroll.vector;

import com.qxotic.toknroll.spi.CharClasses;
import com.qxotic.toknroll.spi.Kernels;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * The tiktoken family of pre-tokenization regexes as bitstream programs, 64 bytes per step.
 *
 * <p>A block of text becomes one bitmap per character class, a multi-byte character setting its
 * class on all of its bytes, and every rule of a regex becomes boolean algebra and
 * carry-propagating adds over those bitmaps: no branch per piece, no branch per character. What
 * reaches across a block edge is carried in a few scalars; the two questions that reach further (a
 * letter run's case, a whitespace run's last newline) are answered by looking ahead by hand, which
 * is rare. The result is a bitmap of piece starts, popped one trailing-zero count at a time.
 *
 * <p>Three grammars: r50k (GPT-2's byte level regex), the cl100k family (Llama 3, Qwen2) and the
 * o200k family (gpt-4o, gpt-oss, Tekken). After Hugging Face's {@code bitcannon}, rule for rule;
 * the doc comments there say why each rule is stated the way it is. Exactness is proven as the
 * scalar scanner's is: against the regex, over a fuzz corpus and all of enwik8.
 */
final class BitSplit {
    private static final VectorSpecies<Byte> BYTES = ByteVector.SPECIES_512;

    /** Blocks of starts kept before they are popped: a piece boundary may move this far back. */
    private static final int RING = 64;

    // Contraction spellings: the lowercase letters, either ASCII case, or Unicode case folding,
    // which also admits the long s, U+017F.
    private static final int LOWER = 0;
    private static final int ASCII_BOTH = 1;
    private static final int FOLD = 2;

    private final int rules;

    private BitSplit(int rules) {
        this.rules = rules;
    }

    /** The engine for one of the rule sets of {@link Kernels#split}, or null. */
    static BitSplit forRules(int rules) {
        return rules >= Kernels.CL100K && rules <= Kernels.TEKKEN ? new BitSplit(rules) : null;
    }

    // ---- one block's class streams ----

    /** One 64-byte block's bitmaps, bit i for byte i of the block. */
    private static final class Block {
        long lead; // first byte of a character
        long cont; // continuation byte
        long lu; // \p{Lu} \p{Lt}
        long ll; // \p{Ll}
        long lc; // \p{Lm} \p{Lo}: caseless, in both of o200k's case classes
        long mark; // \p{M}
        long letters; // \p{L}
        long word; // [\p{L}\p{M}], o200k's letter
        long n; // \p{N}
        long nl; // [\r\n]
        long sp; // ' '
        long ws; // \s
        long oth; // [^\s\p{L}\p{M}\p{N}], the apostrophe included
        long other; // [^\s\p{L}\p{N}]
        long apo; // '
        long slash; // /

        void clear() {
            lead = cont = lu = ll = lc = mark = letters = word = 0;
            n = nl = sp = ws = oth = other = apo = slash = 0;
        }

        /**
         * The streams the rules read one position later ("what held just before here"), carrying in
         * the previous block's last bit. The rest are left alone.
         */
        void back(Block c, Block p) {
            letters = (c.letters << 1) | (p.letters >>> 63);
            n = (c.n << 1) | (p.n >>> 63);
            nl = (c.nl << 1) | (p.nl >>> 63);
            sp = (c.sp << 1) | (p.sp >>> 63);
            ws = (c.ws << 1) | (p.ws >>> 63);
            other = (c.other << 1) | (p.other >>> 63);
        }

        /**
         * The streams the rules read one position earlier ("what holds just after here"), carrying
         * in the next block's first bit. The rest are left alone.
         */
        void forward(Block c, Block x) {
            lead = (c.lead >>> 1) | (x.lead << 63);
            ws = (c.ws >>> 1) | (x.ws << 63);
        }
    }

    // A decoded character's class, beside CharClasses' bits: the literals the rules read.
    private static final int SPACE_CHAR = 1 << 8;
    private static final int APOSTROPHE = 1 << 9;
    private static final int SLASH = 1 << 10;

    /**
     * Classifies {@code text[base, base + len)} into {@code out}: the ASCII bytes 64 at a time,
     * every other character decoded once and its class set on all of its bytes. Characters are what
     * the scanner in core decodes from the piece boundary on, so an ill-formed byte is a character
     * of its own and a lead claims the bytes its length says, whatever they are. A character cut by
     * the block edge continues into the next block through {@code seed}: its class in the low half,
     * the bytes still owed above it. Returns the seed for the block after.
     */
    private static int classify(byte[] text, int base, int len, int textEnd, int seed, Block out) {
        ByteVector v;
        if (base + 64 <= text.length) {
            v = ByteVector.fromArray(BYTES, text, base);
        } else {
            v = ByteVector.fromArray(BYTES, text, base, BYTES.indexInRange(base, text.length));
        }
        long inside = len == 64 ? -1L : (1L << len) - 1;
        long high = v.compare(VectorOperators.LT, (byte) 0).toLong() & inside;
        long upper =
                v.compare(VectorOperators.GE, (byte) 'A')
                        .and(v.compare(VectorOperators.LE, (byte) 'Z'))
                        .toLong();
        long lower =
                v.compare(VectorOperators.GE, (byte) 'a')
                        .and(v.compare(VectorOperators.LE, (byte) 'z'))
                        .toLong();
        long digits =
                v.compare(VectorOperators.GE, (byte) '0')
                        .and(v.compare(VectorOperators.LE, (byte) '9'))
                        .toLong();
        long nl =
                v.compare(VectorOperators.EQ, (byte) '\n')
                        .or(v.compare(VectorOperators.EQ, (byte) '\r'))
                        .toLong();
        long sp = v.compare(VectorOperators.EQ, (byte) ' ').toLong();
        long blanks =
                v.compare(VectorOperators.GE, (byte) 0x9)
                        .and(v.compare(VectorOperators.LE, (byte) 0xD))
                        .toLong();
        long apo = v.compare(VectorOperators.EQ, (byte) '\'').toLong();
        long slash = v.compare(VectorOperators.EQ, (byte) '/').toLong();
        long ascii = ~high & inside;
        out.cont = 0;
        out.lu = upper & ascii;
        out.ll = lower & ascii;
        out.lc = 0;
        out.mark = 0;
        out.n = digits & ascii;
        out.nl = nl & ascii;
        out.sp = sp & ascii;
        out.ws = (sp | blanks) & ascii;
        out.apo = apo & ascii;
        out.slash = slash & ascii;
        out.oth = ascii & ~(out.lu | out.ll | out.n | out.ws);
        int owed = seed >>> 16;
        if ((high | owed) != 0) {
            long todo = high;
            int cls = seed & 0xFFFF;
            if (owed > 0) {
                // the rest of a character the previous block began
                int count = Math.min(owed, len);
                long covered = count == 64 ? -1L : (1L << count) - 1;
                cover(out, cls, covered, covered);
                todo &= ~covered;
                owed -= count;
            }
            while (todo != 0) {
                int p = Long.numberOfTrailingZeros(todo);
                int at = at(text, base + p, textEnd);
                int cp = at & 0x1FFFFF;
                cls = CharClasses.of(cp);
                if (cp == ' ') {
                    cls |= SPACE_CHAR;
                } else if (cp == '\'') {
                    cls |= APOSTROPHE;
                } else if (cp == '/') {
                    cls |= SLASH;
                }
                int length = at >>> 24;
                int count = Math.min(length, len - p);
                long covered = ((1L << count) - 1) << p;
                cover(out, cls, covered, covered & ~(1L << p));
                todo &= ~covered;
                owed = length - count;
            }
            seed = cls | (owed << 16);
        } else {
            seed = 0;
        }
        out.lead = inside & ~out.cont;
        out.letters = out.lu | out.ll | out.lc;
        out.word = out.letters | out.mark;
        out.other = out.oth | out.mark;
        return seed;
    }

    /**
     * Gives the bytes {@code covered} the class {@code cls}, {@code cont} of them continuations.
     */
    private static void cover(Block out, int cls, long covered, long cont) {
        long keep = ~covered;
        out.lu &= keep;
        out.ll &= keep;
        out.lc &= keep;
        out.mark &= keep;
        out.n &= keep;
        out.nl &= keep;
        out.sp &= keep;
        out.ws &= keep;
        out.oth &= keep;
        out.apo &= keep;
        out.slash &= keep;
        out.cont |= cont;
        if ((cls & CharClasses.LETTER) != 0) {
            boolean upper = (cls & CharClasses.UPPER) != 0;
            boolean lower = (cls & CharClasses.LOWER) != 0;
            if (upper && !lower) {
                out.lu |= covered;
            } else if (lower && !upper) {
                out.ll |= covered;
            } else {
                out.lc |= covered;
            }
        } else if ((cls & CharClasses.MARK) != 0) {
            out.mark |= covered;
        } else if ((cls & CharClasses.NUMBER) != 0) {
            out.n |= covered;
        } else if ((cls & CharClasses.SPACE) != 0) {
            out.ws |= covered;
            if ((cls & CharClasses.NEWLINE) != 0) {
                out.nl |= covered;
            } else if ((cls & SPACE_CHAR) != 0) {
                out.sp |= covered;
            }
        } else {
            out.oth |= covered;
            if ((cls & APOSTROPHE) != 0) {
                out.apo |= covered;
            } else if ((cls & SLASH) != 0) {
                out.slash |= covered;
            }
        }
    }

    /** The code point at {@code i} in the low 21 bits and its byte length above them. */
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

    // ---- bitstream primitives ----

    /** Markers of {@code m} moved past the run of {@code c} each sits in, landing excluded. */
    private static long scanthru(long m, long c) {
        return (m + c) & ~c;
    }

    /**
     * Whether a run of {@code c} reaching bit 63 carried a marker of {@code m} out of the block.
     */
    private static boolean carries(long m, long c) {
        return Long.compareUnsigned(m + c, m) < 0;
    }

    /** Markers advanced by one character: one byte, then through the continuation bytes. */
    private static long adv(long m, long cont) {
        return scanthru(m << 1, cont);
    }

    /** From each marker of {@code m}, the span forward to the end of its run of {@code c}. */
    private static long fillToLast(long m, long c) {
        return m == 0 ? 0 : (scanthru(m, c) - m) | m;
    }

    /**
     * Backward: is there a marker of {@code m} at or after this position, in the same run of {@code
     * c}?
     */
    private static long laterInRun(long m, long c) {
        return Long.reverse(fillToLast(Long.reverse(m), Long.reverse(c)));
    }

    /** The run of {@code x} that starts at bit 0. */
    private static long leadRun(long x, long valid) {
        long z = ~x & valid;
        return z == 0 ? valid : (z & -z) - 1;
    }

    /** The run of {@code x} that ends at bit {@code len - 1}, or 0. */
    private static long trailRun(long x, long valid, int len) {
        if ((x & (1L << (len - 1))) == 0) {
            return 0;
        }
        long z = ~x & valid;
        return z == 0 ? valid : valid & ~((1L << (64 - Long.numberOfLeadingZeros(z))) - 1);
    }

    private static long bit(boolean b) {
        return b ? 1 : 0;
    }

    // ---- the programs ----

    /**
     * One split call's registers: the blocks in flight, the starts not yet popped, and the carries
     * every grammar has. A grammar is a subclass with its own carries and its rule.
     */
    private abstract static class Run {
        final byte[] text;
        final int start;
        final int end;
        Block pv = new Block();
        Block cur = new Block();
        Block nx = new Block();
        final Block bk = new Block();
        final Block fw = new Block();
        final long[] starts = new long[RING];
        int seed;

        // the block in hand
        int bi;
        int base;
        int len;
        long valid;
        long lead;
        long lb; // the last byte of its character
        long eof; // the text's final byte as a bit
        boolean prevWs; // the previous block's last byte is blank

        // carried across the block edge
        boolean digitsOpen; // a digit run continues into the next block
        int digitsSince; // digits of its current group consumed so far
        int retractable = -1; // a start a newline further on may retract, absolute, or -1
        long stealPatch; // a start of the previous block found by this one
        long contractionOpens; // bits of the next block a contraction ending in it opens
        long contractionInner; // bits of the next block inside a contraction

        Run(byte[] text, int start, int end) {
            this.text = text;
            this.start = start;
            this.end = end;
        }

        /** The starts of the block in hand; the driver keeps the leads and opens position 0. */
        abstract long rule();

        // ---- shared rules

        /**
         * Each bit of {@code x} moved back to the lead byte of its character, at most three steps;
         * a character cut by the block edge lands its lead in {@link #stealPatch}, for the previous
         * block.
         */
        final long toLead(long x, long cont, long prevCont) {
            long hi = x;
            long lo = 0;
            for (int i = 0; i < 3; i++) {
                long mHi = hi & cont;
                long mLo = lo & prevCont;
                if ((mHi | mLo) == 0) {
                    break;
                }
                hi = (hi & ~cont) | (mHi >>> 1);
                lo = (lo & ~prevCont) | (mLo >>> 1) | (mHi << 63);
            }
            stealPatch = lo;
            return hi;
        }

        /** `\p{N}{1,cap}`: a group boundary every cap characters from the run's start. */
        final long digitGroups(int cap) {
            long n = cur.n;
            long cont = cur.cont;
            boolean prevIsDigit = pv.n < 0;
            long m = n & lead & ~((n << 1) | bit(prevIsDigit));
            if (digitsOpen && prevIsDigit) {
                long s = lead & -lead & n; // the block's first lead, when a digit
                for (int i = (cap - digitsSince % cap) % cap; i > 0; i--) {
                    s = adv(s, cont) & n & lead;
                }
                m |= s;
            }
            long groups = m;
            if ((n & cont) == 0) {
                // every digit is one byte: cap characters on is a shift, against a mask asking
                // that the skipped positions were digits too
                long nk = n;
                for (int i = 1; i < cap; i++) {
                    nk &= n << i;
                }
                while (m != 0) {
                    m = (m << cap) & nk;
                    groups |= m;
                }
            } else {
                while (m != 0) {
                    long e = m;
                    for (int i = 0; i < cap; i++) {
                        e = adv(e, cont) & n & lead;
                    }
                    if (e == 0) {
                        break;
                    }
                    groups |= e;
                    m = e;
                }
            }
            return groups;
        }

        /** Carries the digit run into the next block. */
        final void commitDigits(int cap, long groups) {
            long tn = trailRun(cur.n, valid, len);
            digitsOpen = tn != 0 && (nx.n & 1) != 0;
            if (!digitsOpen) {
                digitsSince = 0;
                return;
            }
            long g = groups & tn;
            int counted;
            if (g == 0) {
                counted = digitsSince + Long.bitCount(cur.n & lead & tn);
            } else {
                long fromLastGroup = ~((1L << (63 - Long.numberOfLeadingZeros(g))) - 1);
                counted = Long.bitCount(cur.n & lead & tn & fromLastGroup);
            }
            digitsSince = counted % cap;
        }

        /**
         * `\s*[\r\n]+` versus `\s+(?!\S)`, the one backward rule: the first runs through a
         * whitespace run's last newline, so a piece opens right after it, but a newline arriving in
         * a later block moves which newline is last. Top of a block: a newline inside the run that
         * was open at the edge retracts the start committed for it.
         */
        final void retract() {
            if (retractable >= 0
                    && prevWs
                    && (cur.ws & 1) != 0
                    && (cur.nl & leadRun(cur.ws, valid)) != 0) {
                int p = retractable - start;
                starts[(p >>> 6) & (RING - 1)] &= ~(1L << (p & 63));
                retractable = -1;
            }
        }

        /**
         * Bottom of a block: remembers the start a later newline could still retract. {@code
         * absorbed} is what an other-run's tail claimed; a start that tail made cannot be reached
         * back past, since the absorption cut the whitespace run.
         */
        final void commitRetractable(long afterNl, long absorbed) {
            long tws = trailRun(cur.ws, valid, len);
            if (tws != 0 && (nx.ws & 1) != 0) {
                long a = afterNl & tws & ~absorbed;
                if (a != 0) {
                    retractable = base + 63 - Long.numberOfLeadingZeros(a);
                } else if (!((tws & 1) != 0 && prevWs)) {
                    retractable = -1; // a fresh run with no newline yet
                }
            } else {
                retractable = -1;
            }
        }

        /**
         * The contraction alternative: `'s` is a piece of its own, so nothing inside it opens one
         * and the character after it does. Apostrophes are rare, so the literal is read at the few
         * marked positions rather than as a stream per letter. Chaining (`'re've`) needs no loop:
         * the character after a contraction is an apostrophe whose predecessor is a letter, a run
         * start already.
         */
        final long contractions(long st, int mode) {
            long opens = contractionOpens;
            long inner = contractionInner;
            contractionOpens = 0;
            contractionInner = 0;
            long a = st & cur.apo;
            while (a != 0) {
                int j = Long.numberOfTrailingZeros(a);
                a &= a - 1;
                int p = base + j;
                int k = contractionLength(text, p, end, mode);
                if (k == 0) {
                    continue;
                }
                int hi = j + k; // one past the literal
                if (j + 1 < 64) {
                    long up = hi >= 64 ? -1L : (1L << hi) - 1;
                    inner |= up & ~((1L << (j + 1)) - 1);
                }
                if (hi > 64) {
                    contractionInner |= (1L << (hi - 64)) - 1;
                }
                if (p + k < end) {
                    if (hi < 64) {
                        opens |= 1L << hi;
                    } else {
                        contractionOpens |= 1L << (hi - 64);
                    }
                }
            }
            return (st & ~inner) | opens;
        }
    }

    /** The byte length of the contraction at {@code i}, or 0: the letters decoded as characters. */
    private static int contractionLength(byte[] text, int i, int end, int mode) {
        int p1 = i + 1;
        if (p1 >= end) {
            return 0;
        }
        int a1 = at(text, p1, end);
        int c1 = a1 & 0x1FFFFF;
        if (is(c1, 's', mode) || is(c1, 't', mode) || is(c1, 'm', mode) || is(c1, 'd', mode)) {
            return p1 + (a1 >>> 24) - i;
        }
        int p2 = p1 + (a1 >>> 24);
        if (p2 >= end) {
            return 0;
        }
        int a2 = at(text, p2, end);
        int c2 = a2 & 0x1FFFFF;
        if ((is(c1, 'r', mode) && is(c2, 'e', mode))
                || (is(c1, 'v', mode) && is(c2, 'e', mode))
                || (is(c1, 'l', mode) && is(c2, 'l', mode))) {
            return p2 + (a2 >>> 24) - i;
        }
        return 0;
    }

    /** Whether {@code c} spells {@code lower} in a contraction under {@code mode}. */
    private static boolean is(int c, char lower, int mode) {
        if (c == lower) {
            return true;
        }
        switch (mode) {
            case ASCII_BOTH:
                return c == lower - ('a' - 'A');
            case FOLD:
                return c == lower - ('a' - 'A') || (c == 0x17F && lower == 's');
            default:
                return false;
        }
    }

    /**
     * r50k, GPT-2's byte level: {@code 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+|
     * ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+}. Every alternative but the contraction is ` ?X+` over a
     * class run, so a piece opens at each run start, pushed back one character when a space sits in
     * front of it. No newline rule: a newline is ordinary whitespace, and may be the stolen
     * character.
     */
    private static final class R50k extends Run {
        R50k(byte[] text, int start, int end) {
            super(text, start, end);
        }

        @Override
        long rule() {
            // `\s+(?!\S)` hands the run's last blank to what follows, as a ` ?` prefix if it is a
            // space, else as a piece of its own: either way it opens one, unless it ends the text
            long steal = toLead(cur.ws & lb & ~eof & ~fw.ws, cur.cont, pv.cont);
            long st =
                    (cur.letters & ~bk.letters & ~bk.sp)
                            | (cur.n & ~bk.n & ~bk.sp)
                            | (cur.other & ~bk.other & ~bk.sp)
                            | (cur.ws & ~bk.ws)
                            | steal;
            st &= lead;
            if (bi == 0) {
                st |= 1;
            }
            return contractions(st, LOWER);
        }
    }

    /**
     * The cl100k family: {@code (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
     * ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+}, with one digit at a time for Qwen2.
     */
    private static final class Cl100k extends Run {
        private final int digitCap;
        private final int contraction;

        // carried across the block edge
        boolean nlRun; // an other-run's [\r\n]* tail is still open
        boolean prevOsf; // the last byte belonged to a piece-opening other char

        Cl100k(byte[] text, int start, int end, int digitCap, int contraction) {
            super(text, start, end);
            this.digitCap = digitCap;
            this.contraction = contraction;
        }

        @Override
        long rule() {
            // ` ?[^\s\p{L}\p{N}]+` opens where an other char follows neither an other nor a space
            long oStart = cur.other & lead & ~bk.other & ~bk.sp;
            long wsStart = cur.ws & lead & ~bk.ws;
            // `\s+(?!\S)`: the run's last char opens a piece, unless it is a newline, which
            // `\s*[\r\n]+` has already swallowed
            long steal = toLead(cur.ws & ~cur.nl & lb & ~eof & ~fw.ws, cur.cont, pv.cont);
            // `\s*[\r\n]+` runs through the run's last newline: a piece opens right after it,
            // unless a further newline still follows inside the run
            long afterNl = bk.nl & cur.ws & lead & ~laterInRun(cur.nl, cur.ws);

            // `[^\r\n\p{L}\p{N}]?\p{L}+`: a letter run right after a piece-opening other char is
            // one piece with it; osf smears each such start over its character's bytes, so "my
            // predecessor opened a piece" is a plain shift
            long carried = bit(prevOsf);
            long osf = fillToLast(((oStart << 1) | carried) & cur.cont, cur.cont) | oStart;
            long lStart =
                    cur.letters & lead & ~bk.letters & ~(bk.ws & ~bk.nl) & ~((osf << 1) | carried);

            // `\p{N}{1,cap}`
            long groups = digitGroups(digitCap);

            // the other-run's `[\r\n]*` tail swallows the newlines right behind it
            long nlM = (bk.other & cur.nl & lead) | bit(nlRun);
            long nlE = scanthru(nlM, cur.nl);
            boolean nlCarries = carries(nlM, cur.nl);
            long nlSpan = nlE - nlM;

            retract();

            long st = groups | lStart | oStart | wsStart | afterNl | steal;
            st &= ~nlSpan;
            st |= nlE;

            // ---- carries
            nlRun = nlCarries;
            commitDigits(digitCap, groups);
            commitRetractable(afterNl, nlE);
            prevOsf = ((osf >>> (len - 1)) & 1) != 0;

            st &= lead;
            if (bi == 0) {
                st |= 1;
            }
            return contractions(st, contraction);
        }
    }

    /**
     * The o200k family: {@code [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}
     * \p{Lo}\p{M}]+(?i:'s|...)?|} the same with {@code +} and {@code *} swapped {@code |\p{N}{1,3}|
     * ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+}. Tekken drops the contraction suffix and
     * takes one digit at a time. cl100k's skeleton; the letter half differs.
     */
    private static final class O200k extends Run {
        private final boolean suffix;
        private final int digitCap;

        // carried across the block edge
        boolean nlRun;
        boolean othEdge; // rule 4's effective oth at the last byte
        boolean letterEdge; // and the adjusted letter there
        boolean markOthOpen; // a mark stretch rule 4 took was still running at the edge
        boolean prevOsf;
        boolean prevAbsorbed; // the last byte was eaten by a [\r\n/]* tail
        boolean lcOpen; // a lowercase run was still open at the edge
        long sfxCarry; // bytes of a contraction suffix that spilled past the edge
        long force; // and the start it opens just past itself, if that landed past the edge
        int sfxEnd = -1; // just past the last suffix: the apostrophe there is a prefix, not a

        // second suffix, since (?i:...)? applies once

        O200k(byte[] text, int start, int end, boolean suffix, int digitCap) {
            super(text, start, end);
            this.suffix = suffix;
            this.digitCap = digitCap;
        }

        @Override
        long rule() {
            long cont = cur.cont;
            // Rule 4's `[\r\n/]*` tail: `/` is in both the `+` body and the tail, so one tail
            // run can collect several markers, which is what fillToLast is for.
            long tailCls = cur.nl | cur.slash;

            // ` ?[^\s\p{L}\p{N}]+[\r\n/]*`. oth is a fixpoint: a mark is in both the letter
            // classes and rule 4's, so a mark rule 4 keeps joins its run, which can suppress a
            // later oStart and demote further marks. Monotone and bounded by the marks in the
            // block; with no marks it settles on the first pass.
            long oth = cur.oth;
            long oStart;
            long osf;
            long markOth;
            long nlE;
            long nlSpan;
            boolean nlCarries;
            long prevOsfBit = bit(prevOsf);
            long edge = bit(othEdge && !prevAbsorbed);
            while (true) {
                long bkOth = (oth << 1) | bit(othEdge);
                // the carried marker is not masked with the tail class: if bit 0 is not in it,
                // the tail landed there and that position opens a piece
                long nlM = (bkOth & cur.nl & lead) | bit(nlRun);
                nlE = scanthru(nlM, tailCls);
                nlCarries = carries(nlM, tailCls);
                nlSpan = fillToLast(nlM & tailCls, tailCls);
                // a char absorbed by a tail is not part of the `+` body, so an other after it
                // opens a fresh run
                long oPrev = oth & ~nlSpan;
                oStart = oth & lead & ~nlSpan & ~((oPrev << 1) | edge) & ~bk.sp;
                // `[^\r\n\p{L}\p{N}]?` before a letter run, smeared across its character
                osf = fillToLast(((oStart << 1) | prevOsfBit) & cont, cont) | oStart;
                // Which arm claims a mark? Only one inside rule 4's `+` body is contested, and
                // the letter arm takes at most one char as its prefix, so it can only claim a
                // stretch right after the run's first char: `!́a` is one piece, `!!́a` is `!!́`
                // then `a`.
                long contested = fillToLast(((oPrev << 1) | edge) & cur.mark & lead, cur.mark);
                long claimed = fillToLast(((osf << 1) | prevOsfBit) & cur.mark & lead, cur.mark);
                markOth = contested & ~claimed;
                if (markOthOpen) {
                    // a stretch rule 4 already took, still running at the edge: its lead is in
                    // the previous block, out of contested's sight
                    markOth |= leadRun(cur.mark, valid);
                }
                long grown = cur.oth | markOth;
                if (grown == oth) {
                    break;
                }
                oth = grown;
            }
            long letter = cur.word & ~markOth;
            long bkLetter = (letter << 1) | bit(letterEdge);

            long wsStart = cur.ws & lead & ~bk.ws;
            long steal = toLead(cur.ws & ~cur.nl & lb & ~eof & ~fw.ws, cont, pv.cont);
            long afterNl = bk.nl & cur.ws & lead & ~laterInRun(cur.nl, cur.ws);

            long lStart =
                    letter & lead & ~bkLetter & ~(bk.ws & ~bk.nl) & ~((osf << 1) | prevOsfBit);

            // The contraction suffix `(?i:'s|...)?`, resolved first because its letters belong
            // to the piece before it.
            long sfx = sfxCarry;
            long sfxOpen = force;
            sfxCarry = 0;
            force = 0;
            if (suffix) {
                long a = cur.apo & lead & bkLetter;
                while (a != 0) {
                    int j = Long.numberOfTrailingZeros(a);
                    a &= a - 1;
                    int p = base + j;
                    if (p == sfxEnd) {
                        continue; // its predecessor is inside a suffix: `?` applies once
                    }
                    int k = contractionLength(text, p, end, FOLD);
                    if (k == 0) {
                        continue;
                    }
                    sfxEnd = p + k;
                    int hi = j + k;
                    long inb = hi >= 64 ? -1L : (1L << hi) - 1;
                    sfx |= inb & ~((1L << j) - 1);
                    if (hi > 64) {
                        sfxCarry |= (1L << (hi - 64)) - 1;
                    }
                    // the suffix ends its piece, so the next char opens one: nothing else would,
                    // since it sits mid-run
                    if (p + k < end) {
                        if (hi < 64) {
                            sfxOpen |= 1L << hi;
                        } else {
                            force |= 1L << (hi - 64);
                        }
                    }
                }
            }

            // The letter alternation `[UC]*[LC]+ | [UC]+[LC]*`: every piece is one maximal
            // uc-run followed by one maximal lc-run, so the phase inside a letter run flips at
            // its first strictly lowercase char, and an uppercase after one opens a new piece.
            // Every lowercase marker scanned through lc lands on exactly that uppercase.
            long lalt = letter & ~sfx;
            // the gate: a run splits only where an uppercase sits after a letter, so most text
            // skips the case algebra
            long caseGate = cur.lu & lead & bkLetter;
            long uc = lalt & ~cur.ll;
            long lc = lalt & ~cur.lu;
            long cm = (cur.lc | cur.mark) & lalt;
            long mm = (cur.ll & lalt) | bit(lcOpen);
            long f = scanthru(mm, lc);
            lcOpen = carries(mm, lc);
            long ltIn = f & lalt;
            // the one backtrack: the first arm needs a lowercase after `uc*`, so a uc-run that
            // ends the letter run cuts after its last caseless char (`ʰABC` is `ʰ`, `ABC`)
            long ltCut = 0;
            if (caseGate != 0) {
                long word = cur.word;
                long ucRaw = word & ~cur.ll;
                long la = laterInRun(cur.ll, word);
                long ca = laterInRun((cur.lc | cur.mark) & word, ucRaw);
                // a run leaving the block inherits the answer from what follows it
                if (word < 0 && laterLowercase()) {
                    la |= trailRun(word, valid, len);
                }
                if (ucRaw < 0 && laterCaseless()) {
                    ca |= trailRun(ucRaw, valid, len);
                }
                ltCut = uc & lead & ((cm << 1) | bit((pv.lc | pv.mark) < 0)) & ~ca & ~la;
            }

            long groups = digitGroups(digitCap);

            retract();

            long st = groups | lStart | ltIn | ltCut | oStart | wsStart | afterNl | steal;
            st &= ~nlSpan;
            st |= nlE;
            st &= ~sfx; // nothing inside a contraction suffix opens a piece
            st |= sfxOpen;

            // ---- carries
            nlRun = nlCarries;
            prevAbsorbed = ((nlSpan >>> (len - 1)) & 1) != 0;
            commitDigits(digitCap, groups);
            commitRetractable(afterNl, nlE);
            prevOsf = ((osf >>> (len - 1)) & 1) != 0;
            othEdge = ((oth >>> (len - 1)) & 1) != 0;
            letterEdge = ((letter >>> (len - 1)) & 1) != 0;
            markOthOpen = ((markOth >>> (len - 1)) & 1) != 0;
            return st;
        }

        /** Whether the letter run going on at the next block's first byte has a lowercase in it. */
        private boolean laterLowercase() {
            if ((nx.word & 1) == 0) {
                return false;
            }
            if ((laterInRun(nx.ll, nx.word) & 1) != 0) {
                return true;
            }
            int nxLen = Math.min(64, end - base - 64);
            long nxValid = nxLen == 64 ? -1L : (1L << nxLen) - 1;
            if (leadRun(nx.word, nxValid) != nxValid) {
                return false;
            }
            // the run leaves the next block too: look by hand
            for (int i = base + 64 + nxLen; i < end; ) {
                int at = at(text, i, end);
                int cls = CharClasses.of(at & 0x1FFFFF);
                if ((cls & (CharClasses.LETTER | CharClasses.MARK)) == 0) {
                    return false;
                }
                if ((cls & (CharClasses.LOWER | CharClasses.UPPER)) == CharClasses.LOWER) {
                    return true;
                }
                i += at >>> 24;
            }
            return false;
        }

        /** Whether the uc-run going on at the next block's first byte has a caseless char in it. */
        private boolean laterCaseless() {
            long uc = nx.word & ~nx.ll;
            if ((uc & 1) == 0) {
                return false;
            }
            if ((laterInRun((nx.lc | nx.mark) & nx.word, uc) & 1) != 0) {
                return true;
            }
            int nxLen = Math.min(64, end - base - 64);
            long nxValid = nxLen == 64 ? -1L : (1L << nxLen) - 1;
            if (leadRun(uc, nxValid) != nxValid) {
                return false;
            }
            for (int i = base + 64 + nxLen; i < end; ) {
                int at = at(text, i, end);
                int cls = CharClasses.of(at & 0x1FFFFF);
                if ((cls & (CharClasses.LETTER | CharClasses.MARK)) == 0) {
                    return false;
                }
                int cased = cls & (CharClasses.LOWER | CharClasses.UPPER);
                if (cased == CharClasses.LOWER) {
                    return false; // a lowercase ends the uc-run
                }
                if (cased == (CharClasses.LOWER | CharClasses.UPPER)) {
                    return true;
                }
                i += at >>> 24;
            }
            return false;
        }
    }

    private Run run(byte[] text, int start, int end) {
        switch (rules) {
            case Kernels.CL100K:
                return new Cl100k(text, start, end, 3, FOLD);
            case Kernels.CL100K_ASCII:
                return new Cl100k(text, start, end, 3, ASCII_BOTH);
            case Kernels.QWEN2:
                return new Cl100k(text, start, end, 1, FOLD);
            case Kernels.QWEN2_ASCII:
                return new Cl100k(text, start, end, 1, ASCII_BOTH);
            case Kernels.R50K:
                return new R50k(text, start, end);
            case Kernels.O200K:
                return new O200k(text, start, end, true, 3);
            case Kernels.TEKKEN:
                return new O200k(text, start, end, false, 1);
            default:
                throw new IllegalStateException("rules " + rules);
        }
    }

    // ---- driving the blocks and popping the pieces ----

    /** Whether the whitespace run open at {@code i} has a newline from there on. */
    private static boolean newlineInRun(byte[] text, int i, int end) {
        while (i < end) {
            int at = at(text, i, end);
            int cls = CharClasses.of(at & 0x1FFFFF);
            if ((cls & CharClasses.SPACE) == 0) {
                return false;
            }
            if ((cls & CharClasses.NEWLINE) != 0) {
                return true;
            }
            i += at >>> 24;
        }
        return false;
    }

    /**
     * Cuts pieces from {@code text[start, end)} into {@code ends}, each entry the end of one piece,
     * until the array is full or the text ends; returns how many. {@code start} must be a piece
     * boundary; nothing past {@code end} is read.
     */
    int split(byte[] text, int start, int end, int[] ends) {
        if (end <= start) {
            return 0;
        }
        Run run = run(text, start, end);
        int nblk = (end - start + 63) >>> 6;
        run.seed = classify(text, start, Math.min(64, end - start), end, 0, run.cur);
        int n = 0;
        int popped = 0; // blocks whose starts are out
        for (int bi = 0; bi < nblk; bi++) {
            int base = start + (bi << 6);
            int len = Math.min(64, end - base);
            boolean last = bi == nblk - 1;
            if (last) {
                run.nx.clear();
            } else {
                int next = base + 64;
                run.seed = classify(text, next, Math.min(64, end - next), end, run.seed, run.nx);
            }
            run.bk.back(run.cur, run.pv);
            run.fw.forward(run.cur, run.nx);
            run.bi = bi;
            run.base = base;
            run.len = len;
            run.valid = len == 64 ? -1L : (1L << len) - 1;
            run.lead = run.cur.lead;
            run.eof = last ? 1L << (len - 1) : 0;
            run.lb = (run.fw.lead & run.valid) | run.eof;
            run.prevWs = run.pv.ws < 0;
            long st = run.rule() & run.lead;
            if (bi == 0) {
                st |= 1;
            }
            run.starts[bi & (RING - 1)] = st;
            if (bi > 0) {
                run.starts[(bi - 1) & (RING - 1)] |= run.stealPatch;
            }
            // A block's starts are final once the next block has run, except one a newline
            // further on may still retract: nothing from it on is popped until it is decided.
            if (run.retractable >= 0 && bi + 2 - popped > RING) {
                // the whitespace run outgrew the ring: decide it by looking ahead by hand
                if (newlineInRun(text, base + len, end)) {
                    int p = run.retractable - start;
                    run.starts[(p >>> 6) & (RING - 1)] &= ~(1L << (p & 63));
                }
                run.retractable = -1;
            }
            int limit = last ? nblk : bi;
            if (run.retractable >= 0) {
                limit = Math.min(limit, (run.retractable - start) >>> 6);
            }
            for (; popped < limit; popped++) {
                long bits = run.starts[popped & (RING - 1)];
                int at = start + (popped << 6);
                while (bits != 0) {
                    int pos = at + Long.numberOfTrailingZeros(bits);
                    bits &= bits - 1;
                    if (pos == start) {
                        continue; // the piece that is open
                    }
                    if (n == ends.length) {
                        return n; // the caller comes back to the piece that is open
                    }
                    ends[n++] = pos;
                }
            }
            Block t = run.pv;
            run.pv = run.cur;
            run.cur = run.nx;
            run.nx = t;
        }
        if (n < ends.length) {
            ends[n++] = end;
        }
        return n;
    }
}
