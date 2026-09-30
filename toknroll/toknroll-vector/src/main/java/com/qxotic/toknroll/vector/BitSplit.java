package com.qxotic.toknroll.vector;

import com.qxotic.toknroll.spi.CharClasses;
import com.qxotic.toknroll.spi.Kernels;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * The cl100k family of pre-tokenization regexes as a bitstream program, 64 bytes per step.
 *
 * <p>The family is {@code (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
 * ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+} (cl100k, Llama 3) and the same with one digit
 * at a time (Qwen2). A block of text becomes one bitmap per character class, a multi-byte character
 * setting its class on all of its bytes, and every rule of the regex becomes boolean algebra and
 * carry-propagating adds over those bitmaps: no branch per piece, no branch per character. What
 * reaches across a block edge is carried in a few scalars. The result is a bitmap of piece starts,
 * popped one trailing-zero count at a time.
 *
 * <p>After Hugging Face's {@code bitcannon}, rule for rule; the doc comments there say why each
 * rule is stated the way it is. Exactness is proven as the scalar scanner's is: against the regex,
 * over a fuzz corpus and all of enwik8.
 */
final class BitSplit {
    private static final VectorSpecies<Byte> BYTES = ByteVector.SPECIES_512;

    /** Blocks of starts kept before they are popped: a piece boundary may move this far back. */
    private static final int RING = 64;

    // A decoded character's class, beside CharClasses' bits: the two literals the rules read.
    private static final int SPACE_CHAR = 1 << 8;
    private static final int APOSTROPHE = 1 << 9;

    /** {@code \p{N}{1,cap}}: 3 for cl100k and Llama 3, 1 for Qwen2. */
    private final int digitCap;

    /** Whether {@code (?i:'s)} admits the long s, U+017F, as Unicode case folding does. */
    private final boolean foldLongS;

    private BitSplit(int digitCap, boolean foldLongS) {
        this.digitCap = digitCap;
        this.foldLongS = foldLongS;
    }

    /** The engine for one of the rule sets of {@link Kernels#split}, or null. */
    static BitSplit forRules(int rules) {
        switch (rules) {
            case Kernels.CL100K:
                return new BitSplit(3, true);
            case Kernels.CL100K_ASCII:
                return new BitSplit(3, false);
            case Kernels.QWEN2:
                return new BitSplit(1, true);
            case Kernels.QWEN2_ASCII:
                return new BitSplit(1, false);
            default:
                return null;
        }
    }

    // ── one block's class streams ────────────────────────────────────────────────────────────

    /** One 64-byte block's bitmaps, bit i for byte i of the block. */
    private static final class Block {
        long lead; // first byte of a character
        long cont; // continuation byte
        long l; // \p{L}
        long n; // \p{N}
        long other; // [^\s\p{L}\p{N}], the apostrophe included
        long nl; // [\r\n]
        long sp; // ' '
        long ws; // \s
        long apo; // '

        void clear() {
            lead = cont = l = n = other = nl = sp = ws = apo = 0;
        }

        /**
         * Every stream of {@code cur} one position later, carrying in the previous block's last
         * bit.
         */
        void back(Block cur, Block prev) {
            lead = (cur.lead << 1) | (prev.lead >>> 63);
            cont = (cur.cont << 1) | (prev.cont >>> 63);
            l = (cur.l << 1) | (prev.l >>> 63);
            n = (cur.n << 1) | (prev.n >>> 63);
            other = (cur.other << 1) | (prev.other >>> 63);
            nl = (cur.nl << 1) | (prev.nl >>> 63);
            sp = (cur.sp << 1) | (prev.sp >>> 63);
            ws = (cur.ws << 1) | (prev.ws >>> 63);
            apo = (cur.apo << 1) | (prev.apo >>> 63);
        }

        /**
         * Every stream of {@code cur} one position earlier, carrying in the next block's first bit.
         */
        void forward(Block cur, Block next) {
            lead = (cur.lead >>> 1) | (next.lead << 63);
            cont = (cur.cont >>> 1) | (next.cont << 63);
            l = (cur.l >>> 1) | (next.l << 63);
            n = (cur.n >>> 1) | (next.n << 63);
            other = (cur.other >>> 1) | (next.other << 63);
            nl = (cur.nl >>> 1) | (next.nl << 63);
            sp = (cur.sp >>> 1) | (next.sp << 63);
            ws = (cur.ws >>> 1) | (next.ws << 63);
            apo = (cur.apo >>> 1) | (next.apo << 63);
        }
    }

    /**
     * Classifies {@code text[base, base + len)} into {@code out}: the ASCII bytes 64 at a time,
     * every other character decoded once and its class set on all of its bytes. Characters are what
     * the scanner in core decodes from the piece boundary on, so an ill-formed byte is a character
     * of its own and a lead claims the bytes its length says, whatever they are. A character cut by
     * the block edge continues into the next block through {@code seed}: its class in the low byte,
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
        ByteVector folded = v.or((byte) 0x20);
        long letters =
                folded.compare(VectorOperators.GE, (byte) 'a')
                        .and(folded.compare(VectorOperators.LE, (byte) 'z'))
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
        long ascii = ~high & inside;
        out.cont = 0;
        out.l = letters & ascii;
        out.n = digits & ascii;
        out.nl = nl & ascii;
        out.sp = sp & ascii;
        out.ws = (sp | blanks) & ascii;
        out.apo = apo & ascii;
        out.other = ascii & ~(out.l | out.n | out.ws);
        int owed = seed >>> 16;
        if ((high | owed) == 0) {
            out.lead = inside;
            return 0;
        }
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
            cls = CharClasses.of(cp) | (cp == ' ' ? SPACE_CHAR : cp == '\'' ? APOSTROPHE : 0);
            int length = at >>> 24;
            int count = Math.min(length, len - p);
            long covered = ((1L << count) - 1) << p;
            cover(out, cls, covered, covered & ~(1L << p));
            todo &= ~covered;
            owed = length - count;
        }
        out.lead = inside & ~out.cont;
        return cls | (owed << 16);
    }

    /**
     * Gives the bytes {@code covered} the class {@code cls}, with {@code cont} of them
     * continuations.
     */
    private static void cover(Block out, int cls, long covered, long cont) {
        long keep = ~covered;
        out.l &= keep;
        out.n &= keep;
        out.nl &= keep;
        out.sp &= keep;
        out.ws &= keep;
        out.apo &= keep;
        out.other &= keep;
        out.cont |= cont;
        if ((cls & CharClasses.LETTER) != 0) {
            out.l |= covered;
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
            out.other |= covered;
            if ((cls & APOSTROPHE) != 0) {
                out.apo |= covered;
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

    // ── bitstream primitives ─────────────────────────────────────────────────────────────────

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

    // ── the program ──────────────────────────────────────────────────────────────────────────

    /** One split call's registers: the blocks in flight, the carries, the starts not yet popped. */
    private final class Run {
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

        // carried across the block edge
        boolean nlRun; // an other-run's [\r\n]* tail is still open
        boolean prevOsf; // the last byte belonged to a piece-opening other char
        boolean digitsOpen; // a digit run continues into the next block
        int digitsSince; // digits of its current group consumed so far
        int retractable = -1; // a start a newline further on may retract, absolute, or -1
        long contractionOpens; // bits of the next block a contraction ending in it opens
        long contractionInner; // bits of the next block inside a contraction
        long stealPatch; // a start of the previous block found by this one

        Run(byte[] text, int start, int end) {
            this.text = text;
            this.start = start;
            this.end = end;
        }

        /**
         * The starts of block {@code bi}, which begins at {@code base} and has {@code len} bytes.
         */
        long rule(int bi, int base, int len, boolean last) {
            Block pv = this.pv;
            Block cur = this.cur;
            Block bk = this.bk;
            Block fw = this.fw;
            long valid = len == 64 ? -1L : (1L << len) - 1;
            long eof = last ? 1L << (len - 1) : 0;
            long lead = cur.lead;
            long lb = (fw.lead & valid) | eof; // the last byte of its character
            boolean prevWs = pv.ws < 0;

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
            long carried = prevOsf ? 1 : 0;
            long osf = fillToLast(((oStart << 1) | carried) & cur.cont, cur.cont) | oStart;
            long lStart = cur.l & lead & ~bk.l & ~(bk.ws & ~bk.nl) & ~((osf << 1) | carried);

            // `\p{N}{1,cap}`
            boolean prevIsDigit = pv.n < 0;
            long groups = digitGroups(cur.n, lead, cur.cont, prevIsDigit);

            // the other-run's `[\r\n]*` tail swallows the newlines right behind it
            long nlM = (bk.other & cur.nl & lead) | (nlRun ? 1 : 0);
            long nlE = scanthru(nlM, cur.nl);
            boolean nlCarries = carries(nlM, cur.nl);
            long nlSpan = nlE - nlM;

            // a newline arriving inside the whitespace run that was open at the edge: the start
            // committed after what was that run's last newline is no longer one
            if (retractable >= 0
                    && prevWs
                    && (cur.ws & 1) != 0
                    && (cur.nl & leadRun(cur.ws, valid)) != 0) {
                int p = retractable - start;
                starts[(p >>> 6) & (RING - 1)] &= ~(1L << (p & 63));
                retractable = -1;
            }

            long st = groups | lStart | oStart | wsStart | afterNl | steal;
            st &= ~nlSpan;
            st |= nlE;

            // ── carries
            nlRun = nlCarries;
            long tn = trailRun(cur.n, valid, len);
            digitsOpen = tn != 0 && (nx.n & 1) != 0;
            digitsSince = digitsOpen ? digitsSince(groups, tn, cur.n, lead) : 0;
            long tws = trailRun(cur.ws, valid, len);
            if (tws != 0 && (nx.ws & 1) != 0) {
                // what an other-run's tail absorbed cut the whitespace run: not reachable back
                long a = afterNl & tws & ~nlE;
                if (a != 0) {
                    retractable = base + 63 - Long.numberOfLeadingZeros(a);
                } else if (!((tws & 1) != 0 && prevWs)) {
                    retractable = -1; // a fresh run with no newline yet
                }
            } else {
                retractable = -1;
            }
            prevOsf = ((osf >>> (len - 1)) & 1) != 0;

            st &= lead;
            if (bi == 0) {
                st |= 1;
            }
            // `'s` and the rest are one piece each: nothing opens inside, the char after opens
            long opens = contractionOpens;
            long inner = contractionInner;
            contractionOpens = 0;
            contractionInner = 0;
            long a = st & cur.apo;
            while (a != 0) {
                int j = Long.numberOfTrailingZeros(a);
                a &= a - 1;
                int p = base + j;
                int k = contractionLength(text, p, end);
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

        /**
         * Each bit of {@code x} moved back to the lead byte of its character, at most three steps;
         * a character cut by the block edge lands its lead in {@link #stealPatch}, for the previous
         * block.
         */
        long toLead(long x, long cont, long prevCont) {
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
        long digitGroups(long n, long lead, long cont, boolean prevIsDigit) {
            long m = n & lead & ~((n << 1) | (prevIsDigit ? 1 : 0));
            int cap = digitCap;
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

        /** How many digits of its current group the run has consumed on leaving the block. */
        int digitsSince(long groups, long tn, long n, long lead) {
            long g = groups & tn;
            int counted;
            if (g == 0) {
                counted = digitsSince + Long.bitCount(n & lead & tn);
            } else {
                long fromLastGroup = ~((1L << (63 - Long.numberOfLeadingZeros(g))) - 1);
                counted = Long.bitCount(n & lead & tn & fromLastGroup);
            }
            return counted % digitCap;
        }
    }

    /** The byte length of the contraction at {@code i}, or 0: the letters decoded as characters. */
    private int contractionLength(byte[] text, int i, int end) {
        int p1 = i + 1;
        if (p1 >= end) {
            return 0;
        }
        int a1 = at(text, p1, end);
        int c1 = a1 & 0x1FFFFF;
        if (is(c1, 's') || is(c1, 't') || is(c1, 'm') || is(c1, 'd')) {
            return p1 + (a1 >>> 24) - i;
        }
        int p2 = p1 + (a1 >>> 24);
        if (p2 >= end) {
            return 0;
        }
        int a2 = at(text, p2, end);
        int c2 = a2 & 0x1FFFFF;
        if ((is(c1, 'r') && is(c2, 'e'))
                || (is(c1, 'v') && is(c2, 'e'))
                || (is(c1, 'l') && is(c2, 'l'))) {
            return p2 + (a2 >>> 24) - i;
        }
        return 0;
    }

    /** Whether {@code c} spells {@code lower} in the contraction, either case. */
    private boolean is(int c, char lower) {
        if (c == lower || c == lower - ('a' - 'A')) {
            return true;
        }
        return foldLongS && c == 0x17F && lower == 's'; // the long s under Unicode case folding
    }

    // ── driving the blocks and popping the pieces ─────────────────────────────────────────────

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
     * boundary, and the text readable for 16 bytes past {@code end}.
     */
    int split(byte[] text, int start, int end, int[] ends) {
        int length = end - start;
        if (length <= 0) {
            return 0;
        }
        Run run = new Run(text, start, end);
        int nblk = (length + 63) >>> 6;
        run.seed = classify(text, start, Math.min(64, length), end, 0, run.cur);
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
            long st = run.rule(bi, base, len, last);
            run.starts[bi & (RING - 1)] = st;
            if (bi > 0) {
                run.starts[(bi - 1) & (RING - 1)] |= run.stealPatch;
            }
            // The starts of a block are final once the next block has run, except one a newline
            // further on may still retract: nothing from it on is popped until it is decided.
            if (run.retractable >= 0 && bi + 2 - popped > RING) {
                // the whitespace run is longer than the ring: decide it by looking ahead by hand
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
