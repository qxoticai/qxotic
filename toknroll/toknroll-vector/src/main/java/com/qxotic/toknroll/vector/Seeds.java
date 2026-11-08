package com.qxotic.toknroll.vector;

import static com.qxotic.toknroll.vector.PairMap.id;
import static com.qxotic.toknroll.vector.PairMap.rank;

import com.qxotic.toknroll.spi.TiktokenTables;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * The characters a word's merge starts from whole instead of from their bytes. A character of two
 * or three bytes whose token the merge loop gives back from those bytes may stand in for them at
 * the start, which spares the merges that only reassemble it: two a character in Chinese, where
 * they are most of a word's merges, one in Cyrillic. The ids come out the same when no merge could
 * have reached into the character before it formed, which is proven here, per vocabulary, when the
 * encoder is made.
 *
 * <p>Two things could reach in. A merge of two whole neighbours that outranks one of the
 * character's own merges would, in the seeded loop, be at hand before the byte loop had the
 * character ready; such a character is not seeded. And a merge that takes a leading or trailing
 * part of the character together with what stands beside it, a straddling token such as a space
 * with a lead byte: the byte loop takes it when it outranks the character's merge that would have
 * consumed the part. That depends on the neighbour, so a seeded character carries the bytes that
 * may not stand right before or after it, and beside one of those it starts from its bytes.
 *
 * <p>Characters beyond the Basic Multilingual Plane start from their bytes. ponytail: one 64K
 * table, 256 KB a tokenizer, covers every script of the corpora; emoji and the CJK extensions merge
 * from bytes, a second table if they ever matter.
 */
final class Seeds {
    private static final VarHandle LONGS =
            MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);
    private static final int BMP = 1 << 16;
    private static final int NO_MASK = 0;

    private final int[] entryOf; // code point -> entry + 1, 0 for none
    private final int[] ids;
    private final int[] before; // entry -> the mask of the bytes that may not come before it
    private final int[] after;
    private final long[] masks; // four longs a mask, a bit per byte value; mask 0 is empty
    private final int[] singleByte;

    private Seeds(
            int[] entryOf, int[] ids, int[] before, int[] after, long[] masks, int[] singleByte) {
        this.entryOf = entryOf;
        this.ids = ids;
        this.before = before;
        this.after = after;
        this.masks = masks;
        this.singleByte = singleByte;
    }

    /** How many characters are seeded. */
    int count() {
        int n = 0;
        for (int e : entryOf) {
            if (e != 0) {
                n++;
            }
        }
        return n;
    }

    /** Whether the word has no byte above 0x7F; such a word has nothing to seed. */
    static boolean ascii(byte[] bytes, int off, int length) {
        long acc = 0;
        int i = 0;
        for (; i + 8 <= length; i += 8) {
            acc |= (long) LONGS.get(bytes, off + i);
        }
        for (; i < length; i++) {
            acc |= bytes[off + i];
        }
        return (acc & 0x8080808080808080L) == 0;
    }

    /**
     * The tokens the merge of {@code bytes[off, off + length)} starts from, into {@code tokens}: a
     * seeded character's id where the vocabulary and the neighbours allow, else the bytes' own;
     * returns how many.
     */
    int fill(byte[] bytes, int off, int length, int[] tokens) {
        int[] singleByte = this.singleByte;
        int end = off + length;
        int n = 0;
        for (int i = off; i < end; ) {
            int b = bytes[i];
            if (b >= 0) {
                tokens[n++] = singleByte[b];
                i++;
                continue;
            }
            int len = length(bytes, i, end);
            if (len == 2 || len == 3) {
                int e = entryOf[decode(bytes, i, len)] - 1;
                if (e >= 0
                        && allowed(before[e], i > off ? bytes[i - 1] & 0xFF : -1)
                        && allowed(after[e], i + len < end ? bytes[i + len] & 0xFF : -1)) {
                    tokens[n++] = ids[e];
                    i += len;
                    continue;
                }
            }
            // a byte of a character that is not seeded, or of no character: its own token
            tokens[n++] = singleByte[b & 0xFF];
            i++;
        }
        return n;
    }

    private boolean allowed(int mask, int neighbour) {
        return mask == NO_MASK
                || neighbour < 0
                || (masks[(mask << 2) + (neighbour >>> 6)] & (1L << (neighbour & 63))) == 0;
    }

    // ---- the proof, at load ----

    /** The seeds the model allows, or null when it allows none. */
    static Seeds prove(
            TiktokenTables model, byte[][] tokenBytes, PairMap merges, int[] singleByte) {
        int tokenCount = tokenBytes.length;
        int[] entryOf = new int[BMP];
        int candidates = 0;
        for (byte[] bytes : tokenBytes) {
            if (character(bytes) >= 0) {
                candidates++;
            }
        }
        int[] ids = new int[candidates];
        int[] maxInternal = new int[candidates]; // the rank of the character's slowest own merge
        // the character's leading (trailing) tokens along its own merge, with the highest rank
        // merged while each was leading (trailing): a straddling merge under that rank wins
        Sides lead = new Sides(2 * candidates);
        Sides trail = new Sides(2 * candidates);
        int entries = 0;
        int[] toks = new int[3];
        for (byte[] bytes : tokenBytes) {
            int cp = character(bytes);
            if (cp < 0 || entryOf[cp] != 0) {
                continue;
            }
            int n = bytes.length;
            for (int k = 0; k < n; k++) {
                toks[k] = singleByte[bytes[k] & 0xFF];
            }
            int max = -1;
            int leadFrom = lead.size;
            int trailFrom = trail.size;
            while (n > 1) {
                long best = PairMap.NONE;
                int at = -1;
                for (int k = 0; k + 1 < n; k++) {
                    long m = merges.get(toks[k], toks[k + 1]);
                    if (m != PairMap.NONE && (best == PairMap.NONE || rank(m) < rank(best))) {
                        best = m;
                        at = k;
                    }
                }
                if (best == PairMap.NONE) {
                    break;
                }
                int rank = rank(best);
                max = Math.max(max, rank);
                lead.note(toks[0], entries, rank);
                trail.note(toks[n - 1], entries, rank);
                toks[at] = id(best);
                System.arraycopy(toks, at + 2, toks, at + 1, n - at - 2);
                n--;
            }
            if (n != 1) {
                lead.forget(leadFrom);
                trail.forget(trailFrom);
                continue; // its bytes do not merge into one token
            }
            ids[entries] = toks[0];
            maxInternal[entries] = max;
            entryOf[cp] = ++entries;
        }
        if (entries == 0) {
            return null;
        }
        lead.index(tokenCount);
        trail.index(tokenCount);

        long[] before = new long[4 * entries];
        long[] after = new long[4 * entries];
        boolean[] unsafe = new boolean[entries];
        for (int i = 0, count = model.mergeCount(); i < count; i++) {
            int l = model.mergeLeft(i);
            int r = model.mergeRight(i);
            int rank = model.mergeRank(i);
            byte[] lb = tokenBytes[l];
            byte[] rb = tokenBytes[r];
            if (lb == null || rb == null || lb.length == 0 || rb.length == 0) {
                continue;
            }
            // a straddling merge into a leading part: the byte before is the left token's last
            for (int p = lead.start[r]; p < lead.start[r + 1]; p++) {
                int k = lead.order[p];
                if (rank <= lead.max[k]) {
                    set(before, lead.entry[k], lb[lb.length - 1] & 0xFF);
                }
            }
            for (int p = trail.start[l]; p < trail.start[l + 1]; p++) {
                int k = trail.order[p];
                if (rank <= trail.max[k]) {
                    set(after, trail.entry[k], rb[0] & 0xFF);
                }
            }
            // a merge taking a whole character that outranks one of the character's own: the
            // seeded loop would have it at hand before the byte loop had the character
            if (aligned(lb)) {
                markOutranked(lb, rank, entryOf, maxInternal, unsafe);
            }
            if (aligned(rb)) {
                markOutranked(rb, rank, entryOf, maxInternal, unsafe);
            }
        }
        int seeded = 0;
        for (int cp = 0; cp < BMP; cp++) {
            int e = entryOf[cp] - 1;
            if (e >= 0) {
                if (unsafe[e]) {
                    entryOf[cp] = 0;
                } else {
                    seeded++;
                }
            }
        }
        if (seeded == 0) {
            return null;
        }
        Masks masks = new Masks();
        int[] beforeMask = new int[entries];
        int[] afterMask = new int[entries];
        for (int e = 0; e < entries; e++) {
            if (!unsafe[e]) {
                beforeMask[e] = masks.of(before, e);
                afterMask[e] = masks.of(after, e);
            }
        }
        return new Seeds(entryOf, ids, beforeMask, afterMask, masks.toArray(), singleByte);
    }

    /** Marks the seeded characters of {@code bytes} that a merge of rank {@code rank} outranks. */
    private static void markOutranked(
            byte[] bytes, int rank, int[] entryOf, int[] maxInternal, boolean[] unsafe) {
        for (int i = 0; i < bytes.length; ) {
            int len = length(bytes, i);
            if (len == 2 || len == 3) {
                int e = entryOf[decode(bytes, i, len)] - 1;
                if (e >= 0 && rank <= maxInternal[e]) {
                    unsafe[e] = true;
                }
            }
            i += len;
        }
    }

    private static void set(long[] bits, int entry, int value) {
        bits[(entry << 2) + (value >>> 6)] |= 1L << (value & 63);
    }

    /** The code point of a token that is one character of two or three bytes, else -1. */
    private static int character(byte[] bytes) {
        if (bytes == null || bytes.length < 2 || bytes.length > 3) {
            return -1;
        }
        return length(bytes, 0) == bytes.length ? decode(bytes, 0, bytes.length) : -1;
    }

    /** Whether the bytes are whole characters from the first to the last. */
    private static boolean aligned(byte[] bytes) {
        for (int i = 0; i < bytes.length; ) {
            int len = length(bytes, i);
            if (len == 0) {
                return false;
            }
            i += len;
        }
        return true;
    }

    private static int length(byte[] bytes, int i) {
        return length(bytes, i, bytes.length);
    }

    /**
     * The length of the well-formed character at {@code i}, or 0 when none starts there: a lead
     * byte with its continuation bytes present before {@code end}, no overlong form, no surrogate.
     */
    private static int length(byte[] bytes, int i, int end) {
        int lead = bytes[i] & 0xFF;
        if (lead < 0x80) {
            return 1;
        }
        int len = lead < 0xC2 ? 0 : lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : lead < 0xF5 ? 4 : 0;
        if (len == 0 || i + len > end) {
            return 0;
        }
        for (int k = 1; k < len; k++) {
            if ((bytes[i + k] & 0xC0) != 0x80) {
                return 0;
            }
        }
        int c1 = bytes[i + 1] & 0xFF;
        if ((lead == 0xE0 && c1 < 0xA0)
                || (lead == 0xED && c1 >= 0xA0)
                || (lead == 0xF0 && c1 < 0x90)
                || (lead == 0xF4 && c1 >= 0x90)) {
            return 0;
        }
        return len;
    }

    private static int decode(byte[] bytes, int i, int len) {
        int lead = bytes[i] & 0xFF;
        return len == 2
                ? (lead & 0x1F) << 6 | (bytes[i + 1] & 0x3F)
                : (lead & 0x0F) << 12 | (bytes[i + 1] & 0x3F) << 6 | (bytes[i + 2] & 0x3F);
    }

    /** The leading or trailing tokens of the candidates along their merges, indexed by token. */
    private static final class Sides {
        int[] token;
        int[] entry;
        int[] max;
        int size;
        int[] start; // by token, after index()
        int[] order;

        Sides(int capacity) {
            token = new int[capacity];
            entry = new int[capacity];
            max = new int[capacity];
        }

        /**
         * {@code tok} is entry {@code e}'s token on this side while a merge of {@code rank} is
         * taken.
         */
        void note(int tok, int e, int rank) {
            if (size > 0 && token[size - 1] == tok && entry[size - 1] == e) {
                max[size - 1] = Math.max(max[size - 1], rank);
                return;
            }
            if (size == token.length) {
                token = Arrays.copyOf(token, 2 * size);
                entry = Arrays.copyOf(entry, 2 * size);
                max = Arrays.copyOf(max, 2 * size);
            }
            token[size] = tok;
            entry[size] = e;
            max[size] = rank;
            size++;
        }

        /**
         * Drops what was noted since the size was {@code from}: that candidate did not merge whole.
         */
        void forget(int from) {
            size = from;
        }

        /** Sorts the notes by token, so that a merge finds those of its token in one run. */
        void index(int tokenCount) {
            start = new int[tokenCount + 2];
            for (int k = 0; k < size; k++) {
                start[token[k] + 2]++;
            }
            for (int t = 2; t < start.length; t++) {
                start[t] += start[t - 1];
            }
            order = new int[size];
            for (int k = 0; k < size; k++) {
                order[start[token[k] + 1]++] = k;
            }
        }
    }

    /** The distinct byte masks, mask 0 empty. */
    private static final class Masks {
        private record Key(long a, long b, long c, long d) {}

        private final Map<Key, Integer> index = new HashMap<>();
        private long[] masks = new long[4 * 64];
        private int count = 1;

        int of(long[] bits, int entry) {
            int at = entry << 2;
            Key key = new Key(bits[at], bits[at + 1], bits[at + 2], bits[at + 3]);
            if (key.a == 0 && key.b == 0 && key.c == 0 && key.d == 0) {
                return NO_MASK;
            }
            Integer found = index.get(key);
            if (found != null) {
                return found;
            }
            if (4 * (count + 1) > masks.length) {
                masks = Arrays.copyOf(masks, 2 * masks.length);
            }
            System.arraycopy(bits, at, masks, 4 * count, 4);
            index.put(key, count);
            return count++;
        }

        long[] toArray() {
            return Arrays.copyOf(masks, 4 * count);
        }
    }
}
