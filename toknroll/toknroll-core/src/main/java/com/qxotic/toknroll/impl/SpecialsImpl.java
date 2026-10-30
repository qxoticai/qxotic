package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Specials;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;

/**
 * Finds the specials in text the way Hugging Face's added vocabulary does. A position is a
 * candidate only when its first char can begin a special (one bit test); its first two chars then
 * name the lengths of the specials that begin with them (a bitmask), and each of those lengths is
 * confirmed by hashing the text up to it and probing a table. No special is a prefix of another, so
 * a position matches at most one, and the leftmost occurrence of any special is the next match.
 *
 * <p>Well-formed UTF-8 is matched the same way on its bytes, against the specials' UTF-8 spellings.
 * A spelling starts with a lead byte and ends a character, so in well-formed text a byte match is a
 * char match, and the text between matches stays on the tokenizer's bytes path.
 */
final class SpecialsImpl implements Specials {
    private static final SpecialsImpl NONE = new SpecialsImpl(Set.of(), Map.of(), List.of());

    private static final int PAIRS = 1 << 12; // the pair tables, indexed by pair(first, second)
    private static final int LONG = 63; // the length bit of every special past 64 units

    private final Set<String> tokens;
    private final Map<String, Integer> specialToId;
    private final String[] spellings;
    private final int[] ids;

    // chars
    private final long[] first = new long[1 << 10]; // a special's first char, one bit per char
    private final long[] alone = new long[1 << 10]; // a one-char special
    private final long[] pairLengths = new long[PAIRS]; // bit length - 2 of each special there
    private final int[] longLengths; // distinct lengths past 64, ascending
    private final int[] slots; // open addressing on String.hashCode: index + 1, 0 empty
    private final int[] hashes;

    // UTF-8: the same tables over bytes
    private final boolean[] leads = new boolean[256];
    // up to SWAR_LEADS lead bytes, each in every byte of a word (repeated to fill all four), so the
    // scan tests eight bytes at a time; more leads, and it tests one byte at a time
    private static final int SWAR_LEADS = 4;
    private final boolean swar;
    private final long lead0, lead1, lead2, lead3;
    private final int[] singleBytes = new int[256]; // the one-byte special, or -1
    private final long[] bytePairLengths = new long[PAIRS];
    // the bytes every special of a pair slot begins with, up to eight, little-endian, and their
    // mask
    private final long[] bytePairPrefix = new long[PAIRS];
    private final long[] bytePairMask = new long[PAIRS];
    private final int[] longByteLengths;
    private final byte[][] utf8;
    private final int[] byteSlots;
    private final int[] byteHashes;

    private SpecialsImpl(
            Set<String> tokens, Map<String, Integer> specialToId, List<String> sorted) {
        this.tokens = tokens;
        this.specialToId = specialToId;
        int count = sorted.size();
        spellings = sorted.toArray(new String[0]);
        ids = new int[count];
        hashes = new int[count];
        slots = new int[Integer.highestOneBit(Math.max(1, count) * 4)];
        utf8 = new byte[count][];
        byteHashes = new int[count];
        byteSlots = new int[slots.length];
        Arrays.fill(singleBytes, -1);
        Set<Integer> longs = new TreeSet<>();
        Set<Integer> longBytes = new TreeSet<>();
        for (int t = 0; t < count; t++) {
            String special = spellings[t];
            ids[t] = specialToId.get(special);
            hashes[t] = special.hashCode();
            insert(slots, hashes[t], t);
            first[special.charAt(0) >>> 6] |= 1L << special.charAt(0);
            if (special.length() == 1) {
                alone[special.charAt(0) >>> 6] |= 1L << special.charAt(0);
            } else {
                pairLengths[pair(special.charAt(0), special.charAt(1))] |=
                        lengthBit(special.length());
                if (special.length() > 64) longs.add(special.length());
            }

            byte[] bytes = special.getBytes(StandardCharsets.UTF_8);
            utf8[t] = bytes;
            byteHashes[t] = hash(bytes, 0, bytes.length);
            insert(byteSlots, byteHashes[t], t);
            leads[bytes[0] & 0xFF] = true;
            if (bytes.length == 1) {
                singleBytes[bytes[0] & 0xFF] = t;
            } else {
                int slot = pair(bytes[0] & 0xFF, bytes[1] & 0xFF);
                long prefix = 0;
                int n = Math.min(8, bytes.length);
                for (int k = n - 1; k >= 0; k--) prefix = prefix << 8 | (bytes[k] & 0xFF);
                long mask = n == 8 ? -1L : (1L << (8 * n)) - 1;
                if (bytePairLengths[slot] != 0) { // keep what this one shares with the others
                    long differ = (bytePairPrefix[slot] ^ prefix) | ~(bytePairMask[slot] & mask);
                    mask =
                            differ == 0
                                    ? mask
                                    : (1L << (Long.numberOfTrailingZeros(differ) & ~7)) - 1;
                }
                bytePairLengths[slot] |= lengthBit(bytes.length);
                bytePairPrefix[slot] = prefix & mask;
                bytePairMask[slot] = mask;
                if (bytes.length > 64) longBytes.add(bytes.length);
            }
        }
        longLengths = longs.stream().mapToInt(Integer::intValue).toArray();
        long[] words = new long[SWAR_LEADS];
        int distinctLeads = 0;
        for (int b = 0; b < 256; b++) {
            if (!leads[b]) continue;
            if (distinctLeads < SWAR_LEADS) words[distinctLeads] = b * ONES;
            distinctLeads++;
        }
        swar = distinctLeads > 0 && distinctLeads <= SWAR_LEADS;
        for (int k = Math.max(1, distinctLeads); k < SWAR_LEADS; k++) words[k] = words[0];
        lead0 = words[0];
        lead1 = words[1];
        lead2 = words[2];
        lead3 = words[3];
        longByteLengths = longBytes.stream().mapToInt(Integer::intValue).toArray();
    }

    private static final VarHandle LONGS =
            MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);
    private static final long ONES = 0x0101010101010101L;
    private static final long HIGHS = 0x8080808080808080L;

    /** The high bit of each zero byte of {@code x}; exact at the lowest one, all a scan needs. */
    private static long zeros(long x) {
        return (x - ONES) & ~x & HIGHS;
    }

    /**
     * The first position in {@code text[i, to)} whose first two bytes can begin a special, or
     * {@code to}. Eight bytes are tested for a lead at a time, each lead then for its pair; a
     * position it returns may still not match, but none it skips can.
     */
    private int nextLead(byte[] text, int i, int to) {
        if (swar) {
            for (; i + 8 <= to; i += 8) {
                long w = (long) LONGS.get(text, i);
                long hits =
                        zeros(w ^ lead0) | zeros(w ^ lead1) | zeros(w ^ lead2) | zeros(w ^ lead3);
                for (; hits != 0; hits &= hits - 1) {
                    int k = i + (Long.numberOfTrailingZeros(hits) >>> 3);
                    if (canBegin(text, k, to)) return k;
                }
            }
        }
        for (; i < to; i++) {
            if (leads[text[i] & 0xFF] && canBegin(text, i, to)) return i;
        }
        return to;
    }

    private boolean canBegin(byte[] text, int k, int to) {
        int b = text[k] & 0xFF;
        if (singleBytes[b] >= 0) return true;
        if (k + 1 >= to) return false;
        int slot = pair(b, text[k + 1] & 0xFF);
        if (bytePairLengths[slot] == 0) return false;
        // the prefix its specials share, when eight bytes are there to compare
        return k + 8 > to
                || ((long) LONGS.get(text, k) & bytePairMask[slot]) == bytePairPrefix[slot];
    }

    private static int pair(int first, int second) {
        return (31 * first + second) & (PAIRS - 1);
    }

    private static long lengthBit(int length) {
        return 1L << Math.min(length - 2, LONG);
    }

    private static void insert(int[] slots, int hash, int t) {
        int slot = hash & (slots.length - 1);
        while (slots[slot] != 0) slot = (slot + 1) & (slots.length - 1);
        slots[slot] = t + 1;
    }

    private static int hash(CharSequence text, int from, int to) {
        int h = 0;
        for (int k = from; k < to; k++) h = 31 * h + text.charAt(k);
        return h;
    }

    private static int hash(byte[] text, int from, int to) {
        int h = 0;
        for (int k = from; k < to; k++) h = 31 * h + (text[k] & 0xFF);
        return h;
    }

    private static boolean has(long[] bits, char c) {
        return (bits[c >>> 6] & (1L << c)) != 0;
    }

    /** The special that starts at {@code i}, whose first char can begin one, or -1. */
    private int matchAt(CharSequence text, int i, int end) {
        char c = text.charAt(i);
        if (has(alone, c)) return probe(text, i, 1, c);
        if (i + 1 >= end) return -1;
        char d = text.charAt(i + 1);
        int h = 31 * c + d;
        int k = 2;
        for (long mask = pairLengths[pair(c, d)]; mask != 0; mask &= mask - 1) {
            int bit = Long.numberOfTrailingZeros(mask);
            if (bit == LONG) {
                for (int length : longLengths) {
                    if (i + length > end) return -1;
                    int t = probe(text, i, length, hash(text, i, i + length));
                    if (t >= 0) return t;
                }
                return -1;
            }
            int length = bit + 2;
            if (i + length > end) return -1;
            for (; k < length; k++) h = 31 * h + text.charAt(i + k);
            int t = probe(text, i, length, h);
            if (t >= 0) return t;
        }
        return -1;
    }

    private int probe(CharSequence text, int i, int length, int h) {
        for (int slot = h & (slots.length - 1);
                slots[slot] != 0;
                slot = (slot + 1) & (slots.length - 1)) {
            int t = slots[slot] - 1;
            if (hashes[t] == h
                    && spellings[t].length() == length
                    && regionEquals(text, i, spellings[t])) {
                return t;
            }
        }
        return -1;
    }

    /** The special whose UTF-8 starts at {@code i}, whose first byte can begin one, or -1. */
    private int matchAt(byte[] text, int i, int end) {
        int b = text[i] & 0xFF;
        if (singleBytes[b] >= 0) return singleBytes[b];
        if (i + 1 >= end) return -1;
        int c = text[i + 1] & 0xFF;
        int h = 31 * b + c;
        int k = 2;
        for (long mask = bytePairLengths[pair(b, c)]; mask != 0; mask &= mask - 1) {
            int bit = Long.numberOfTrailingZeros(mask);
            if (bit == LONG) {
                for (int length : longByteLengths) {
                    if (i + length > end) return -1;
                    int t = probe(text, i, length, hash(text, i, i + length));
                    if (t >= 0) return t;
                }
                return -1;
            }
            int length = bit + 2;
            if (i + length > end) return -1;
            for (; k < length; k++) h = 31 * h + (text[i + k] & 0xFF);
            int t = probe(text, i, length, h);
            if (t >= 0) return t;
        }
        return -1;
    }

    private int probe(byte[] text, int i, int length, int h) {
        for (int slot = h & (byteSlots.length - 1);
                byteSlots[slot] != 0;
                slot = (slot + 1) & (byteSlots.length - 1)) {
            int t = byteSlots[slot] - 1;
            if (byteHashes[t] == h
                    && utf8[t].length == length
                    && Arrays.equals(text, i, i + length, utf8[t], 0, length)) {
                return t;
            }
        }
        return -1;
    }

    private static boolean regionEquals(CharSequence text, int i, String special) {
        for (int k = 0; k < special.length(); k++) {
            if (text.charAt(i + k) != special.charAt(k)) return false;
        }
        return true;
    }

    public static Specials none() {
        return NONE;
    }

    public static Specials compile(Vocabulary vocabulary, Set<String> specials) {
        if (specials.isEmpty()) {
            return NONE;
        }

        List<String> sorted = new ArrayList<>(specials.size());
        for (String special : specials) {
            if (special == null) {
                throw new IllegalArgumentException("special token cannot be null");
            }
            if (special.isEmpty()) {
                throw new IllegalArgumentException("special token cannot be empty");
            }
            if (!vocabulary.contains(special)) {
                throw new IllegalArgumentException(
                        "special token not present in vocabulary: " + special);
            }
            sorted.add(special);
        }
        Collections.sort(sorted);
        validateNoPrefixConflicts(sorted);

        Map<String, Integer> specialToId = new LinkedHashMap<>(sorted.size());
        for (String special : sorted) {
            specialToId.put(special, vocabulary.id(special));
        }
        Set<String> tokens = Collections.unmodifiableSet(new LinkedHashSet<>(sorted));
        return new SpecialsImpl(tokens, Collections.unmodifiableMap(specialToId), sorted);
    }

    private static void validateNoPrefixConflicts(List<String> sorted) {
        for (int i = 1; i < sorted.size(); i++) {
            String previous = sorted.get(i - 1);
            String current = sorted.get(i);
            if (current.startsWith(previous)) {
                throw new IllegalArgumentException(
                        "special token prefix conflict: '"
                                + previous
                                + "' is a prefix of '"
                                + current
                                + "'");
            }
        }
    }

    @Override
    public Set<String> tokens() {
        return tokens;
    }

    @Override
    public void encodeInto(Tokenizer tokenizer, CharSequence text, IntSequence.Builder out) {
        Objects.requireNonNull(tokenizer, "tokenizer");
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(out, "out");

        if (specialToId.isEmpty()) {
            tokenizer.encodeInto(text, out);
            return;
        }

        int n = text.length();
        int cursor = 0;
        for (int i = 0; i < n; i++) {
            if (!has(first, text.charAt(i))) continue; // most chars: one bit test
            int t = matchAt(text, i, n);
            if (t < 0) continue;
            if (i > cursor) {
                tokenizer.encodeInto(text, cursor, i, out);
            }
            out.add(ids[t]);
            cursor = i + spellings[t].length();
            i = cursor - 1;
        }
        if (cursor < n) {
            tokenizer.encodeInto(text, cursor, n, out);
        }
    }

    @Override
    public void encodeUtf8Into(
            Tokenizer tokenizer, byte[] text, int from, int to, IntSequence.Builder out) {
        Objects.requireNonNull(tokenizer, "tokenizer");
        Objects.requireNonNull(text, "utf8");
        Objects.requireNonNull(out, "out");
        Objects.checkFromToIndex(from, to, text.length);
        if (specialToId.isEmpty()) {
            tokenizer.encodeUtf8Into(text, from, to, out);
            return;
        }
        int cursor = from;
        for (int i = nextLead(text, from, to); i < to; ) {
            int t = matchAt(text, i, to);
            if (t < 0) {
                i = nextLead(text, i + 1, to);
                continue;
            }
            if (i > cursor) {
                tokenizer.encodeUtf8Into(text, cursor, i, out);
            }
            out.add(ids[t]);
            cursor = i + utf8[t].length;
            i = nextLead(text, cursor, to);
        }
        if (cursor < to) {
            tokenizer.encodeUtf8Into(text, cursor, to, out);
        }
    }
}
