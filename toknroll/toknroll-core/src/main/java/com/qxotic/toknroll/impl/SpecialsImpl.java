package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Specials;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;

/**
 * Finds the specials in text the way Hugging Face's added vocabulary does: a position is a
 * candidate only when its first two chars can begin a special (two bit tests), and a candidate is
 * confirmed by hashing the text once up to the longest special and probing a table at each
 * special's length. No special is a prefix of another, so a position matches at most one, and the
 * leftmost occurrence of any special is the next match.
 */
final class SpecialsImpl implements Specials {
    private static final SpecialsImpl NONE = new SpecialsImpl(Set.of(), Map.of(), List.of());

    private final Set<String> tokens;
    private final Map<String, Integer> specialToId;

    private final long[] first = new long[1 << 10]; // a special's first char, one bit per char
    private final long[] second = new long[1 << 10]; // its second char
    private final long[] alone = new long[1 << 10]; // a one-char special
    private final int[] lengths; // distinct lengths, ascending
    private final String[] spellings;
    private final int[] ids;
    private final int[] slots; // open addressing on String.hashCode: index + 1, 0 empty
    private final int[] hashes;

    private SpecialsImpl(
            Set<String> tokens, Map<String, Integer> specialToId, List<String> sorted) {
        this.tokens = tokens;
        this.specialToId = specialToId;
        int count = sorted.size();
        spellings = sorted.toArray(new String[0]);
        ids = new int[count];
        hashes = new int[count];
        slots = new int[Integer.highestOneBit(Math.max(1, count) * 4)];
        Set<Integer> distinct = new TreeSet<>();
        for (int t = 0; t < count; t++) {
            String special = spellings[t];
            ids[t] = specialToId.get(special);
            hashes[t] = special.hashCode();
            set(first, special.charAt(0));
            if (special.length() == 1) set(alone, special.charAt(0));
            else set(second, special.charAt(1));
            distinct.add(special.length());
            int slot = hashes[t] & (slots.length - 1);
            while (slots[slot] != 0) slot = (slot + 1) & (slots.length - 1);
            slots[slot] = t + 1;
        }
        lengths = distinct.stream().mapToInt(Integer::intValue).toArray();
    }

    private static void set(long[] bits, char c) {
        bits[c >>> 6] |= 1L << c;
    }

    private static boolean has(long[] bits, char c) {
        return (bits[c >>> 6] & (1L << c)) != 0;
    }

    /** The special that starts at {@code i}, or -1. */
    private int matchAt(CharSequence text, int i, int end) {
        char c = text.charAt(i);
        if (!has(first, c)) return -1;
        if (!has(alone, c) && (i + 1 >= end || !has(second, text.charAt(i + 1)))) return -1;
        int h = 0;
        int k = 0;
        for (int length : lengths) {
            if (i + length > end) return -1;
            for (; k < length; k++) h = 31 * h + text.charAt(i + k);
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
}
