package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Specials;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

/**
 * The special tokens as two tries, one over chars and one over UTF-8 bytes: from each position the
 * walk follows the input for as long as it spells the start of some special token, and the last
 * node that ends one gives the longest special token starting there. A bit per possible first unit
 * skips the positions no special token starts at with one test, so the cost is per position, not
 * per special token: Gemma 3 has 6322 starting with {@code <}.
 *
 * <p>A special token's UTF-8 starts with a lead byte and ends a character, so in well-formed UTF-8
 * a byte match is a char match, and the text between matches stays on the tokenizer's bytes path.
 */
final class SpecialsImpl implements Specials {
    private static final SpecialsImpl NONE = new SpecialsImpl(Map.of());

    private final Map<String, Integer> tokens;
    private final Trie chars;
    private final Trie bytes;

    private SpecialsImpl(Map<String, Integer> tokens) {
        this.tokens = Map.copyOf(tokens);
        Node charRoot = new Node();
        Node byteRoot = new Node();
        this.tokens.forEach(
                (text, id) -> {
                    charRoot.add(text.chars().toArray(), id);
                    byte[] utf8 = text.getBytes(StandardCharsets.UTF_8);
                    int[] units = new int[utf8.length];
                    for (int i = 0; i < utf8.length; i++) {
                        units[i] = utf8[i] & 0xFF;
                    }
                    byteRoot.add(units, id);
                });
        this.chars = new Trie(charRoot, Character.MAX_VALUE + 1);
        this.bytes = new Trie(byteRoot, 256);
    }

    static Specials none() {
        return NONE;
    }

    static Specials of(Tokenizer tokenizer) {
        Vocabulary vocabulary = tokenizer.vocabulary();
        BitSet ids = VocabularyImpl.ids(vocabulary);
        Map<String, Integer> tokens = new HashMap<>();
        for (int id = ids.nextSetBit(0); id >= 0; id = ids.nextSetBit(id + 1)) {
            if (isSpecial(vocabulary, id)) {
                String text = textAfterAnother(tokenizer, id);
                if (!text.isEmpty()) {
                    tokens.merge(text, id, Math::min);
                }
            }
        }
        return tokens.isEmpty() ? NONE : new SpecialsImpl(tokens);
    }

    /**
     * The text of {@code id} as a token after another decodes: a decoder may trim the start of the
     * text, as Metaspace's leading space, and the token keeps it anywhere else.
     */
    private static String textAfterAnother(Tokenizer tokenizer, int id) {
        int first = tokenizer.countBytes(IntSequence.of(id));
        byte[] twice = tokenizer.decodeBytes(IntSequence.of(id, id));
        return new String(twice, first, twice.length - first, StandardCharsets.UTF_8);
    }

    private static boolean isSpecial(Vocabulary vocabulary, int id) {
        return vocabulary.isTokenOfType(id, StandardTokenType.CONTROL)
                || vocabulary.isTokenOfType(id, StandardTokenType.USER_DEFINED)
                || vocabulary.isTokenOfType(id, StandardTokenType.UNKNOWN);
    }

    @Override
    public Map<String, Integer> tokens() {
        return tokens;
    }

    @Override
    public void encodeInto(Tokenizer tokenizer, CharSequence text, IntSequence.Builder out) {
        Objects.requireNonNull(tokenizer, "tokenizer");
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(out, "out");
        int n = text.length();
        int gap = 0; // where the text not yet encoded starts
        for (int at = 0; at < n; at++) {
            if (!chars.starts(text.charAt(at))) {
                continue; // most positions: one bit test
            }
            // the longest special token starting at `at`: the deepest node on the walk ending one
            int id = -1;
            int end = at;
            Node node = chars.root;
            for (int i = at; i < n && (node = node.child(text.charAt(i))) != null; i++) {
                if (node.id >= 0) {
                    id = node.id;
                    end = i + 1;
                }
            }
            if (id < 0) {
                continue;
            }
            if (at > gap) {
                tokenizer.encodeInto(text, gap, at, out);
            }
            out.add(id);
            gap = end;
            at = end - 1;
        }
        if (gap < n) {
            tokenizer.encodeInto(text, gap, n, out);
        }
    }

    @Override
    public void encodeUtf8Into(
            Tokenizer tokenizer, byte[] utf8, int from, int to, IntSequence.Builder out) {
        Objects.requireNonNull(tokenizer, "tokenizer");
        Objects.requireNonNull(utf8, "utf8");
        Objects.requireNonNull(out, "out");
        Objects.checkFromToIndex(from, to, utf8.length);
        int gap = from;
        for (int at = bytes.next(utf8, from, to); at < to; at = bytes.next(utf8, at + 1, to)) {
            int id = -1;
            int end = at;
            Node node = bytes.root;
            for (int i = at; i < to && (node = node.child(utf8[i] & 0xFF)) != null; i++) {
                if (node.id >= 0) {
                    id = node.id;
                    end = i + 1;
                }
            }
            if (id < 0) {
                continue;
            }
            if (at > gap) {
                tokenizer.encodeUtf8Into(utf8, gap, at, out);
            }
            out.add(id);
            gap = end;
            at = end - 1;
        }
        if (gap < to) {
            tokenizer.encodeUtf8Into(utf8, gap, to, out);
        }
    }

    /** A trie and a bit per unit that begins some special token. */
    private static final class Trie {
        private static final VarHandle LONGS =
                MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);
        private static final long ONES = 0x0101010101010101L;

        final Node root;
        final long[] first;
        // up to four first bytes, each repeated in every byte of a word, for next to test eight
        // bytes at once; more, and swar is off
        private final boolean swar;
        private final long lead0, lead1, lead2, lead3;

        Trie(Node root, int units) {
            this.root = root.freeze();
            this.first = new long[units >>> 6];
            for (int unit : this.root.keys) {
                first[unit >>> 6] |= 1L << unit;
            }
            int[] k = this.root.keys;
            this.swar = units == 256 && k.length >= 1 && k.length <= 4;
            this.lead0 = swar ? k[0] * ONES : 0;
            this.lead1 = swar ? k[Math.min(1, k.length - 1)] * ONES : 0;
            this.lead2 = swar ? k[Math.min(2, k.length - 1)] * ONES : 0;
            this.lead3 = swar ? k[Math.min(3, k.length - 1)] * ONES : 0;
        }

        boolean starts(int unit) {
            return (first[unit >>> 6] & (1L << unit)) != 0;
        }

        /** The first position of {@code utf8[i, to)} whose byte begins some special, or to. */
        int next(byte[] utf8, int i, int to) {
            if (swar) {
                for (; i + 8 <= to; i += 8) {
                    long w = (long) LONGS.get(utf8, i);
                    long hits =
                            zeros(w ^ lead0)
                                    | zeros(w ^ lead1)
                                    | zeros(w ^ lead2)
                                    | zeros(w ^ lead3);
                    if (hits != 0) {
                        return i + (Long.numberOfTrailingZeros(hits) >>> 3);
                    }
                }
            }
            while (i < to && !starts(utf8[i] & 0xFF)) {
                i++;
            }
            return i;
        }

        /** The high bit of each zero byte of {@code x}; exact at the lowest one, all next reads. */
        private static long zeros(long x) {
            return (x - ONES) & ~x & 0x8080808080808080L;
        }
    }

    /**
     * A prefix of some special token: its continuations, sorted, and the id of the one it spells,
     * or -1. Built through {@link #add}, then frozen into the two arrays the walk reads.
     */
    private static final class Node {
        private TreeMap<Integer, Node> building = new TreeMap<>();
        int[] keys;
        Node[] children;
        int id = -1;

        void add(int[] units, int tokenId) {
            Node node = this;
            for (int unit : units) {
                node = node.building.computeIfAbsent(unit, u -> new Node());
            }
            node.id = tokenId;
        }

        /** Every node frozen, by a loop: a token is as deep as it is long, too deep to recurse. */
        Node freeze() {
            ArrayDeque<Node> unfrozen = new ArrayDeque<>();
            unfrozen.push(this);
            while (!unfrozen.isEmpty()) {
                Node node = unfrozen.pop();
                node.keys = new int[node.building.size()];
                node.children = new Node[node.keys.length];
                int i = 0;
                for (Map.Entry<Integer, Node> child : node.building.entrySet()) {
                    node.keys[i] = child.getKey();
                    node.children[i++] = child.getValue();
                    unfrozen.push(child.getValue());
                }
                node.building = null;
            }
            return this;
        }

        Node child(int unit) {
            int[] k = keys;
            if (k.length <= 8) { // most nodes have one child: a scan for few, a search for many
                for (int i = 0; i < k.length; i++) {
                    if (k[i] == unit) {
                        return children[i];
                    }
                }
                return null;
            }
            int i = Arrays.binarySearch(k, unit);
            return i >= 0 ? children[i] : null;
        }
    }
}
