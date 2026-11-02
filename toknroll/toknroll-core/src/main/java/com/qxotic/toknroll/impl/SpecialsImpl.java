package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Specials;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.nio.charset.StandardCharsets;
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
                    for (int i = 0; i < utf8.length; i++) units[i] = utf8[i] & 0xFF;
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
        // every id, from both sides: two ids can share a spelling, and the spelling side keeps one
        BitSet ids = new BitSet();
        for (Map.Entry<String, Integer> entry : vocabulary) ids.set(entry.getValue());
        for (int id = 0; id < vocabulary.size(); id++) if (vocabulary.contains(id)) ids.set(id);
        Map<String, Integer> tokens = new HashMap<>();
        for (int id = ids.nextSetBit(0); id >= 0; id = ids.nextSetBit(id + 1)) {
            if (isSpecial(vocabulary, id)) {
                String text = tokenizer.decode(IntSequence.of(id));
                if (!text.isEmpty()) tokens.merge(text, id, Math::min);
            }
        }
        return tokens.isEmpty() ? NONE : new SpecialsImpl(tokens);
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
            if (!chars.starts(text.charAt(at))) continue; // most positions: one bit test
            // the longest special token starting at `at`: the deepest node on the walk ending one
            int id = -1, end = at;
            Node node = chars.root;
            for (int i = at; i < n && (node = node.child(text.charAt(i))) != null; i++) {
                if (node.id >= 0) {
                    id = node.id;
                    end = i + 1;
                }
            }
            if (id < 0) continue;
            if (at > gap) tokenizer.encodeInto(text, gap, at, out);
            out.add(id);
            gap = end;
            at = end - 1;
        }
        if (gap < n) tokenizer.encodeInto(text, gap, n, out);
    }

    @Override
    public void encodeUtf8Into(
            Tokenizer tokenizer, byte[] utf8, int from, int to, IntSequence.Builder out) {
        Objects.requireNonNull(tokenizer, "tokenizer");
        Objects.requireNonNull(utf8, "utf8");
        Objects.requireNonNull(out, "out");
        Objects.checkFromToIndex(from, to, utf8.length);
        int gap = from;
        for (int at = from; at < to; at++) {
            if (!bytes.starts(utf8[at] & 0xFF)) continue;
            int id = -1, end = at;
            Node node = bytes.root;
            for (int i = at; i < to && (node = node.child(utf8[i] & 0xFF)) != null; i++) {
                if (node.id >= 0) {
                    id = node.id;
                    end = i + 1;
                }
            }
            if (id < 0) continue;
            if (at > gap) tokenizer.encodeUtf8Into(utf8, gap, at, out);
            out.add(id);
            gap = end;
            at = end - 1;
        }
        if (gap < to) tokenizer.encodeUtf8Into(utf8, gap, to, out);
    }

    /** A trie and a bit per unit that begins some special token. */
    private static final class Trie {
        final Node root;
        final long[] first;

        Trie(Node root, int units) {
            this.root = root.freeze();
            this.first = new long[units >>> 6];
            for (int unit : this.root.keys) first[unit >>> 6] |= 1L << unit;
        }

        boolean starts(int unit) {
            return (first[unit >>> 6] & (1L << unit)) != 0;
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
            for (int unit : units) node = node.building.computeIfAbsent(unit, u -> new Node());
            node.id = tokenId;
        }

        Node freeze() {
            keys = building.keySet().stream().mapToInt(Integer::intValue).toArray();
            children = building.values().stream().map(Node::freeze).toArray(Node[]::new);
            building = null;
            return this;
        }

        Node child(int unit) {
            int[] k = keys;
            if (k.length <= 8) { // ponytail: most nodes have one child; a search for the rest
                for (int i = 0; i < k.length; i++) if (k[i] == unit) return children[i];
                return null;
            }
            int i = Arrays.binarySearch(k, unit);
            return i >= 0 ? children[i] : null;
        }
    }
}
