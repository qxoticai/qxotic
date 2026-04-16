package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.ByteLevel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Builder for prototype byte-level Aho-Corasick BPE model. */
public final class AcBpeBuilder {

    private static final int ALPHABET = 256;

    private AcBpeBuilder() {}

    /**
     * Builds a prototype AC model from a subset of mergeable ranks.
     *
     * <p>Includes all singleton-byte tokens and then up to {@code maxTokens} additional tokens by
     * increasing rank, filtered by {@code maxTokenBytes}.
     */
    public static AcBpeModel buildPrototype(
            Map<String, Integer> mergeableRanks, int maxTokens, int maxTokenBytes) {
        Objects.requireNonNull(mergeableRanks, "mergeableRanks");
        if (maxTokens < 256) {
            throw new IllegalArgumentException("maxTokens must be >= 256");
        }
        if (maxTokenBytes <= 0) {
            throw new IllegalArgumentException("maxTokenBytes must be > 0");
        }

        int[] singleByteTokenId = new int[256];
        List<TokenEntry> entries =
                selectEntries(mergeableRanks, maxTokens, maxTokenBytes, singleByteTokenId);
        Trie trie = buildTrie(entries);
        buildFailAndCompleteTransitions(trie);
        return toModel(trie, singleByteTokenId);
    }

    private static List<TokenEntry> selectEntries(
            Map<String, Integer> mergeableRanks,
            int maxTokens,
            int maxTokenBytes,
            int[] singleByteTokenId) {
        Arrays.fill(singleByteTokenId, -1);
        List<TokenEntry> singletons = new ArrayList<>(256);
        List<TokenEntry> others = new ArrayList<>(Math.max(1024, maxTokens));

        for (Map.Entry<String, Integer> e : mergeableRanks.entrySet()) {
            byte[] bytes = ByteLevel.decode(e.getKey());
            if (bytes.length == 1) {
                singletons.add(new TokenEntry(bytes, e.getValue()));
                singleByteTokenId[bytes[0] & 0xFF] = e.getValue();
            } else if (bytes.length <= maxTokenBytes) {
                others.add(new TokenEntry(bytes, e.getValue()));
            }
        }

        if (singletons.size() != 256) {
            throw new IllegalArgumentException(
                    "Expected 256 singleton-byte tokens, found " + singletons.size());
        }
        for (int i = 0; i < 256; i++) {
            if (singleByteTokenId[i] < 0) {
                throw new IllegalArgumentException("Missing singleton-byte token for byte=" + i);
            }
        }

        others.sort(Comparator.comparingInt(t -> t.tokenId));

        int remaining = Math.max(0, maxTokens - 256);
        int extra = Math.min(remaining, others.size());
        List<TokenEntry> selected = new ArrayList<>(256 + extra);
        selected.addAll(singletons);
        selected.addAll(others.subList(0, extra));
        return selected;
    }

    private static Trie buildTrie(List<TokenEntry> entries) {
        Trie trie = new Trie();
        trie.entries.addAll(entries);

        for (TokenEntry e : entries) {
            int state = 0;
            byte[] b = e.bytes;
            for (byte value : b) {
                int c = value & 0xFF;
                int idx = (state << 8) | c;
                int nextState = trie.next[idx];
                if (nextState == 0) {
                    nextState = trie.addState();
                    trie.next[idx] = nextState;
                }
                state = nextState;
            }
            trie.addOutput(state, e.tokenId, b.length);
        }

        return trie;
    }

    private static void buildFailAndCompleteTransitions(Trie trie) {
        int[] q = new int[Math.max(16, trie.stateCount)];
        int qh = 0;
        int qt = 0;

        for (int c = 0; c < ALPHABET; c++) {
            int s = trie.next[c];
            if (s != 0) {
                q[qt++] = s;
            }
        }

        while (qh < qt) {
            int state = q[qh++];
            if (qt + ALPHABET >= q.length) {
                q = Arrays.copyOf(q, q.length << 1);
            }

            int row = state << 8;
            for (int c = 0; c < ALPHABET; c++) {
                int s = trie.next[row | c];
                if (s != 0) {
                    int f = trie.fail[state];
                    while (f != 0 && trie.next[(f << 8) | c] == 0) {
                        f = trie.fail[f];
                    }
                    int fs = trie.next[(f << 8) | c];
                    trie.fail[s] = fs;
                    trie.mergeOutput(s, fs);
                    q[qt++] = s;
                } else {
                    trie.next[row | c] = trie.next[(trie.fail[state] << 8) | c];
                }
            }
        }
    }

    private static AcBpeModel toModel(Trie trie, int[] singleByteTokenId) {
        int maxTokenId = -1;
        int maxTokenLen = 1;
        for (int i = 0; i < trie.outSize; i++) {
            if (trie.outToken[i] > maxTokenId) {
                maxTokenId = trie.outToken[i];
            }
            int len = trie.outLen[i] & 0xFFFF;
            if (len > maxTokenLen) {
                maxTokenLen = len;
            }
        }

        int[] nextPrefix = new int[Math.max(0, maxTokenId + 1)];
        short[] tokenLen = new short[Math.max(0, maxTokenId + 1)];
        Arrays.fill(nextPrefix, -1);

        HashMap<BytesKey, Integer> exact = new HashMap<>();
        for (TokenEntry e : trie.entries) {
            exact.put(new BytesKey(e.bytes), e.tokenId);
            if (e.tokenId >= 0 && e.tokenId < tokenLen.length) {
                tokenLen[e.tokenId] = (short) e.bytes.length;
            }
        }
        for (TokenEntry e : trie.entries) {
            if (e.bytes.length <= 1 || e.tokenId < 0 || e.tokenId >= nextPrefix.length) {
                continue;
            }
            byte[] prefix = Arrays.copyOf(e.bytes, e.bytes.length - 1);
            Integer p = exact.get(new BytesKey(prefix));
            nextPrefix[e.tokenId] = (p == null) ? -1 : p;
        }

        return new AcBpeModel(
                trie.stateCount,
                trie.next,
                trie.outHead,
                Arrays.copyOf(trie.outNext, trie.outSize),
                Arrays.copyOf(trie.outToken, trie.outSize),
                Arrays.copyOf(trie.outLen, trie.outSize),
                singleByteTokenId,
                nextPrefix,
                tokenLen,
                maxTokenLen);
    }

    private static final class TokenEntry {
        final byte[] bytes;
        final int tokenId;

        TokenEntry(byte[] bytes, int tokenId) {
            this.bytes = bytes;
            this.tokenId = tokenId;
        }
    }

    private static final class BytesKey {
        final byte[] bytes;
        final int hash;

        BytesKey(byte[] bytes) {
            this.bytes = bytes;
            this.hash = Arrays.hashCode(bytes);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (!(o instanceof BytesKey)) {
                return false;
            }
            BytesKey other = (BytesKey) o;
            return Arrays.equals(bytes, other.bytes);
        }

        @Override
        public int hashCode() {
            return hash;
        }
    }

    private static final class Trie {
        final List<TokenEntry> entries = new ArrayList<>();
        int stateCount = 1; // root = 0
        int[] next = new int[ALPHABET * 16];
        int[] fail = new int[16];

        int[] outHead = new int[16];
        int[] outNext = new int[256];
        int[] outToken = new int[256];
        short[] outLen = new short[256];
        int outSize;

        Trie() {
            Arrays.fill(outHead, -1);
            Arrays.fill(outNext, -1);
        }

        int addState() {
            int s = stateCount++;
            ensureStateCapacity(stateCount);
            outHead[s] = -1;
            return s;
        }

        void addOutput(int state, int tokenId, int length) {
            ensureOutputCapacity(outSize + 1);
            outToken[outSize] = tokenId;
            outLen[outSize] = (short) length;
            outNext[outSize] = outHead[state];
            outHead[state] = outSize;
            outSize++;
        }

        void mergeOutput(int state, int from) {
            int head = outHead[from];
            while (head != -1) {
                addOutput(state, outToken[head], outLen[head] & 0xFFFF);
                head = outNext[head];
            }
        }

        private void ensureStateCapacity(int neededStates) {
            if (fail.length < neededStates) {
                int newCap = grow(fail.length, neededStates);
                fail = Arrays.copyOf(fail, newCap);

                int oldOut = outHead.length;
                outHead = Arrays.copyOf(outHead, newCap);
                Arrays.fill(outHead, oldOut, newCap, -1);

                int requiredNext = newCap * ALPHABET;
                next = Arrays.copyOf(next, requiredNext);
            }
        }

        private void ensureOutputCapacity(int needed) {
            if (outToken.length < needed) {
                int newCap = grow(outToken.length, needed);
                outToken = Arrays.copyOf(outToken, newCap);
                outLen = Arrays.copyOf(outLen, newCap);
                int old = outNext.length;
                outNext = Arrays.copyOf(outNext, newCap);
                Arrays.fill(outNext, old, newCap, -1);
            }
        }

        private static int grow(int current, int needed) {
            int c = Math.max(16, current);
            while (c < needed) {
                c += c >>> 1;
            }
            return c;
        }
    }
}
