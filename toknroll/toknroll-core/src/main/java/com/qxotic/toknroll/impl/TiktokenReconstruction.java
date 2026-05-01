package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.ByteLevel;
import com.qxotic.toknroll.Toknroll;
import com.qxotic.toknroll.Vocabulary;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/** Internal helpers to reconstruct merges from tiktoken mergeable ranks. */
final class TiktokenReconstruction {

    private TiktokenReconstruction() {}

    public static Vocabulary vocabulary(
            Map<String, Integer> mergeableRanks, Map<String, Integer> specialTokens) {
        Objects.requireNonNull(mergeableRanks, "mergeableRanks");
        Objects.requireNonNull(specialTokens, "specialTokens");

        Map<String, Integer> tokenToId = new LinkedHashMap<>(mergeableRanks);
        Set<Integer> usedIds = new HashSet<>(mergeableRanks.values());
        int nextId = maxId(mergeableRanks.values()) + 1;

        for (Map.Entry<String, Integer> entry : specialTokens.entrySet()) {
            String token = entry.getKey();
            Integer requestedId = entry.getValue();
            if (token == null || requestedId == null) {
                continue;
            }

            int id = requestedId;
            if (id < 0 || (usedIds.contains(id) && !token.equals(tokenForId(tokenToId, id)))) {
                while (usedIds.contains(nextId)) {
                    nextId++;
                }
                id = nextId++;
            }

            tokenToId.put(token, id);
            usedIds.add(id);
        }

        return new VocabularyImpl(tokenToId);
    }

    public static List<Toknroll.MergeRule> mergeRules(Map<String, Integer> mergeableRanks) {
        Objects.requireNonNull(mergeableRanks, "mergeableRanks");

        List<Map.Entry<String, Integer>> entries = new ArrayList<>(mergeableRanks.entrySet());
        entries.sort(Comparator.comparingInt(Map.Entry::getValue));

        List<Toknroll.MergeRule> merges = new ArrayList<>();
        int normalizedRank = 0;
        for (Map.Entry<String, Integer> entry : entries) {
            String token = entry.getKey();
            int maxRank = entry.getValue();
            if (token.length() <= 1) {
                continue;
            }
            List<String> split = bpeSplit(mergeableRanks, token, maxRank);
            if (split.size() != 2) {
                continue;
            }
            Integer left = mergeableRanks.get(split.get(0));
            Integer right = mergeableRanks.get(split.get(1));
            if (left == null || right == null) {
                continue;
            }
            merges.add(Toknroll.MergeRule.of(left, right, normalizedRank));
            normalizedRank++;
        }
        return merges;
    }

    static LongLongMap packTiktokenMerges(Vocabulary vocabulary, List<Toknroll.MergeRule> merges) {
        return packMerges(vocabulary, merges, false);
    }

    static LongLongMap packSentencePieceMerges(
            Vocabulary vocabulary, List<Toknroll.MergeRule> merges) {
        return packMerges(vocabulary, merges, true);
    }

    private static LongLongMap packMerges(
            Vocabulary vocabulary, List<Toknroll.MergeRule> merges, boolean sentencePiece) {
        Objects.requireNonNull(vocabulary, "vocabulary");
        Objects.requireNonNull(merges, "merges");
        if (merges.isEmpty()) {
            throw new IllegalArgumentException("merges must not be empty");
        }

        long[] keys = new long[merges.size()];
        long[] values = new long[merges.size()];
        int size = 0;

        Map<ByteArrayKey, Integer> tokenIdByByteSurface =
                sentencePiece ? Map.of() : buildByteSurfaceIndex(vocabulary);

        Set<Long> pairKeys = new HashSet<>(merges.size() * 2);
        Set<Integer> ranks = new HashSet<>(merges.size() * 2);
        int maxRank = -1;

        for (int i = 0; i < merges.size(); i++) {
            Toknroll.MergeRule rule = merges.get(i);
            if (rule == null) {
                throw new NullPointerException("merges[" + i + "]");
            }
            int leftId = rule.leftId();
            int rightId = rule.rightId();
            int rank = rule.rank();

            if (rank < 0) {
                throw new IllegalArgumentException("Merge rank must be >= 0: " + rank);
            }
            if (!vocabulary.contains(leftId)) {
                throw new IllegalArgumentException("Unknown left token id: " + leftId);
            }
            if (!vocabulary.contains(rightId)) {
                throw new IllegalArgumentException("Unknown right token id: " + rightId);
            }

            long pairKey = IntPair.of(leftId, rightId);
            if (!pairKeys.add(pairKey)) {
                throw new IllegalArgumentException(
                        "Duplicate merge pair: (" + leftId + ", " + rightId + ")");
            }
            if (!ranks.add(rank)) {
                throw new IllegalArgumentException("Duplicate merge rank: " + rank);
            }

            String leftToken = vocabulary.token(leftId);
            String rightToken = vocabulary.token(rightId);
            String mergedToken = leftToken + rightToken;
            int mergedId;
            if (vocabulary.contains(mergedToken)) {
                mergedId = vocabulary.id(mergedToken);
            } else if (!sentencePiece) {
                mergedId = mergedIdByByteSurface(tokenIdByByteSurface, leftToken, rightToken);
                if (mergedId < 0) {
                    throw new IllegalArgumentException(
                            "Merged token not found in vocabulary for pair ("
                                    + leftId
                                    + ", "
                                    + rightId
                                    + "): '"
                                    + mergedToken
                                    + "'");
                }
            } else {
                throw new IllegalArgumentException(
                        "Merged token not found in vocabulary for pair ("
                                + leftId
                                + ", "
                                + rightId
                                + "): '"
                                + mergedToken
                                + "'");
            }
            keys[size] = pairKey;
            values[size] =
                    sentencePiece
                            ? SentencePieceBpeModel.packMerge(rank, mergedId)
                            : IntPair.of(mergedId, rank);
            size++;

            if (rank > maxRank) {
                maxRank = rank;
            }
        }

        for (int rank = 0; rank <= maxRank; rank++) {
            if (!ranks.contains(rank)) {
                throw new IllegalArgumentException("Missing merge rank: " + rank);
            }
        }

        if (size < keys.length) {
            keys = Arrays.copyOf(keys, size);
            values = Arrays.copyOf(values, size);
        }
        return new LongLongMap(keys, values);
    }

    private static Map<ByteArrayKey, Integer> buildByteSurfaceIndex(Vocabulary vocabulary) {
        Map<ByteArrayKey, Integer> out = new HashMap<>();
        for (Map.Entry<String, Integer> entry : vocabulary) {
            byte[] bytes = tokenBytesLikeModel(entry.getKey());
            ByteArrayKey key = new ByteArrayKey(bytes);
            Integer existing = out.get(key);
            if (existing == null || entry.getValue() < existing) {
                out.put(key, entry.getValue());
            }
        }
        return out;
    }

    private static int mergedIdByByteSurface(
            Map<ByteArrayKey, Integer> tokenIdByByteSurface, String leftToken, String rightToken) {
        byte[] left = tokenBytesLikeModel(leftToken);
        byte[] right = tokenBytesLikeModel(rightToken);
        byte[] merged = Arrays.copyOf(left, left.length + right.length);
        System.arraycopy(right, 0, merged, left.length, right.length);
        Integer id = tokenIdByByteSurface.get(new ByteArrayKey(merged));
        return id == null ? -1 : id;
    }

    private static byte[] tokenBytesLikeModel(String token) {
        try {
            return ByteLevel.decode(token);
        } catch (RuntimeException ignored) {
            // Non-byte-level token (e.g., raw UTF-8 "hello" in an SP-BPE vocab).
            // Fall back to raw UTF-8 bytes.
            return token.getBytes(StandardCharsets.UTF_8);
        }
    }

    private static final class ByteArrayKey {
        private final byte[] bytes;
        private final int hash;

        private ByteArrayKey(byte[] bytes) {
            this.bytes = bytes;
            this.hash = Arrays.hashCode(bytes);
        }

        @Override
        public int hashCode() {
            return hash;
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            }
            if (!(other instanceof ByteArrayKey)) {
                return false;
            }
            ByteArrayKey o = (ByteArrayKey) other;
            return Arrays.equals(bytes, o.bytes);
        }
    }

    private static String tokenForId(Map<String, Integer> tokenToId, int id) {
        for (Map.Entry<String, Integer> entry : tokenToId.entrySet()) {
            if (entry.getValue() == id) {
                return entry.getKey();
            }
        }
        return null;
    }

    private static int maxId(Iterable<Integer> ids) {
        int max = -1;
        for (Integer id : ids) {
            if (id != null && id > max) {
                max = id;
            }
        }
        return max;
    }

    private static List<String> bpeSplit(
            Map<String, Integer> mergeableRanks, String token, Integer maxRank) {
        byte[] tokenBytes = ByteLevel.decode(token);
        List<String> parts = new ArrayList<>(tokenBytes.length);
        for (byte b : tokenBytes) {
            parts.add(ByteLevel.encode(new byte[] {b}));
        }

        while (true) {
            Integer bestPos = null;
            Integer bestRank = null;
            String bestMerged = null;
            for (int i = 0; i + 1 < parts.size(); i++) {
                String merged = parts.get(i) + parts.get(i + 1);
                Integer rank = mergeableRanks.get(merged);
                if (rank != null && (bestRank == null || rank < bestRank)) {
                    bestPos = i;
                    bestRank = rank;
                    bestMerged = merged;
                }
            }

            if (bestRank == null || (maxRank != null && bestRank >= maxRank)) {
                break;
            }

            List<String> newParts = new ArrayList<>(parts.size() - 1);
            newParts.addAll(parts.subList(0, bestPos));
            newParts.add(bestMerged);
            newParts.addAll(parts.subList(bestPos + 2, parts.size()));
            parts = newParts;
        }

        return parts;
    }
}
