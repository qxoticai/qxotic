package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.Vocabulary;
import com.qxotic.toknroll.advanced.SymbolCodec;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Factory helpers for creating reusable generic BPE tokenizers. */
public final class GenericBPE {

    private GenericBPE() {}

    public static GenericBpeTokenizer fromTiktoken(
            Map<String, Integer> mergeableRanks, Map<String, Integer> specialTokens) {
        return fromTiktoken(mergeableRanks, specialTokens, SymbolCodec.BYTE_LEVEL);
    }

    public static GenericBpeTokenizer fromTiktoken(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            SymbolCodec symbolCodec) {
        Objects.requireNonNull(mergeableRanks, "mergeableRanks");
        Objects.requireNonNull(specialTokens, "specialTokens");
        Objects.requireNonNull(symbolCodec, "symbolCodec");

        BpeMergeTable mergeTable = new LongLongBpeMergeTable(buildMerges(mergeableRanks));
        Vocabulary vocabulary =
                VocabularyWithSpecials.create(new VocabularyImpl(mergeableRanks), specialTokens);
        BpeSymbolEncoder symbolEncoder =
                symbolCodec == SymbolCodec.BYTE_LEVEL
                        ? new DirectByteBpeSymbolEncoder(vocabulary)
                        : new CodecBpeSymbolEncoder(symbolCodec);
        return create(vocabulary, mergeTable, symbolEncoder);
    }

    public static GenericBpeTokenizer create(
            Vocabulary vocabulary, BpeMergeTable mergeTable, BpeSymbolEncoder symbolEncoder) {
        Objects.requireNonNull(vocabulary, "vocabulary");
        Objects.requireNonNull(mergeTable, "mergeTable");
        Objects.requireNonNull(symbolEncoder, "symbolEncoder");
        return new GenericBpeTokenizer(vocabulary, mergeTable, symbolEncoder);
    }

    private static LongLongMap buildMerges(Map<String, Integer> mergeableRanks) {
        List<Map.Entry<String, Integer>> entries = new ArrayList<>(mergeableRanks.entrySet());
        entries.sort(Comparator.comparingInt(Map.Entry::getValue));

        List<long[]> pairs = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : entries) {
            if (entry.getKey().length() <= 1) {
                continue;
            }
            List<String> split = bpeSplit(mergeableRanks, entry.getKey(), entry.getValue());
            if (split.size() != 2) {
                continue;
            }
            Integer left = mergeableRanks.get(split.get(0));
            Integer right = mergeableRanks.get(split.get(1));
            if (left == null || right == null) {
                continue;
            }

            int mergeIndex = mergeableRanks.get(entry.getKey());
            int rank = entry.getValue();
            pairs.add(new long[] {IntPair.of(left, right), IntPair.of(mergeIndex, rank)});
        }

        long[] keys = new long[pairs.size()];
        long[] values = new long[pairs.size()];
        for (int i = 0; i < pairs.size(); i++) {
            keys[i] = pairs.get(i)[0];
            values[i] = pairs.get(i)[1];
        }
        return new LongLongMap(keys, values);
    }

    private static List<String> bpeSplit(
            Map<String, Integer> mergeableRanks, String token, Integer maxRank) {
        List<String> parts = new ArrayList<>(token.length());
        for (int i = 0; i < token.length(); i++) {
            parts.add(String.valueOf(token.charAt(i)));
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
