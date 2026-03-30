package com.qxotic.tokenizers.impl;

import com.qxotic.tokenizers.Tokenizer;
import com.qxotic.tokenizers.Vocabulary;
import com.qxotic.tokenizers.advanced.Normalizer;
import com.qxotic.tokenizers.advanced.Splitter;
import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class ClassicBPE {

    private static List<String> bpe(
            Map<String, Integer> mergeableRanks, String token, Integer maxRank) {
        List<String> parts = new ArrayList<>();
        for (int i = 0; i < token.length(); i++) {
            char ch = token.charAt(i);
            parts.add(String.valueOf(ch));
        }

        while (true) {
            Integer minIdx = null;
            Integer minRank = null;
            String mergedPair = null;
            for (int i = 0; i < parts.size() - 1; i++) {
                String merged = parts.get(i) + parts.get(i + 1);
                Integer rank = mergeableRanks.get(merged);
                if (rank != null && (minRank == null || rank < minRank)) {
                    minIdx = i;
                    minRank = rank;
                    mergedPair = merged;
                }
            }

            if (minRank == null || (maxRank != null && minRank >= maxRank)) {
                break;
            }

            List<String> newParts = new ArrayList<>(parts.subList(0, minIdx));
            newParts.add(mergedPair);
            newParts.addAll(parts.subList(minIdx + 2, parts.size()));
            parts = newParts;
        }

        return parts;
    }

    private static LongLongMap buildMerges(Map<String, Integer> mergeableRanks) {
        List<Map.Entry<String, Integer>> entries = new ArrayList<>(mergeableRanks.entrySet());
        entries.sort(Comparator.comparingInt(Map.Entry::getValue));

        // Collect keys and values for bulk construction.
        List<long[]> collected = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : entries) {
            if (entry.getKey().length() == 1) {
                continue;
            }
            List<String> result = bpe(mergeableRanks, entry.getKey(), entry.getValue());
            assert result.size() == 2;
            int left = mergeableRanks.get(result.get(0));
            int right = mergeableRanks.get(result.get(1));
            int mergeIndex = mergeableRanks.get(entry.getKey());
            collected.add(
                    new long[] {IntPair.of(left, right), IntPair.of(mergeIndex, entry.getValue())});
        }

        long[] keys = new long[collected.size()];
        long[] values = new long[collected.size()];
        for (int i = 0; i < collected.size(); i++) {
            keys[i] = collected.get(i)[0];
            values[i] = collected.get(i)[1];
        }
        return new LongLongMap(keys, values);
    }

    public static Map<String, Integer> loadMergeableRanks(String blobPath, String expectedHash)
            throws IOException, InterruptedException {
        byte[] bytes = FileCache.readFileCached(blobPath, expectedHash);
        return Tiktoken.loadMergeableRanks(
                new BufferedReader(new InputStreamReader(new ByteArrayInputStream(bytes))));
    }

    public static Tokenizer classicFromTiktoken(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            Normalizer normalizer,
            Splitter splitter) {
        LongLongMap merges = buildMerges(mergeableRanks);
        Vocabulary vocabulary =
                VocabularyWithSpecials.create(new VocabularyImpl(mergeableRanks), specialTokens);
        return new GPT2Tokenizer(vocabulary, normalizer, splitter, merges);
    }
}
