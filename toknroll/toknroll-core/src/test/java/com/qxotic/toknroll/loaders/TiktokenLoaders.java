package com.qxotic.toknroll.loaders;

import com.qxotic.toknroll.MergeRule;
import com.qxotic.toknroll.Vocabulary;
import com.qxotic.toknroll.impl.ImplAccessor;
import com.qxotic.toknroll.testkit.TiktokenFiles;
import java.io.BufferedReader;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Public helpers for working with TikToken mergeable-ranks assets. */
public final class TiktokenLoaders {

    private TiktokenLoaders() {}

    /** Loads mergeable ranks from a local path or HTTP(S) URL. */
    public static Map<String, Integer> loadMergeableRanks(String blobPath, String expectedHash)
            throws IOException, InterruptedException {
        return TiktokenFiles.loadMergeableRanks(
                Objects.requireNonNull(blobPath, "blobPath"), expectedHash);
    }

    /** Loads mergeable ranks from an already-open reader. */
    public static Map<String, Integer> loadMergeableRanks(BufferedReader reader) {
        return ImplAccessor.loadTiktokenMergeableRanks(Objects.requireNonNull(reader, "reader"));
    }

    /** Reconstructs a vocabulary from Tiktoken ranks plus optional specials. */
    public static Vocabulary vocabulary(
            Map<String, Integer> mergeableRanks, Map<String, Integer> specialTokens) {
        return ImplAccessor.reconstructTiktokenVocabulary(
                Objects.requireNonNull(mergeableRanks, "mergeableRanks"),
                Objects.requireNonNull(specialTokens, "specialTokens"));
    }

    /** Reconstructs merge rules from Tiktoken mergeable ranks. */
    public static List<MergeRule> mergeRules(Map<String, Integer> mergeableRanks) {
        return ImplAccessor.reconstructTiktokenMergeRules(
                Objects.requireNonNull(mergeableRanks, "mergeableRanks"));
    }
}
