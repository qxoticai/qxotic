package com.qxotic.toknroll.impl;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.Toknroll;
import com.qxotic.toknroll.Vocabulary;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class TiktokenReconstructionTest {

    // ------------------------------------------------------------------
    // vocabulary
    // ------------------------------------------------------------------

    @Test
    void vocabularyBasicMergeableRanks() {
        Map<String, Integer> mergeableRanks = new LinkedHashMap<>();
        mergeableRanks.put("a", 0);
        mergeableRanks.put("b", 1);
        mergeableRanks.put("ab", 2);

        Vocabulary vocab = TiktokenReconstruction.vocabulary(mergeableRanks, Map.of());

        assertEquals(3, vocab.size());
        assertEquals(0, vocab.id("a"));
        assertEquals(1, vocab.id("b"));
        assertEquals(2, vocab.id("ab"));
    }

    @Test
    void vocabularyWithSpecialTokens() {
        Map<String, Integer> mergeableRanks = new LinkedHashMap<>();
        mergeableRanks.put("a", 0);
        mergeableRanks.put("b", 1);

        Map<String, Integer> specialTokens = new LinkedHashMap<>();
        specialTokens.put("<|endoftext|>", 2);

        Vocabulary vocab = TiktokenReconstruction.vocabulary(mergeableRanks, specialTokens);

        assertEquals(3, vocab.size());
        assertEquals(2, vocab.id("<|endoftext|>"));
    }

    @Test
    void vocabularySpecialTokenIdConflictReassigns() {
        Map<String, Integer> mergeableRanks = new LinkedHashMap<>();
        mergeableRanks.put("a", 0);

        Map<String, Integer> specialTokens = new LinkedHashMap<>();
        specialTokens.put("<|special|>", 0); // conflicts with "a"

        Vocabulary vocab = TiktokenReconstruction.vocabulary(mergeableRanks, specialTokens);

        assertEquals(2, vocab.size());
        assertTrue(vocab.contains(vocab.id("<|special|>")));
    }

    @Test
    void vocabularySpecialTokenNegativeIdReassigns() {
        Map<String, Integer> mergeableRanks = new LinkedHashMap<>();
        mergeableRanks.put("a", 0);

        Map<String, Integer> specialTokens = new LinkedHashMap<>();
        specialTokens.put("<|special|>", -1);

        Vocabulary vocab = TiktokenReconstruction.vocabulary(mergeableRanks, specialTokens);

        assertEquals(2, vocab.size());
        assertTrue(vocab.contains(vocab.id("<|special|>")));
    }

    @Test
    void vocabularySkipNullSpecialTokenKeyOrValue() {
        Map<String, Integer> mergeableRanks = new LinkedHashMap<>();
        mergeableRanks.put("a", 0);

        Map<String, Integer> specialTokens = new LinkedHashMap<>();
        specialTokens.put(null, 5);
        specialTokens.put("b", null);

        // Should not throw
        Vocabulary vocab = TiktokenReconstruction.vocabulary(mergeableRanks, specialTokens);

        assertEquals(1, vocab.size());
        assertEquals(0, vocab.id("a"));
    }

    @Test
    void vocabularyNullMergeableRanksThrows() {
        assertThrows(
                NullPointerException.class,
                () -> TiktokenReconstruction.vocabulary(null, Map.of()));
    }

    @Test
    void vocabularyNullSpecialTokensThrows() {
        assertThrows(
                NullPointerException.class,
                () -> TiktokenReconstruction.vocabulary(Map.of(), null));
    }

    // ------------------------------------------------------------------
    // mergeRules
    // ------------------------------------------------------------------

    @Test
    void mergeRulesBasic() {
        Map<String, Integer> mergeableRanks = new LinkedHashMap<>();
        mergeableRanks.put("a", 0);
        mergeableRanks.put("b", 1);
        mergeableRanks.put("ab", 2);

        List<Toknroll.MergeRule> rules = TiktokenReconstruction.mergeRules(mergeableRanks);

        assertFalse(rules.isEmpty());
        // "ab" merges from "a"(0) + "b"(1) with the rank equal to "ab"'s entry (2)
        Toknroll.MergeRule abRule = rules.get(0);
        assertEquals(0, abRule.leftId());
        assertEquals(1, abRule.rightId());
    }

    @Test
    void mergeRulesEmptyMergesReturnsEmpty() {
        Map<String, Integer> mergeableRanks = new LinkedHashMap<>();
        mergeableRanks.put("a", 0);

        List<Toknroll.MergeRule> rules = TiktokenReconstruction.mergeRules(mergeableRanks);

        // Single-char tokens don't produce merge rules
        assertTrue(rules.isEmpty());
    }

    @Test
    void mergeRulesNullInputThrows() {
        assertThrows(NullPointerException.class, () -> TiktokenReconstruction.mergeRules(null));
    }

    // ------------------------------------------------------------------
    // packTiktokenMerges (package-private, test from same package)
    // ------------------------------------------------------------------

    @Test
    void packTiktokenMergesEmptyMergesThrows() {
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        TiktokenReconstruction.packTiktokenMerges(
                                new VocabularyImpl(new String[] {"a"}, normalTypes(1)), List.of()));
    }

    @Test
    void packTiktokenMergesNullVocabularyThrows() {
        assertThrows(
                NullPointerException.class,
                () ->
                        TiktokenReconstruction.packTiktokenMerges(
                                null, List.of(Toknroll.MergeRule.of(0, 1, 0))));
    }

    @Test
    void packTiktokenMergesNullMergesThrows() {
        assertThrows(
                NullPointerException.class,
                () ->
                        TiktokenReconstruction.packTiktokenMerges(
                                new VocabularyImpl(new String[] {"a"}, normalTypes(1)), null));
    }

    // ------------------------------------------------------------------
    // packSentencePieceMerges error paths
    // ------------------------------------------------------------------

    @Test
    void packSentencePieceMergesEmptyMergesThrows() {
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        TiktokenReconstruction.packSentencePieceMerges(
                                new VocabularyImpl(new String[] {"a"}, normalTypes(1)), List.of()));
    }

    @Test
    void packSentencePieceMergesDuplicatePairThrows() {
        VocabularyImpl vocab =
                new VocabularyImpl(new String[] {"a", "b", "ab", "ba"}, normalTypes(4));
        List<Toknroll.MergeRule> merges =
                List.of(
                        Toknroll.MergeRule.of(0, 1, 0),
                        Toknroll.MergeRule.of(0, 1, 1) // duplicate pair
                        );
        assertThrows(
                IllegalArgumentException.class,
                () -> TiktokenReconstruction.packSentencePieceMerges(vocab, merges));
    }

    @Test
    void packSentencePieceMergesDuplicateRankThrows() {
        VocabularyImpl vocab =
                new VocabularyImpl(new String[] {"a", "b", "ab", "ba"}, normalTypes(4));
        List<Toknroll.MergeRule> merges =
                List.of(
                        Toknroll.MergeRule.of(0, 1, 0),
                        Toknroll.MergeRule.of(1, 0, 0) // duplicate rank
                        );
        assertThrows(
                IllegalArgumentException.class,
                () -> TiktokenReconstruction.packSentencePieceMerges(vocab, merges));
    }

    @Test
    void packSentencePieceMergesMissingRankThrows() {
        VocabularyImpl vocab = new VocabularyImpl(new String[] {"a", "b", "ab"}, normalTypes(3));
        List<Toknroll.MergeRule> merges =
                List.of(
                        Toknroll.MergeRule.of(0, 1, 3) // rank 3 but 2 is missing
                        );
        assertThrows(
                IllegalArgumentException.class,
                () -> TiktokenReconstruction.packSentencePieceMerges(vocab, merges));
    }

    @Test
    void packSentencePieceMergesNegativeRankThrows() {
        VocabularyImpl vocab = new VocabularyImpl(new String[] {"a", "b", "ab"}, normalTypes(3));
        List<Toknroll.MergeRule> merges = List.of(Toknroll.MergeRule.of(0, 1, -1));
        assertThrows(
                IllegalArgumentException.class,
                () -> TiktokenReconstruction.packSentencePieceMerges(vocab, merges));
    }

    @Test
    void packSentencePieceMergesUnknownLeftIdThrows() {
        VocabularyImpl vocab = new VocabularyImpl(new String[] {"a", "b", "ab"}, normalTypes(3));
        List<Toknroll.MergeRule> merges = List.of(Toknroll.MergeRule.of(99, 1, 0));
        assertThrows(
                IllegalArgumentException.class,
                () -> TiktokenReconstruction.packSentencePieceMerges(vocab, merges));
    }

    @Test
    void packSentencePieceMergesUnknownRightIdThrows() {
        VocabularyImpl vocab = new VocabularyImpl(new String[] {"a", "b", "ab"}, normalTypes(3));
        List<Toknroll.MergeRule> merges = List.of(Toknroll.MergeRule.of(0, 99, 0));
        assertThrows(
                IllegalArgumentException.class,
                () -> TiktokenReconstruction.packSentencePieceMerges(vocab, merges));
    }

    @Test
    void packSentencePieceMergesMissingMergedTokenThrows() {
        VocabularyImpl vocab = new VocabularyImpl(new String[] {"a", "b"}, normalTypes(2));
        // "ab" not in vocabulary, so merge (a,b) has no merged token
        List<Toknroll.MergeRule> merges = List.of(Toknroll.MergeRule.of(0, 1, 0));
        assertThrows(
                IllegalArgumentException.class,
                () -> TiktokenReconstruction.packSentencePieceMerges(vocab, merges));
    }

    @Test
    void packSentencePieceMergesNullMergeRuleThrows() {
        VocabularyImpl vocab = new VocabularyImpl(new String[] {"a", "b", "ab"}, normalTypes(3));
        List<Toknroll.MergeRule> merges = new java.util.ArrayList<>();
        merges.add(null);
        assertThrows(
                NullPointerException.class,
                () -> TiktokenReconstruction.packSentencePieceMerges(vocab, merges));
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private static int[] normalTypes(int length) {
        int[] types = new int[length];
        for (int i = 0; i < length; i++) {
            types[i] = com.qxotic.toknroll.StandardTokenType.NORMAL.getId();
        }
        return types;
    }
}
