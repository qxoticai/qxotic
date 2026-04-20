package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Specials;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.TokenizationModel;
import com.qxotic.toknroll.Tokenizers;
import com.qxotic.toknroll.Vocabulary;
import java.io.BufferedReader;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Public bridge to package-private implementation details.
 *
 * <p>API policy:
 *
 * <ul>
 *   <li>Methods used by public types in {@code com.qxotic.toknroll} are the stable bridge.
 *   <li>Methods marked {@code @Deprecated} are test/benchmark bridges and may change without
 *       compatibility guarantees.
 * </ul>
 */
public final class ImplAccessor {

    private static final int[] EMPTY_ARRAY = new int[0];
    private static final IntSequence EMPTY_SEQUENCE = wrap(EMPTY_ARRAY);

    private ImplAccessor() {}

    public static IntSequence empty() {
        return EMPTY_SEQUENCE;
    }

    public static IntSequence.Builder newBuilder() {
        return new IntSequenceBuilder();
    }

    public static IntSequence.Builder newBuilder(int initialCapacity) {
        return new IntSequenceBuilder(initialCapacity);
    }

    public static IntSequence wrap(int[] array) {
        return new ArrayIntSequence(array);
    }

    public static IntSequence wrap(List<Integer> list) {
        return new ListIntSequence(list);
    }

    public static Vocabulary createVocabularyWithSpecials(
            Map<String, Integer> tokenToId, Map<String, Integer> specialTokens) {
        return VocabularyWithSpecials.create(new VocabularyImpl(tokenToId), specialTokens);
    }

    public static Vocabulary createVocabulary(Map<String, Integer> tokenToId) {
        return new VocabularyImpl(tokenToId);
    }

    /**
     * Test-only bridge for GGUF fixtures that include token-type metadata.
     *
     * @deprecated Prefer public vocabulary constructors/factories in {@code Tokenizers} for
     *     production usage.
     */
    @Deprecated(forRemoval = false, since = "0.1.0")
    public static Vocabulary createVocabulary(String[] tokens, int[] tokenTypes) {
        return new VocabularyImpl(tokens, tokenTypes);
    }

    public static TokenizationModel createTikTokenModel(
            Vocabulary vocabulary, List<Tokenizers.MergeRule> merges, boolean ignoreMerges) {
        LongLongMap packed =
                merges.isEmpty()
                        ? new LongLongMap(new long[0], new long[0])
                        : TiktokenReconstruction.packTikTokenMerges(vocabulary, merges);
        return TikTokenModel.fromVocabularyAndMerges(vocabulary, packed, ignoreMerges);
    }

    public static TokenizationModel createSentencePieceBpeModel(
            Vocabulary vocabulary, List<Tokenizers.MergeRule> merges) {
        LongLongMap packed =
                merges.isEmpty()
                        ? new LongLongMap(new long[0], new long[0])
                        : TiktokenReconstruction.packSentencePieceMerges(vocabulary, merges);
        return SentencePieceBpeModel.fromVocabularyAndMerges(vocabulary, packed);
    }

    public static TokenizationModel createSentencePieceBpeModel(
            Vocabulary vocabulary, float[] scores) {
        return SentencePieceBpeModel.fromVocabulary(vocabulary, scores);
    }

    public static Splitter createRegexSplitter(Pattern pattern) {
        return RegexSplitter.create(pattern);
    }

    public static Specials specialsNone() {
        return SpecialsImpl.none();
    }

    public static Specials compileSpecials(Vocabulary vocabulary, Set<String> specials) {
        return SpecialsImpl.compile(vocabulary, specials);
    }

    /**
     * Test fixture bridge for loading local/remote tiktoken mergeable ranks.
     *
     * @deprecated Prefer dedicated public loader APIs once exposed.
     */
    @Deprecated(forRemoval = false, since = "0.1.0")
    public static Map<String, Integer> loadMergeableRanks(String blobPath, String expectedHash)
            throws IOException, InterruptedException {
        return loadTikTokenMergeableRanks(blobPath, expectedHash);
    }

    /**
     * Test fixture bridge for loading tiktoken mergeable ranks from a reader.
     *
     * @deprecated Prefer dedicated public loader APIs once exposed.
     */
    @Deprecated(forRemoval = false, since = "0.1.0")
    public static Map<String, Integer> loadMergeableRanks(BufferedReader reader) {
        return loadTikTokenMergeableRanks(reader);
    }

    /**
     * Test fixture bridge for reconstructing vocabulary from tiktoken ranks.
     *
     * @deprecated Prefer dedicated public loader APIs once exposed.
     */
    @Deprecated(forRemoval = false, since = "0.1.0")
    public static Vocabulary tiktokenVocabulary(
            Map<String, Integer> mergeableRanks, Map<String, Integer> specialTokens) {
        return reconstructTikTokenVocabulary(mergeableRanks, specialTokens);
    }

    /**
     * Test fixture bridge for reconstructing merge rules from tiktoken ranks.
     *
     * @deprecated Prefer dedicated public loader APIs once exposed.
     */
    @Deprecated(forRemoval = false, since = "0.1.0")
    public static List<Tokenizers.MergeRule> tiktokenMergeRules(
            Map<String, Integer> mergeableRanks) {
        return reconstructTikTokenMergeRules(mergeableRanks);
    }

    public static Map<String, Integer> loadTikTokenMergeableRanks(
            String blobPath, String expectedHash) throws IOException, InterruptedException {
        return TiktokenFiles.loadMergeableRanks(blobPath, expectedHash);
    }

    public static Map<String, Integer> loadTikTokenMergeableRanks(BufferedReader reader) {
        return TiktokenFiles.loadMergeableRanks(reader);
    }

    public static Vocabulary reconstructTikTokenVocabulary(
            Map<String, Integer> mergeableRanks, Map<String, Integer> specialTokens) {
        return TiktokenReconstruction.vocabulary(mergeableRanks, specialTokens);
    }

    public static List<Tokenizers.MergeRule> reconstructTikTokenMergeRules(
            Map<String, Integer> mergeableRanks) {
        return TiktokenReconstruction.mergeRules(mergeableRanks);
    }
}
