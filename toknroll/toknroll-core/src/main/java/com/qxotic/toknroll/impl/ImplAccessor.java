package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.TokenizationModel;
import com.qxotic.toknroll.Tokenizers;
import com.qxotic.toknroll.Vocabulary;
import java.util.List;
import java.util.Map;

public final class ImplAccessor {

    private static final int[] EMPTY_ARRAY = new int[0];
    private static final IntSequence EMPTY_SEQUENCE = wrap(EMPTY_ARRAY);

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
}
