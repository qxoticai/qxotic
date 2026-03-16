package com.qxotic.model.llm.mistral;

import com.qxotic.tokenizers.impl.VocabularyImpl;

public class VocabularyImplWithScores extends VocabularyImpl {

    public float getScore(int tokenIndex) {
        return scores[tokenIndex];
    }

    public VocabularyImplWithScores(String[] vocabulary, float[] scores, int[] tokenTypes) {
        super(vocabulary, scores, tokenTypes);
    }
}
