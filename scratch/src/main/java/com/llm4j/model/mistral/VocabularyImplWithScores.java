package com.llm4j.model.mistral;

import com.llm4j.tokenizers.impl.VocabularyImpl;

public class VocabularyImplWithScores extends VocabularyImpl {

    public float getScore(int tokenIndex) {
        return scores[tokenIndex];
    }

    public VocabularyImplWithScores(String[] vocabulary, float[] scores, int[] tokenTypes) {
        super(vocabulary, scores, tokenTypes);
    }
}
