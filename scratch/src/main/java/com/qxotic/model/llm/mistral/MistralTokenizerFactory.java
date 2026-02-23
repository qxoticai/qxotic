package com.qxotic.model.llm.mistral;

import com.google.auto.service.AutoService;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.model.llm.GGUFTokenizerFactory;
import com.qxotic.model.llm.TokenizerFactory;
import com.qxotic.tokenizers.Tokenizer;
import com.qxotic.tokenizers.advanced.Normalizer;
import com.qxotic.tokenizers.advanced.Splitter;

// Implementation for Mistral tokenizer
@AutoService(TokenizerFactory.class)
public class MistralTokenizerFactory implements GGUFTokenizerFactory {
    @Override
    public String getTokenizerName() {
        return "llama";
    }

    @Override
    public Tokenizer createTokenizer(GGUF gguf, Normalizer normalizer, Splitter splitter) {
        String[] tokens = gguf.getValue(String[].class, "tokenizer.ggml.tokens");
        int[] tokenTypes = gguf.getValue(int[].class, "tokenizer.ggml.token_type");
        float[] scores = gguf.getValue(float[].class, "tokenizer.ggml.scores");
        VocabularyImplWithScores vocabulary =
                new VocabularyImplWithScores(tokens, scores, tokenTypes);
        return new MistralTokenizer(vocabulary, normalizer, splitter);
    }
}
