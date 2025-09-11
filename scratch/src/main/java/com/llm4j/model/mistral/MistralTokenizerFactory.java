package com.llm4j.model.mistral;

import com.google.auto.service.AutoService;
import com.llm4j.gguf.GGUF;
import com.llm4j.model.GGUFTokenizerFactory;
import com.llm4j.model.TokenizerFactory;
import com.llm4j.tokenizers.Normalizer;
import com.llm4j.tokenizers.TextSplitter;
import com.llm4j.tokenizers.Tokenizer;

// Implementation for Mistral tokenizer
@AutoService(TokenizerFactory.class)
public class MistralTokenizerFactory implements GGUFTokenizerFactory {
    @Override
    public String getTokenizerName() {
        return "llama";
    }

    @Override
    public Tokenizer createTokenizer(GGUF gguf, Normalizer normalizer, TextSplitter textSplitter) {
        String[] tokens = gguf.getValue(String[].class, "tokenizer.ggml.tokens");
        int[] tokenTypes = gguf.getValue(int[].class, "tokenizer.ggml.token_type");
        float[] scores = gguf.getValue(float[].class, "tokenizer.ggml.scores");
        VocabularyImplWithScores vocabulary = new VocabularyImplWithScores(tokens, scores, tokenTypes);
        return new MistralTokenizer(vocabulary, normalizer, textSplitter);
    }
}
