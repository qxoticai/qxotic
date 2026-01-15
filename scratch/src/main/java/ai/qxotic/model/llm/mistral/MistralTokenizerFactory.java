package ai.qxotic.model.llm.mistral;

import com.google.auto.service.AutoService;
import ai.qxotic.format.gguf.GGUF;
import ai.qxotic.model.llm.GGUFTokenizerFactory;
import ai.qxotic.model.llm.TokenizerFactory;
import ai.qxotic.tokenizers.Normalizer;
import ai.qxotic.tokenizers.TextSplitter;
import ai.qxotic.tokenizers.Tokenizer;

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
