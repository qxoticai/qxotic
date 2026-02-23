package com.qxotic.model.llm;

import com.google.auto.service.AutoService;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.tokenizers.Tokenizer;
import com.qxotic.tokenizers.Vocabulary;
import com.qxotic.tokenizers.advanced.Normalizer;
import com.qxotic.tokenizers.advanced.Splitter;
import com.qxotic.tokenizers.impl.GPT2Tokenizer;
import com.qxotic.tokenizers.impl.IntPair;
import com.qxotic.tokenizers.impl.VocabularyImpl;
import java.util.Arrays;
import java.util.List;

// Implementation for GPT2 tokenizer
@AutoService(TokenizerFactory.class)
public class GPT2TokenizerFactory implements GGUFTokenizerFactory {

    @Override
    public String getTokenizerName() {
        return "gpt2";
    }

    @Override
    public Tokenizer createTokenizer(GGUF gguf, Normalizer normalizer, Splitter splitter) {
        String[] tokens = gguf.getValue(String[].class, "tokenizer.ggml.tokens");
        int[] tokenTypes = gguf.getValue(int[].class, "tokenizer.ggml.token_type");
        VocabularyImpl vocabulary = new VocabularyImpl(tokens, null, tokenTypes);
        List<IntPair> merges = loadMerges(gguf, vocabulary);
        return new GPT2Tokenizer(vocabulary, normalizer, splitter, merges);
    }

    public static List<IntPair> loadMerges(GGUF gguf, Vocabulary vocabulary) {
        String[] mergeLines = gguf.getValue(String[].class, "tokenizer.ggml.merges");
        List<IntPair> merges =
                Arrays.stream(mergeLines)
                        .map(line -> line.split(" "))
                        .map(parts -> new IntPair(vocabulary.id(parts[0]), vocabulary.id(parts[1])))
                        .toList();
        return merges;
    }
}
