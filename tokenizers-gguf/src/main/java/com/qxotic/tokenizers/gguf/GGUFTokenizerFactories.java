package com.qxotic.tokenizers.gguf;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.tokenizers.Tokenizer;
import com.qxotic.tokenizers.Vocabulary;
import com.qxotic.tokenizers.advanced.Normalizer;
import com.qxotic.tokenizers.advanced.Splitter;
import com.qxotic.tokenizers.impl.GPT2Tokenizer;
import com.qxotic.tokenizers.impl.IntPair;
import com.qxotic.tokenizers.impl.VocabularyImpl;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

final class GGUFTokenizerFactories {

    private GGUFTokenizerFactories() {}

    static void registerDefaults(GGUFTokenizerRegistry registry) {
        registry.register("gpt2", GGUFTokenizerFactories::createGpt2LikeTokenizer);
        registry.register("llama", GGUFTokenizerFactories::createGpt2LikeTokenizer);
    }

    private static Tokenizer createGpt2LikeTokenizer(GGUF gguf, Splitter splitter) {
        String[] tokens = gguf.getValue(String[].class, "tokenizer.ggml.tokens");
        int[] tokenTypes = gguf.getValueOrDefault(int[].class, "tokenizer.ggml.token_type", null);
        Vocabulary vocabulary = new VocabularyImpl(tokens, null, tokenTypes);
        List<IntPair> merges = loadMerges(gguf, vocabulary);
        return new GPT2Tokenizer(vocabulary, Normalizer.IDENTITY, splitter, merges);
    }

    private static List<IntPair> loadMerges(GGUF gguf, Vocabulary vocabulary) {
        String[] mergeLines = gguf.getValueOrDefault(String[].class, "tokenizer.ggml.merges", null);
        if (mergeLines == null || mergeLines.length == 0) {
            return Collections.emptyList();
        }
        return Arrays.stream(mergeLines)
                .map(line -> line.split(" "))
                .filter(parts -> parts.length == 2)
                .map(parts -> new IntPair(vocabulary.id(parts[0]), vocabulary.id(parts[1])))
                .toList();
    }
}
