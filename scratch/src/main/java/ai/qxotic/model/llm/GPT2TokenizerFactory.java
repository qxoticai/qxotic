package ai.qxotic.model.llm;

import ai.qxotic.format.gguf.GGUF;
import ai.qxotic.tokenizers.Normalizer;
import ai.qxotic.tokenizers.TextSplitter;
import ai.qxotic.tokenizers.Tokenizer;
import ai.qxotic.tokenizers.Vocabulary;
import ai.qxotic.tokenizers.impl.GPT2Tokenizer;
import ai.qxotic.tokenizers.impl.IntPair;
import ai.qxotic.tokenizers.impl.VocabularyImpl;
import com.google.auto.service.AutoService;
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
    public Tokenizer createTokenizer(GGUF gguf, Normalizer normalizer, TextSplitter textSplitter) {
        String[] tokens = gguf.getValue(String[].class, "tokenizer.ggml.tokens");
        int[] tokenTypes = gguf.getValue(int[].class, "tokenizer.ggml.token_type");
        VocabularyImpl vocabulary = new VocabularyImpl(tokens, null, tokenTypes);
        List<IntPair> merges = loadMerges(gguf, vocabulary);
        return new GPT2Tokenizer(vocabulary, normalizer, textSplitter, merges);
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
