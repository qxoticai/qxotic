package com.qxotic.toknroll.gguf;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.toknroll.Tokenizer;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * A word that is a token of its own encodes as that token under the pre-tokenizers llama.cpp marks
 * {@code ignore_merges}, though the merges never build it; under the others the merges decide.
 * Llama 3's GGUF encodes " Cumhuriyeti" as one token in llama.cpp, five without the rule.
 */
class GGUFIgnoreMergesTest {

    private static final int AB = 256;
    private static final int ABC = 257;

    // the 256 byte tokens (GPT-2's byte alphabet, id = byte), then "ab" and "abc"; the only merge
    // is a + b, so the merges never build "abc"
    static GGUF gguf(String pre) {
        String[] tokens = new String[258];
        for (int b = 0, shifted = 0; b < 256; b++) {
            boolean printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || b >= 0xAE;
            tokens[b] = String.valueOf((char) (printable ? b : 256 + shifted++));
        }
        tokens[AB] = "ab";
        tokens[ABC] = "abc";
        return Builder.newBuilder()
                .putString("tokenizer.ggml.model", "gpt2")
                .putString("tokenizer.ggml.pre", pre)
                .putArrayOfString("tokenizer.ggml.tokens", tokens)
                .putArrayOfString("tokenizer.ggml.merges", new String[] {"a b"})
                .build();
    }

    private static int[] ids(String pre, String text) {
        Tokenizer tokenizer =
                GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf(pre));
        return tokenizer.encode(text).toArray();
    }

    @ParameterizedTest
    @ValueSource(strings = {"llama-bpe", "llama3", "lfm2", "tekken", "pixtral", "glm4"})
    void aWholeWordTokenWinsWhereLlamaCppIgnoresMerges(String pre) {
        assertArrayEquals(new int[] {ABC}, ids(pre, "abc"));
        assertArrayEquals(new int[] {AB}, ids(pre, "ab"));
    }

    @ParameterizedTest
    @ValueSource(strings = {"smaug-bpe", "qwen2", "gpt-4o", "gpt2"})
    void theMergesDecideElsewhere(String pre) {
        assertArrayEquals(new int[] {AB, 'c'}, ids(pre, "abc"));
    }
}
