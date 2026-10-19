package com.qxotic.toknroll.gguf;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import com.qxotic.toknroll.Tokenizer;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * A literal U+2581 in the text encodes as a space on Gemma 4, whose vocab spells its spaces with
 * it: the ids are those of Hugging Face ({@code google/gemma-4-12b-it} tokenizer.json) and
 * llama.cpp ({@code llama-tokenize}).
 */
@Tag("network")
@Tag("local-external")
class GGUFTokenizerGemma4MetaspaceTest {

    @Test
    void aLiteralMetaspaceEncodesAsASpace() {
        Tokenizer tokenizer =
                GGUFTokenizerLoader.createBuilderWithBuiltins()
                        .build()
                        .fromHuggingFace(
                                "unsloth", "gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q8_0.gguf");
        assertArrayEquals(new int[] {236743}, tokenizer.encode("▁").toArray());
        assertArrayEquals(new int[] {138}, tokenizer.encode("▁▁").toArray());
        assertArrayEquals(new int[] {236746, 518}, tokenizer.encode("a▁b").toArray());
        assertArrayEquals(new int[] {236746, 139, 236763}, tokenizer.encode("a ▁ b").toArray());
    }
}
