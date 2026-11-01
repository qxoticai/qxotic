package com.qxotic.toknroll.gguf;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.TokenizationPipeline;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * The pre names whose regex is one of the tiktoken family's get core's hand-written scanner, which
 * the vector engine takes over: a pre registered under a regex spelled otherwise falls back to
 * {@code java.util.regex}, several times slower, with no other sign.
 */
class GGUFFastSplitterTest {

    private static String splitter(String pre) {
        TokenizationPipeline pipeline =
                (TokenizationPipeline)
                        GGUFTokenizerLoader.createBuilderWithBuiltins()
                                .build()
                                .fromGGUF(GGUFIgnoreMergesTest.gguf(pre));
        return pipeline.splitter().orElseThrow().getClass().getSimpleName();
    }

    @ParameterizedTest
    @ValueSource(
            strings = {
                "gpt-2",
                "llama3",
                "llama-bpe",
                "smollm3",
                "lfm2",
                "qwen2",
                "gpt-4o",
                "llama4",
                "tekken",
                "pixtral",
                "qwen35"
            })
    void theFamilyGetsTheScanner(String pre) {
        assertEquals("TiktokenSplitter", splitter(pre), pre);
    }
}
