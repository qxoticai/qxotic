package com.qxotic.toknroll.gguf;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

/** llama.cpp's SentencePiece space prefix: before every text, the empty one excepted. */
class MetaspaceNormalizerTest {

    @Test
    void everyNonEmptyTextGetsALeadingMetaspace() {
        assertEquals("▁a▁b", GGUFTokenizerDefaults.METASPACE_NORMALIZER.apply("a b").toString());
        assertEquals(
                "▁▁",
                GGUFTokenizerDefaults.METASPACE_NORMALIZER
                        .apply(" ")
                        .toString()); // llama.cpp: [259]
    }

    @Test
    void anEmptyTextStaysEmpty() {
        assertEquals(
                "",
                GGUFTokenizerDefaults.METASPACE_NORMALIZER
                        .apply("")
                        .toString()); // llama.cpp: no ids
    }
}
