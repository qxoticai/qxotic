package com.qxotic.toknroll.impl;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

/**
 * Hugging Face's Metaspace, prepend scheme "first", as Mistral-7B v0.3's tokenizer.json uses it.
 */
class MetaspaceTest {

    @Test
    void theFirstSegmentGetsALeadingMetaspace() {
        assertEquals("▁a▁b", ImplAccessor.metaspaced("a b", 0, 3));
    }

    @Test
    void anEmptyTextStaysEmpty() {
        assertEquals("", ImplAccessor.metaspaced("", 0, 0)); // HF encodes "" to no ids
    }

    @Test
    void aTextStartingWithASpaceGetsNoSecondMetaspace() {
        assertEquals("▁", ImplAccessor.metaspaced(" ", 0, 1)); // HF: " " is one ▁, id 29473
        assertEquals("▁a", ImplAccessor.metaspaced(" a", 0, 2));
    }

    @Test
    void aLaterSegmentGetsNone() {
        assertEquals("b", ImplAccessor.metaspaced("<s>b", 3, 4)); // after a special token
    }
}
