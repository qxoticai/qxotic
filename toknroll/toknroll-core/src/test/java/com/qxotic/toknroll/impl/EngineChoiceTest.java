package com.qxotic.toknroll.impl;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.testkit.TiktokenFixtures;
import org.junit.jupiter.api.Test;

/**
 * Running out of memory while choosing an engine fails the call; it does not rule the engine out.
 */
class EngineChoiceTest {
    @Test
    void anOutOfMemoryIsNotADisagreement() {
        Tokenizer tokenizer = TiktokenFixtures.createTiktokenTokenizer("cl100k_base");
        ArmedEngine.armed = true;
        try {
            assertThrows(OutOfMemoryError.class, () -> tokenizer.encode("hello world"));
        } finally {
            ArmedEngine.armed = false;
        }
        int before = ArmedEngine.asked.get();
        assertEquals(
                TiktokenFixtures.createJtokkitTokenizer("cl100k_base").encode("hello world"),
                tokenizer.encode("hello world"));
        assertTrue(ArmedEngine.asked.get() > before, "the engine was asked again");
    }
}
