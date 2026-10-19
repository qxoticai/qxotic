package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class ValuesTest {

    @Test
    void numbersAreJsonNumbersNotStrings() {
        assertEquals(42L, Values.longValue(42, "seed", 0));
        assertEquals(42L, Values.longValue(42.0, "seed", 0));
        assertEquals(7L, Values.longValue(null, "seed", 7));
        assertEquals(0.5f, Values.floatValue(0.5, "temperature", 0));
        assertEquals(0.5f, Values.floatValue(null, "temperature", 0.5f));
        // a string-encoded number is a client mistake and says so, as the OpenAI API does
        Values.InvalidParam seed =
                assertThrows(Values.InvalidParam.class, () -> Values.longValue("42", "seed", 0));
        assertTrue(
                seed.getMessage().contains("seed must be an integer; got '42'"), seed.getMessage());
        Values.InvalidParam temperature =
                assertThrows(
                        Values.InvalidParam.class,
                        () -> Values.floatValue("0.5", "temperature", 0));
        assertTrue(
                temperature.getMessage().contains("temperature must be a number; got '0.5'"),
                temperature.getMessage());
        assertThrows(Values.InvalidParam.class, () -> Values.longValue(2.5, "max_tokens", 0));
        assertThrows(Values.InvalidParam.class, () -> Values.longValue(true, "max_tokens", 0));
    }
}
