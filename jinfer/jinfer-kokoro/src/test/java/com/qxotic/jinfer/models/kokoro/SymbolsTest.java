package com.qxotic.jinfer.models.kokoro;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

final class SymbolsTest {

    @Test
    void mapsUnicodeCodePoints() {
        String[] vocabulary = {"", "a", "ɐ", "!", ""};
        Symbols symbols = new Symbols(vocabulary);
        assertArrayEquals(new int[] {1, 2, 3}, symbols.toRaw("a?ɐ!"));
    }

    @Test
    void refusesMultiCodePointTokensRatherThanMisencodingThem() {
        assertThrows(IllegalArgumentException.class, () -> new Symbols(new String[] {"", "ab"}));
    }
}
