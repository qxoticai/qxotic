package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

class ByteLevelTest {

    @Test
    void roundTripUtf8Bytes() {
        byte[] bytes = "Hello 🌍".getBytes(StandardCharsets.UTF_8);
        String encoded = ByteLevel.encode(bytes);
        byte[] decoded = ByteLevel.decode(encoded);
        assertArrayEquals(bytes, decoded);
    }

    @Test
    void encodeDecodePreservesTextWhenUsingUtf8() {
        String text = "ASCII + emoji 😀 and accents cafe";
        String encoded = ByteLevel.encode(text.getBytes(StandardCharsets.UTF_8));
        String decoded = new String(ByteLevel.decode(encoded), StandardCharsets.UTF_8);
        assertEquals(text, decoded);
    }

    @Test
    void decodeRejectsUnmappedSymbols() {
        assertThrows(IllegalArgumentException.class, () -> ByteLevel.decode("\u0000"));
    }

    @Test
    void singleByteRoundTripWorks() {
        char symbol = ByteLevel.encodeSingle((byte) 0);
        assertEquals('\u0100', symbol);
        assertEquals((byte) 0, ByteLevel.decodeSingle(symbol));
    }
}
