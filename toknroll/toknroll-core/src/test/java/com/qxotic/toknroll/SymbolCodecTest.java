package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

class SymbolCodecTest {

    @Test
    void identityRoundTripUtf8Text() {
        String text = "Hello 🌍";
        String encoded = text;
        String decoded = encoded;
        assertEquals(text, decoded);
    }

    @Test
    void byteLevelRoundTripUtf8Bytes() {
        byte[] bytes = "Hello 🌍".getBytes(StandardCharsets.UTF_8);
        String encoded = ByteLevel.encode(bytes);
        byte[] decoded = ByteLevel.decode(encoded);
        assertArrayEquals(bytes, decoded);
    }

    @Test
    void byteLevelEncodeTextDecodeTextRoundTrip() {
        String text = "ASCII + emoji 😀 and accents café";
        String encoded = ByteLevel.encode(text.getBytes(StandardCharsets.UTF_8));
        String decoded = new String(ByteLevel.decode(encoded), StandardCharsets.UTF_8);
        assertEquals(text, decoded);
    }
}
