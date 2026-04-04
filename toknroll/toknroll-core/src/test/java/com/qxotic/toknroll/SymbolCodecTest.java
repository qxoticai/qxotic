package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.advanced.SymbolCodec;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

class SymbolCodecTest {

    @Test
    void identityRoundTripUtf8Text() {
        String text = "Hello 🌍";
        String encoded = SymbolCodec.IDENTITY.encodeText(text);
        String decoded = SymbolCodec.IDENTITY.decodeToText(encoded);
        assertEquals(text, decoded);
    }

    @Test
    void byteLevelRoundTripUtf8Bytes() {
        byte[] bytes = "Hello 🌍".getBytes(StandardCharsets.UTF_8);
        String encoded = SymbolCodec.BYTE_LEVEL.encodeBytes(bytes);
        byte[] decoded = SymbolCodec.BYTE_LEVEL.decodeSymbols(encoded);
        assertArrayEquals(bytes, decoded);
    }

    @Test
    void byteLevelEncodeTextDecodeTextRoundTrip() {
        String text = "ASCII + emoji 😀 and accents café";
        String encoded = SymbolCodec.BYTE_LEVEL.encodeText(text);
        String decoded = SymbolCodec.BYTE_LEVEL.decodeToText(encoded);
        assertEquals(text, decoded);
    }
}
