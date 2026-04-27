package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.util.List;
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
    void isValidEncodingRecognizesValidAndInvalidSymbols() {
        String valid = ByteLevel.encode("Hello �".getBytes(StandardCharsets.UTF_8));
        assertTrue(ByteLevel.isValidEncoding(valid));

        assertFalse(ByteLevel.isValidEncoding("�"));
        assertFalse(ByteLevel.isValidEncoding("Hello � world"));
    }

    @Test
    void tiktokenModelRejectsNonByteLevelVocabularyTokens() {
        String[] tokens = new String[257];
        for (int i = 0; i < 256; i++) {
            tokens[i] = String.valueOf(ByteLevel.encodeSingle((byte) i));
        }
        tokens[256] = "�";

        Vocabulary vocabulary = Tokenizers.vocabulary(tokens);
        IllegalArgumentException error =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Tokenizers.tiktokenModel(vocabulary, List.of()));
        assertTrue(error.getMessage().contains("non-bytelevel token"));
    }
}
