package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.impl.SentencePieceBpeModel;
import com.qxotic.toknroll.impl.VocabularyImpl;
import org.junit.jupiter.api.Test;

class SentencePieceBpeModelTest {

    @Test
    void encodeGreedilyMergesHighestScorePairs() {
        String[] tokens = {"<0x00>", " ", "h", "e", "l", "o", "he", "ll", "llo", "hello", " hello"};
        float[] scores = {0f, 0f, 0f, 0f, 0f, 0f, 1f, 2f, 3f, 4f, 5f};
        int[] tokenTypes = new int[tokens.length];
        for (int i = 0; i < tokenTypes.length; i++) {
            tokenTypes[i] = StandardTokenType.NORMAL.getId();
        }
        tokenTypes[0] = StandardTokenType.BYTE.getId();

        Vocabulary vocabulary = new VocabularyImpl(tokens, tokenTypes);
        SentencePieceBpeModel model = SentencePieceBpeModel.fromVocabulary(vocabulary, scores);

        IntSequence encoded = model.encode(" hello");

        assertArrayEquals(new int[] {10}, encoded.toArray());
        assertEquals(" hello", model.decode(encoded));
    }

    @Test
    void encodeMergesMultipleSpacesIntoMetaspaceRun() {
        String[] tokens = {"<0x00>", " ", "  ", "a"};
        float[] scores = {0f, 0f, 1f, 0f};
        int[] tokenTypes = {
            StandardTokenType.BYTE.getId(),
            StandardTokenType.NORMAL.getId(),
            StandardTokenType.NORMAL.getId(),
            StandardTokenType.NORMAL.getId()
        };

        Vocabulary vocabulary = new VocabularyImpl(tokens, tokenTypes);
        SentencePieceBpeModel model = SentencePieceBpeModel.fromVocabulary(vocabulary, scores);

        IntSequence encoded = model.encode("  ");

        assertArrayEquals(new int[] {2}, encoded.toArray());
        assertEquals("  ", model.decode(encoded));
    }

    @Test
    void byteFallbackUsesExplicitByteTokenIds() {
        String[] tokens = new String[900];
        float[] scores = new float[tokens.length];
        int[] tokenTypes = new int[tokens.length];
        for (int i = 0; i < tokenTypes.length; i++) {
            tokenTypes[i] = StandardTokenType.NORMAL.getId();
        }

        for (int b = 0; b < 256; b++) {
            int id = 100 + (b * 3);
            tokens[id] = String.format("<0x%02X>", b);
            tokenTypes[id] = StandardTokenType.BYTE.getId();
        }

        Vocabulary vocabulary = new VocabularyImpl(tokens, tokenTypes);
        SentencePieceBpeModel model = SentencePieceBpeModel.fromVocabulary(vocabulary, scores);

        IntSequence encoded = model.encode("é");

        assertArrayEquals(new int[] {685, 607}, encoded.toArray());
        assertEquals("é", model.decode(encoded));
    }

    @Test
    void byteFallbackMissingByteTokenThrowsClearError() {
        String[] tokens = {"<0x00>", " ", "x"};
        float[] scores = {0f, 0f, 0f};
        int[] tokenTypes = {
            StandardTokenType.BYTE.getId(),
            StandardTokenType.NORMAL.getId(),
            StandardTokenType.NORMAL.getId()
        };

        Vocabulary vocabulary = new VocabularyImpl(tokens, tokenTypes);
        SentencePieceBpeModel model = SentencePieceBpeModel.fromVocabulary(vocabulary, scores);

        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> model.encode("é"));
        assertTrue(ex.getMessage().contains("<0xC3>"));
    }
}
