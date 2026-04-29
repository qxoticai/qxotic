package com.qxotic.toknroll.impl;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.ByteLevel;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Toknroll;
import com.qxotic.toknroll.Vocabulary;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class TiktokenModelByteLevelValidationTest {

    @Test
    void acceptsValidByteLevelVocabulary() {
        Vocabulary vocabulary = Toknroll.vocabulary(validByteLevelTokens());
        Tokenizer tokenizer =
                Toknroll.pipeline(
                        Splitter.identity(), Toknroll.tiktokenModel(vocabulary, List.of()));

        String text = "Hello \u001F\u007F and \t tabs";
        IntSequence ids = tokenizer.encode(text);
        assertEquals(text, tokenizer.decode(ids));
    }

    @Test
    void rejectsLiteralReplacementCharacterToken() {
        String[] tokens = withExtraToken(validByteLevelTokens(), "\uFFFD");
        Vocabulary vocabulary = Toknroll.vocabulary(tokens);
        IllegalArgumentException error =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Toknroll.tiktokenModel(vocabulary, List.of()));

        assertTrue(error.getMessage().contains("not valid ByteLevel encoding"));
        assertTrue(error.getMessage().contains("id 256"));
        assertTrue(error.getMessage().contains("All vocabulary tokens must be byte-level"));
    }

    @Test
    void acceptsByteLevelFormOfReplacementBytes() {
        String byteLevelReplacement = ByteLevel.encode("\uFFFD".getBytes(StandardCharsets.UTF_8));
        assertTrue(ByteLevel.isValidEncoding(byteLevelReplacement));

        String[] tokens = withExtraToken(validByteLevelTokens(), byteLevelReplacement);
        Vocabulary vocabulary = Toknroll.vocabulary(tokens);
        Toknroll.tiktokenModel(vocabulary, List.of());
    }

    private static String[] validByteLevelTokens() {
        String[] tokens = new String[256];
        for (int i = 0; i < 256; i++) {
            tokens[i] = String.valueOf(ByteLevel.encodeSingle((byte) i));
        }
        return tokens;
    }

    private static String[] withExtraToken(String[] base, String token) {
        List<String> tokens = new ArrayList<>(base.length + 1);
        for (String t : base) {
            tokens.add(t);
        }
        tokens.add(token);
        return tokens.toArray(new String[0]);
    }
}
