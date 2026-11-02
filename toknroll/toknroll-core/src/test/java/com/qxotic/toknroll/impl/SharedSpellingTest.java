package com.qxotic.toknroll.impl;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.ByteLevel;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Specials;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Toknroll;
import com.qxotic.toknroll.Vocabulary;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * Two ids with one spelling: a special token whose bytes are also an ordinary token. Laguna has
 * one, {@code |◊|} at id 13 and its bytes at id 18651; once the special is spelled byte-level like
 * every other token, both ids share the spelling. Neither id may get lost.
 */
class SharedSpellingTest {

    @Test
    void specialBelowTheOrdinaryToken() {
        check(256, 257); // Laguna's order: the special first
    }

    @Test
    void specialAboveTheOrdinaryToken() {
        check(257, 256);
    }

    private static void check(int special, int ordinary) {
        String[] tokens = new String[258];
        int[] types = new int[258];
        for (int i = 0; i < 256; i++) tokens[i] = String.valueOf(ByteLevel.encodeSingle((byte) i));
        Arrays.fill(types, StandardTokenType.NORMAL.getId());
        tokens[special] = tokens[ordinary] = "ab";
        types[special] = StandardTokenType.CONTROL.getId();
        Vocabulary vocabulary = ImplAccessor.createVocabulary(tokens, types);
        Tokenizer tokenizer =
                Toknroll.pipeline(
                        Splitter.identity(), Toknroll.tiktokenModel(vocabulary, List.of()));

        assertEquals(ordinary, vocabulary.id("ab"), "the spelling names the ordinary token");
        assertEquals("ab", tokenizer.decode(IntSequence.of(special)), "the special decodes");
        assertEquals(
                "ab", tokenizer.decode(IntSequence.of(ordinary)), "the ordinary token decodes");
        assertEquals(
                Map.of("ab", special), Specials.of(tokenizer).tokens(), "the special is found");
    }
}
