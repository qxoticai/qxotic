package com.qxotic.toknroll.impl;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Toknroll;
import com.qxotic.toknroll.Vocabulary;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class GenericBpeCodePointTest {

    @Test
    void supportsCodePointBasedSymbolsViaPluggableEncoder() {
        Vocabulary vocabulary =
                new VocabularyImpl(
                        Map.of(
                                "a", 0,
                                "b", 1,
                                "ab", 2,
                                "🙂", 3));

        List<Toknroll.MergeRule> merges = List.of(Toknroll.MergeRule.of(0, 1, 0));

        Tokenizer tokenizer =
                Toknroll.pipeline(
                        Splitter.identity(), Toknroll.sentencePieceBpeModel(vocabulary, merges));

        IntSequence tokens = tokenizer.encode("ab🙂ab");
        assertArrayEquals(new int[] {2, 3, 2}, tokens.toArray());
        assertEquals(3, tokenizer.countTokens("ab🙂ab"));
        assertEquals("ab🙂ab", tokenizer.decode(tokens));
    }
}
