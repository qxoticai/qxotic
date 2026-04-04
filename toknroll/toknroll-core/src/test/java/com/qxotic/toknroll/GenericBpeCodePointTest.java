package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.advanced.Normalizer;
import com.qxotic.toknroll.advanced.Splitter;
import com.qxotic.toknroll.impl.CodePointSymbolEncoder;
import com.qxotic.toknroll.impl.LongLongBpeMergeTable;
import com.qxotic.toknroll.impl.LongLongMap;
import com.qxotic.toknroll.impl.VocabularyImpl;
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

        long[] keys = {(((long) 0) << 32) | 1L};
        long[] values = {(((long) 2) << 32) | 0L};
        LongLongMap map = new LongLongMap(keys, values);

        Tokenizer tokenizer =
                Tokenizers.genericBpe(
                        vocabulary,
                        Normalizer.identity(),
                        Splitter.identity(),
                        new LongLongBpeMergeTable(map),
                        new CodePointSymbolEncoder());

        IntSequence tokens = tokenizer.encode("ab🙂ab");
        assertArrayEquals(new int[] {2, 3, 2}, tokens.toArray());
        assertEquals(3, tokenizer.countTokens("ab🙂ab"));
        assertEquals("ab🙂ab", tokenizer.decode(tokens));
    }
}
