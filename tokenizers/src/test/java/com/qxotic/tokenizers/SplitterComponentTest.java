package com.qxotic.tokenizers;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import com.qxotic.tokenizers.advanced.Splitter;
import com.qxotic.tokenizers.impl.RegexSplitter;
import java.util.List;
import org.junit.jupiter.api.Test;

class SplitterComponentTest {

    @Test
    void sequenceWithNoStagesReturnsIdentity() {
        Splitter splitter = Splitter.sequence();
        assertSame(Splitter.identity(), splitter);
    }

    @Test
    void repeatedChunksArePreserved() {
        Splitter splitter = text -> List.of("aa", "aa");

        List<CharSequence> tokens = splitter.split("aaaa");
        assertEquals(2, tokens.size());
        assertEquals("aa", tokens.get(0).toString());
        assertEquals("aa", tokens.get(1).toString());
    }

    @Test
    void regexSplitterSplitsInput() {
        Splitter splitter = RegexSplitter.create("\\s+|[,.!?]");

        List<CharSequence> tokens = splitter.split("Hello, world!");
        assertEquals("Hello", tokens.get(0).toString());
        assertEquals(" ", tokens.get(2).toString());
        assertEquals("!", tokens.get(tokens.size() - 1).toString());
    }
}
