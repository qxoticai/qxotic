package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import com.qxotic.toknroll.advanced.Splitter;
import com.qxotic.toknroll.impl.RegexSplitter;
import com.qxotic.toknroll.testkit.SplitterContractHarness;
import com.qxotic.toknroll.testkit.TestCorpora;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class SplitterComponentTest {

    @Test
    void sequenceWithNoStagesReturnsIdentity() {
        Splitter splitter = Splitter.sequence();
        List<CharSequence> chunks = splitter.splitAllToListEagerly("abc");
        assertEquals(1, chunks.size());
        assertEquals("abc", chunks.get(0).toString());
    }

    @Test
    void repeatedChunksArePreserved() {
        Splitter splitter =
                (text, start, end, consumer) -> {
                    consumer.accept(text, start, start + 2);
                    consumer.accept(text, start + 2, start + 4);
                };

        List<CharSequence> tokens = splitter.splitAllToListEagerly("aaaa");
        assertEquals(2, tokens.size());
        assertEquals("aa", tokens.get(0).toString());
        assertEquals("aa", tokens.get(1).toString());
    }

    @Test
    void regexSplitterSplitsInput() {
        Splitter splitter = RegexSplitter.create("\\s+|[,.!?]");

        List<CharSequence> tokens = splitter.splitAllToListEagerly("Hello, world!");
        assertEquals("Hello", tokens.get(0).toString());
        assertEquals(" ", tokens.get(2).toString());
        assertEquals("!", tokens.get(tokens.size() - 1).toString());
    }

    @Test
    void splitRangesPreservesChunkBoundaries() {
        String input = "Hello, world!";
        Splitter splitter = RegexSplitter.create("\\s+|[,.!?]");

        List<String> chunksBySplit =
                splitter.splitAllToListEagerly(input).stream()
                        .map(Object::toString)
                        .collect(Collectors.toList());
        List<String> chunksByRange = new ArrayList<>();
        splitter.splitAll(
                input,
                (source, start, end) ->
                        chunksByRange.add(source.subSequence(start, end).toString()));

        assertEquals(chunksBySplit, chunksByRange);
    }

    @Test
    void splitAllUsesOriginalSourceReference() {
        CharSequence input = new StringBuilder("ab cd");
        Splitter splitter = Splitter.sequence(Splitter.regex("\\s+"));

        splitter.splitAll(input, (source, start, end) -> assertSame(input, source));
    }

    @Test
    void regexSplittersRespectCornerCaseContracts() {
        List<Splitter> splitters =
                List.of(
                        Splitter.regex("\\s+|[,.!?]"),
                        Splitter.regex("\\p{N}{1,3}| ?\\p{L}+| ?[^\\s\\p{L}\\p{N}]+|\\s+"),
                        Splitter.regex(
                                "(?:'[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+|"
                                        + " ?[^\\s\\p{L}\\p{N}]+|\\s+"));

        for (int i = 0; i < splitters.size(); i++) {
            Splitter splitter = splitters.get(i);
            for (String sample : TestCorpora.REGEX_SPLITTER_CORNER_SAMPLES) {
                SplitterContractHarness.assertConformsOnText("regex-corner-" + i, splitter, sample);
            }
        }
    }
}
