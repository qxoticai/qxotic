package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class SplitterSequenceTest {

    @Test
    void sequenceWithSingleSplitterReturnsSameInstance() {
        Splitter splitter = Splitter.regex("\\s+");

        Splitter sequence = Splitter.sequence(splitter);

        assertSame(splitter, sequence);
    }

    @Test
    void splitAllRangeUsesAbsoluteOffsetsOnOriginalSource() {
        CharSequence text = "ab cd ef";
        Splitter splitter = Splitter.regex("\\s+");
        List<String> chunks = new ArrayList<>();

        splitter.splitAll(
                text,
                3,
                8,
                (source, start, end) -> {
                    assertSame(text, source);
                    chunks.add(source.subSequence(start, end).toString());
                });

        assertEquals(List.of("cd", " ", "ef"), chunks);
    }

    @Test
    void sequenceAppliesEachStageToPreviousRanges() {
        Splitter comma = Splitter.regex(",");
        Splitter space = Splitter.regex("\\s+");
        Splitter seq = Splitter.sequence(comma, space);
        List<String> chunks = new ArrayList<>();

        seq.splitAll(
                "a, b c",
                (source, start, end) -> chunks.add(source.subSequence(start, end).toString()));

        assertEquals(List.of("a", ",", " ", "b", " ", "c"), chunks);
    }

    @Test
    void sequenceRejectsNullStages() {
        assertThrows(NullPointerException.class, () -> Splitter.sequence((Splitter[]) null));
        assertThrows(
                NullPointerException.class, () -> Splitter.sequence(Splitter.identity(), null));
    }

    @Test
    void identityRejectsInvalidRange() {
        Splitter identity = Splitter.identity();

        assertThrows(
                IndexOutOfBoundsException.class,
                () -> identity.splitAll("abc", -1, 3, (source, s, e) -> {}));
    }
}
