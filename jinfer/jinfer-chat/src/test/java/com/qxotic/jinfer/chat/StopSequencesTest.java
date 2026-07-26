package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

/** The holdback matcher: streamed partials must always concatenate to the trimmed text. */
class StopSequencesTest {

    @Test
    void ofNullOrEmptyIsNull() {
        assertNull(StopSequences.of(null));
        assertNull(StopSequences.of(List.of()));
    }

    @Test
    void noHitFlushesEverything() {
        StopSequences stops = StopSequences.of(List.of("STOP"));
        StringBuilder streamed = new StringBuilder();
        for (String f : new String[] {"hel", "lo ", "wor", "ld"}) streamed.append(stops.feed(f));
        assertFalse(stops.hit());
        streamed.append(stops.flush());
        assertEquals("hello world", streamed.toString());
        assertEquals("hello world", stops.beforeCut());
    }

    @Test
    void hitAtTheVeryStart() {
        StopSequences stops = StopSequences.of(List.of("STOP"));
        String streamed = stops.feed("STOP and more");
        assertTrue(stops.hit());
        assertEquals("", streamed + stops.flush());
        assertEquals("", stops.beforeCut());
    }

    @Test
    void earliestStopWins() {
        StopSequences stops = StopSequences.of(List.of("BBB", "AA"));
        stops.feed("x AA y BBB z");
        assertTrue(stops.hit());
        assertEquals("x ", stops.beforeCut());
    }

    @Test
    void stopLongerThanFragments() {
        StopSequences stops = StopSequences.of(List.of("<|im_end|>"));
        StringBuilder streamed = new StringBuilder();
        for (String f : new String[] {"ab", "<|im", "_", "end|>", "trailing"}) {
            streamed.append(stops.feed(f));
        }
        streamed.append(stops.flush());
        assertTrue(stops.hit());
        assertEquals("ab", streamed.toString());
        // streamed partials concatenate exactly to the trimmed text
        assertEquals(streamed.toString(), stops.beforeCut());
    }

    @Test
    void heldBackCharsAreNotEmittedBeforeTheMatch() {
        StopSequences stops = StopSequences.of(List.of("XXXXX"));
        // everything but the last 4 chars (longest stop - 1) is safe immediately
        assertEquals("hello", stops.feed("helloXXXX"));
        assertFalse(stops.hit());
        assertEquals("", stops.feed("X and more"));
        assertTrue(stops.hit());
        assertEquals("", stops.flush());
        assertEquals("hello", stops.beforeCut());
    }
}
