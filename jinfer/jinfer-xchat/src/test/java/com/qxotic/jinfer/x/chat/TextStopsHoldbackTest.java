package com.qxotic.jinfer.x.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

/** The holdback matcher: streamed partials must always concatenate to the trimmed text. */
class TextStopsHoldbackTest {

    private static String streamed(List<String> stops, String... fragments) {
        StringBuilder out = new StringBuilder();
        TextStops.Holdback watch = new TextStops.Holdback(stops, out::append);
        for (String f : fragments) watch.accept(f);
        watch.flush();
        return out.toString();
    }

    @Test
    void noHitFlushesEverything() {
        StringBuilder out = new StringBuilder();
        TextStops.Holdback watch = new TextStops.Holdback(List.of("STOP"), out::append);
        for (String f : new String[] {"hel", "lo ", "wor", "ld"}) watch.accept(f);
        assertFalse(watch.stopped());
        watch.flush();
        assertEquals("hello world", out.toString());
        assertEquals("hello world", TextStops.apply("hello world", List.of("STOP")).text());
    }

    @Test
    void hitAtTheVeryStart() {
        StringBuilder out = new StringBuilder();
        TextStops.Holdback watch = new TextStops.Holdback(List.of("STOP"), out::append);
        watch.accept("STOP and more");
        assertTrue(watch.stopped());
        watch.flush();
        assertEquals("", out.toString());
    }

    @Test
    void earliestStopWins() {
        assertEquals("x ", TextStops.apply("x AA y BBB z", List.of("BBB", "AA")).text());
        assertTrue(TextStops.apply("x AA y BBB z", List.of("BBB", "AA")).stopped());
    }

    @Test
    void stopLongerThanFragments() {
        List<String> stops = List.of("<|im_end|>");
        String text = streamed(stops, "ab", "<|im", "_", "end|>", "trailing");
        assertEquals("ab", text);
        // streamed partials concatenate exactly to the trimmed text
        assertEquals(text, TextStops.apply("ab<|im_end|>trailing", stops).text());
    }

    @Test
    void heldBackCharsAreNotEmittedBeforeTheMatch() {
        StringBuilder out = new StringBuilder();
        TextStops.Holdback watch = new TextStops.Holdback(List.of("XXXXX"), out::append);
        watch.accept("helloXXXX");
        // everything but a could-still-be-a-stop suffix is safe immediately
        assertEquals("hello", out.toString());
        assertFalse(watch.stopped());
        watch.accept("X and more");
        assertTrue(watch.stopped());
        watch.flush();
        assertEquals("hello", out.toString());
    }

    @Test
    void emptyStopsPassEverythingThrough() {
        assertEquals("abc", streamed(List.of(), "a", "b", "c"));
        assertFalse(TextStops.apply("abc", List.of()).stopped());
    }
}
