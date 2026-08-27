package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;
import java.util.Arrays;
import org.junit.jupiter.api.Assertions;

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
    void emptyStopStringsAreRefusedNotSilentlyFatal() {
        // "" matches at index 0 of anything: apply() cuts every reply to nothing, so the list is
        // refused at every boundary instead
        assertTrue(TextStops.apply("hello", List.of("")).stopped(), "the hazard this guards");
        assertEquals("", TextStops.apply("hello", List.of("")).text());
        for (List<String> bad :
                List.<List<String>>of(
                        List.of(""), List.of("stop", ""), Arrays.asList((String) null))) {
            var e =
                    Assertions.assertThrows(
                            IllegalArgumentException.class, () -> TextStops.checked(bad));
            assertEquals("stop strings must not be empty", e.getMessage());
            Assertions.assertThrows(
                    IllegalArgumentException.class, () -> new TextStops.Holdback(bad, t -> {}));
        }
        assertEquals(List.of(), TextStops.checked(null));
        assertEquals(List.of("a", "bb"), TextStops.checked(List.of("a", "bb")));
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
