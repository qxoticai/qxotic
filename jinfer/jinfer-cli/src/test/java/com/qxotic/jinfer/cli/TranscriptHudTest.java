package com.qxotic.jinfer.cli;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Transcription;
import com.qxotic.jinfer.cli.TranscriptHud.ColorDepth;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

class TranscriptHudTest {

    private static final int WHITE = 0xFFFFFF, BLACK = 0x000000;

    @Test
    void depthFollowsTheEnvironment() {
        assertEquals(ColorDepth.NONE, ColorDepth.of(Map.of("NO_COLOR", "1", "COLORTERM", "24bit")));
        assertEquals(ColorDepth.TRUE, ColorDepth.of(Map.of("NO_COLOR", "", "COLORTERM", "24bit")));
        assertEquals(ColorDepth.TRUE, ColorDepth.of(Map.of("COLORTERM", "truecolor")));
        assertEquals(ColorDepth.EXTENDED, ColorDepth.of(Map.of("TERM", "xterm-256color")));
        assertEquals(ColorDepth.BASIC, ColorDepth.of(Map.of("TERM", "xterm")));
        assertEquals(ColorDepth.BASIC, ColorDepth.of(Map.of()));
    }

    @Test
    void colorsMapToTheNearestTheTerminalShows() {
        assertEquals("\u001b[38;2;255;255;255m", fg(ColorDepth.TRUE, WHITE));
        assertEquals("\u001b[38;5;231m", fg(ColorDepth.EXTENDED, WHITE));
        assertEquals("\u001b[38;5;16m", fg(ColorDepth.EXTENDED, BLACK));
        assertEquals("\u001b[38;5;46m", fg(ColorDepth.EXTENDED, 0x00FF00));
        assertEquals("\u001b[97m", fg(ColorDepth.BASIC, WHITE));
        assertEquals("\u001b[91m", fg(ColorDepth.BASIC, 0xFA1414));
        assertEquals("\u001b[36m", fg(ColorDepth.BASIC, 0x0AC8BE));
        assertEquals("", fg(ColorDepth.NONE, WHITE));
    }

    /**
     * Without colors every cue is an attribute: the tail dim, a landing piece bold, and once it has
     * landed, doubtful words underlined.
     */
    @Test
    void withoutColorsNoColorEscapeIsWritten() throws InterruptedException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        TranscriptHud hud = hud(bytes, ColorDepth.NONE);
        hud.level(0.1f);
        hud.show(
                List.of(token(" sure", 0, 0.99), token(" maybe", 1, 0.2)),
                List.of(token(" next", 2, 0.9)));
        awaitOutput(bytes, "next");
        String out = awaitOutput(bytes, "\u001b[4mmaybe");
        hud.finish(List.of());
        assertFalse(
                Pattern.compile("\u001b\\[[0-9;]*(38;|3[0-7]m|9[0-7]m)").matcher(out).find(),
                "a color escape without colors");
        assertTrue(out.contains("\u001b[2m\u001b[3mnext"), "the tail is dim italic");
        assertTrue(out.contains("\u001b[1msure"), "the landing piece is bold");
    }

    /** Without Unicode, the chrome is drawn in ASCII: nothing outside it is written. */
    @Test
    void withoutUnicodeOnlyAsciiIsWritten() throws InterruptedException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        TranscriptHud hud = hud(bytes, ColorDepth.BASIC, false);
        hud.level(0.1f);
        hud.level(0.3f);
        hud.show(List.of(token(" plain", 0, 0.99)), List.of(token(" words", 1, 0.9)));
        awaitOutput(bytes, "words");
        hud.finish(List.of());
        String out = bytes.toString(StandardCharsets.UTF_8);
        assertTrue(out.chars().allMatch(c -> c < 128), "non-ASCII drawn");
        assertTrue(out.contains("# 0:00 - 0 words"), "the summary line, in ASCII");
    }

    /** A tail from before the latest commit shows only what follows the final text. */
    @Test
    void aStaleTailNeverRepeatsFinalText() throws InterruptedException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        TranscriptHud hud = hud(bytes, ColorDepth.NONE);
        hud.show(
                List.of(token(" one", 0, 0.99), token(" two", 1, 0.99)),
                List.of(token(" two", 1, 0.9), token(" three", 2, 0.9)));
        String out = awaitOutput(bytes, "three");
        hud.finish(List.of());
        String frame = out.substring(out.lastIndexOf("\u001b[?2026h"));
        assertEquals(1, frame.split("two", -1).length - 1, "shown twice: " + frame);
    }

    /** Steady room noise, however loud, is no speech: the view keeps listening. */
    @Test
    void steadyNoiseKeepsListening() throws InterruptedException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        TranscriptHud hud = hud(bytes, ColorDepth.NONE);
        String frame = "";
        for (int i = 0; i < 120; i++) { // past the silence the label waits for
            hud.level(i % 2 == 0 ? 0.30f : 0.32f);
            Thread.sleep(20);
            String out = bytes.toString(StandardCharsets.UTF_8);
            frame = out.substring(out.lastIndexOf("\u001b[?2026h"));
        }
        hud.finish(List.of());
        assertTrue(frame.contains("listening"), "noise read as speech: " + frame);
    }

    @Test
    void resizingAndClosingALiveViewRestoreTheCursor() throws Exception {
        var bytes = new ByteArrayOutputStream();
        var columns = new java.util.concurrent.atomic.AtomicInteger(80);
        var committed = List.of(token(" hello", 0, 0.99), token(" world", 1, 0.9));
        try (var view =
                new TranscriptHud(
                        new Terminal(
                                new PrintStream(bytes, true, StandardCharsets.UTF_8),
                                true,
                                ColorDepth.NONE),
                        columns::get,
                        TranscriptHud.Theme.BUNDLED.getFirst())) {
            view.show(committed, List.of(token(" provisional", 2, 0.9)));
            awaitOutput(bytes, "provisional");
            columns.set(12);
            view.show(committed, List.of(token(" resized", 2, 0.9)));
            awaitOutput(bytes, "resized");
        }
        assertTrue(bytes.toString(StandardCharsets.UTF_8).endsWith("\u001b[0m\u001b[?25h"));
    }

    private static String fg(ColorDepth depth, int rgb) {
        TranscriptHud hud = hud(new ByteArrayOutputStream(), depth);
        String escape = hud.fg(rgb);
        hud.finish(List.of());
        return escape;
    }

    private static TranscriptHud hud(ByteArrayOutputStream bytes, ColorDepth depth) {
        return hud(bytes, depth, true);
    }

    private static TranscriptHud hud(
            ByteArrayOutputStream bytes, ColorDepth depth, boolean unicode) {
        var out = new PrintStream(bytes, true, StandardCharsets.UTF_8);
        return new TranscriptHud(
                new Terminal(out, unicode, depth),
                () -> 80,
                TranscriptHud.Theme.BUNDLED.getFirst());
    }

    /** A token spanning second {@code at}. */
    private static Transcription.Token token(String text, int at, double confidence) {
        return new Transcription.Token(
                text, Duration.ofSeconds(at), Duration.ofSeconds(at + 1), confidence);
    }

    /**
     * The view paints on its own thread; waits until {@code text} has been drawn, as written or,
     * since colors may style each letter, with the escapes taken out.
     */
    private static String awaitOutput(ByteArrayOutputStream bytes, String text)
            throws InterruptedException {
        for (int i = 0; i < 300; i++) {
            String out = bytes.toString(StandardCharsets.UTF_8);
            if (out.contains(text)
                    || out.replaceAll("\u001b\\[[0-9;?]*[A-Za-z]", "").contains(text)) return out;
            Thread.sleep(10);
        }
        throw new AssertionError("never drawn: " + text);
    }
}
