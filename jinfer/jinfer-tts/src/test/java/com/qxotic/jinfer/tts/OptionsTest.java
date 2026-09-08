package com.qxotic.jinfer.tts;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Test;

class OptionsTest {
    @Test
    void parsesGenericModelAndCompanions() {
        Options options =
                Options.parse(
                        new String[] {
                            "z://models/kokoro.gguf",
                            "--with",
                            "voice=z://voices/heart.gguf",
                            "--text",
                            "Hello.",
                            "--speed",
                            "1.1",
                            "--output",
                            "hello.wav"
                        });

        assertEquals("z://models/kokoro.gguf", options.model());
        assertEquals(Map.of("voice", "z://voices/heart.gguf"), options.companions());
        assertEquals("Hello.", options.text());
        assertEquals(1.1, options.speed());
        assertEquals(Path.of("hello.wav"), options.output());
    }

    @Test
    void noArgumentsShowsHelp() {
        assertTrue(Options.parse(new String[0]).help());
    }

    @Test
    void acceptsNamedModel() {
        assertEquals("model.gguf", Options.parse(new String[] {"--model", "model.gguf"}).model());
    }

    @Test
    void rejectsMalformedAndDuplicateCompanions() {
        assertThrows(
                IllegalArgumentException.class,
                () -> Options.parse(new String[] {"model.gguf", "--with", "voice"}));
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Options.parse(
                                new String[] {
                                    "model.gguf", "--with", "voice=a", "--with", "voice=b"
                                }));
    }

    @Test
    void rejectsInvalidSpeed() {
        assertThrows(
                IllegalArgumentException.class,
                () -> Options.parse(new String[] {"model.gguf", "--speed", "0"}));
        assertThrows(
                IllegalArgumentException.class,
                () -> Options.parse(new String[] {"model.gguf", "--speed", "NaN"}));
    }

    @Test
    void parsesStreamAndRejectsBothPlaybackModes() {
        assertTrue(Options.parse(new String[] {"model.gguf", "--stream"}).stream());
        assertThrows(
                IllegalArgumentException.class,
                () -> Options.parse(new String[] {"model.gguf", "--play", "--stream"}));
    }
}
