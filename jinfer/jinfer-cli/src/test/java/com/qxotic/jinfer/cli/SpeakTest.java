package com.qxotic.jinfer.cli;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Phonemizer;
import com.qxotic.jinfer.SpeechOptions;
import com.qxotic.jinfer.SpeechSynthesisModel;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jota.memory.MemoryArena;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.Predicate;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.junit.jupiter.api.io.TempDir;

final class SpeakTest {

    @Test
    void blankStdinFailsBeforeLoadingOrWriting(@TempDir Path dir) throws Exception {
        Path model = Files.createFile(dir.resolve("model.gguf"));
        Path output = Files.writeString(dir.resolve("existing.wav"), "keep this");
        assertEquals(
                2,
                run(
                        dir,
                        " \n",
                        "-m",
                        model.toString(),
                        "--speak",
                        "-",
                        "--output",
                        output.toString()));
        assertTrue(Files.readString(dir.resolve("stderr.txt")).contains("non-blank text"));
        assertEquals(0, Files.size(dir.resolve("stdout.wav")));
        assertEquals("keep this", Files.readString(output));
    }

    @Test
    @Tag("integration")
    void inflectWritesAWavFile(@TempDir Path dir) throws Exception {
        Path model = TestModels.require("hf.co/remixerdec/Inflect-Nano-v2-GGUF:Q8_0");
        Path output = dir.resolve("speech.wav");
        assertEquals(
                0,
                run(
                        dir,
                        "",
                        "-m",
                        model.toString(),
                        "speak",
                        "Hello world.",
                        "--output",
                        output.toString(),
                        "--speed",
                        "1.2"),
                diagnostics(dir));
        assertEquals(0, Files.size(dir.resolve("stdout.wav")));
        assertSpeech(output);
    }

    @Test
    @Tag("integration")
    void kokoroReadsStdinAndWritesOnlyWavToStdout(@TempDir Path dir) throws Exception {
        Path model = TestModels.require("simonfxr/kokoro.cpp-GGUF/kokoro-82m-q8_0.gguf");
        Path voice =
                TestModels.require("simonfxr/kokoro.cpp-GGUF/voices/kokoro-voice-af_heart.gguf");
        assertEquals(
                0,
                run(
                        dir,
                        "Hello, world.\n",
                        "-m",
                        model.toString(),
                        "--with",
                        "voice=" + voice,
                        "--speak",
                        "-",
                        "--output",
                        "-"),
                diagnostics(dir));
        assertSpeech(dir.resolve("stdout.wav"));
        assertTrue(diagnostics(dir).contains("s of audio"));
    }

    private static void assertSpeech(Path wav) throws Exception {
        byte[] bytes = Files.readAllBytes(wav);
        assertEquals("RIFF", new String(bytes, 0, 4, StandardCharsets.US_ASCII));
        var audio = AudioCodec.decode(bytes);
        assertTrue(audio.pcm().length > 1600, "at least 100 ms of audio");
        double energy = 0;
        for (float sample : audio.pcm()) energy += sample * sample;
        assertTrue(energy / audio.pcm().length > 1e-8, "speech must not be silent");
    }

    @Test
    @Tag("integration")
    @EnabledOnOs({OS.MAC, OS.LINUX})
    void playbackDeliversAudioAndPropagatesPlayerFailures(@TempDir Path dir) throws Exception {
        Path model = TestModels.require("simonfxr/kokoro.cpp-GGUF/kokoro-82m-q8_0.gguf");
        Path voice =
                TestModels.require("simonfxr/kokoro.cpp-GGUF/voices/kokoro-voice-af_heart.gguf");
        Path players = Files.createDirectory(dir.resolve("players"));
        // Capture playback instead of requiring a sound device or making the test speak aloud.
        String capture =
                """
                #!/bin/sh
                if [ "$#" -eq 2 ] && [ "$1" = "-q" ]; then shift; fi
                if [ "$#" -eq 1 ]; then
                    printf '%s\\n' "$1" >> "$PLAYBACK_CAPTURE/temporary-wavs.txt"
                    /bin/cp "$1" "$PLAYBACK_CAPTURE/clip-$$.wav"
                else
                    /bin/cat > "$PLAYBACK_CAPTURE/stream.pcm"
                fi
                """;
        for (String player : List.of("afplay", "aplay", "ffplay")) {
            Path script = Files.writeString(players.resolve(player), capture);
            assertTrue(script.toFile().setExecutable(true));
        }
        for (String mode : List.of("--play", "--stream")) {
            Path captureDir = Files.createDirectory(dir.resolve(mode.substring(2)));
            assertEquals(
                    0,
                    run(
                            captureDir,
                            "",
                            players,
                            "-m",
                            model.toString(),
                            "--with",
                            "voice=" + voice,
                            "--speak",
                            "Hello world. Another sentence.",
                            mode),
                    diagnostics(captureDir));
            assertEquals(0, Files.size(captureDir.resolve("stdout.wav")));
            if (mode.equals("--play") || System.getProperty("os.name").startsWith("Mac")) {
                try (var files = Files.list(captureDir)) {
                    var clips =
                            files.filter(p -> p.getFileName().toString().startsWith("clip-"))
                                    .toList();
                    assertEquals(mode.equals("--play") ? 1 : 2, clips.size());
                    for (Path clip : clips) assertSpeech(clip);
                }
                assertTemporaryWavsDeleted(captureDir);
            } else {
                assertTrue(Files.size(captureDir.resolve("stream.pcm")) > 3200);
            }
            assertEquals(
                    mode.equals("--stream"), diagnostics(captureDir).contains("first audio after"));
        }
        for (String player : List.of("afplay", "aplay", "ffplay")) {
            Files.writeString(
                    players.resolve(player),
                    """
                    #!/bin/sh
                    if [ "$#" -eq 2 ] && [ "$1" = "-q" ]; then shift; fi
                    if [ "$#" -eq 1 ]; then
                        printf '%s\\n' "$1" >> "$PLAYBACK_CAPTURE/temporary-wavs.txt"
                    fi
                    exit 7
                    """);
        }
        for (String mode : List.of("--play", "--stream")) {
            Path failed = Files.createDirectory(dir.resolve("failed-" + mode.substring(2)));
            String[] args = {
                "-m",
                model.toString(),
                "--with",
                "voice=" + voice,
                "--speak",
                "Hello world. Another sentence.",
                mode
            };
            assertEquals(1, run(failed, "", players, args), diagnostics(failed));
            assertTrue(diagnostics(failed).contains("audio player exited with status 7"));
            if (mode.equals("--play") || System.getProperty("os.name").startsWith("Mac")) {
                assertEquals(
                        1,
                        Files.readAllLines(failed.resolve("temporary-wavs.txt")).size(),
                        "a failed player must stop subsequent clips");
                assertTemporaryWavsDeleted(failed);
            }
        }
    }

    private static void assertTemporaryWavsDeleted(Path dir) throws IOException {
        for (String wav : Files.readAllLines(dir.resolve("temporary-wavs.txt")))
            assertFalse(Files.exists(Path.of(wav)), "temporary WAV leaked: " + wav);
    }

    @Test
    void defaultPlaybackAndStreamingForwardTheModelSettings() throws Exception {
        for (String mode : List.of("--play", "--stream")) {
            var model = new Speech();
            var player = new RecordingPlayback();
            var capture = new CliFixtures.Capture("");
            Options o = Options.parse("speak", "-m", "unused", "Hello.", "--speed", "1.2", mode);
            Speak.execute(model, o.input, o, capture.io, player);
            assertEquals("Hello.", model.text);
            assertEquals(1.2, model.speed);
            assertEquals(1, model.closed);
            assertEquals(mode.equals("--play") ? 1 : 2, player.clips);
            assertEquals(mode.equals("--stream"), player.streamed);
            assertEquals("", capture.out());
        }
        assertTrue(Options.parse("speak", "-m", "unused", "Hi").speech.play);
    }

    @Test
    void fileAndStdoutWavsAreIdenticalAndByteClean(@TempDir Path dir) throws Exception {
        Path output = dir.resolve("speech.wav");
        var fileCapture = new CliFixtures.Capture("");
        var pipeCapture = new CliFixtures.Capture("");
        Options file = Options.parse("speak", "-m", "unused", "Hi", "--output", output.toString());
        Options pipe = Options.parse("speak", "-m", "unused", "Hi", "--output", "-");
        Speak.execute(new Speech(), "Hi", file, fileCapture.io, new RecordingPlayback());
        Speak.execute(new Speech(), "Hi", pipe, pipeCapture.io, new RecordingPlayback());
        byte[] wav = pipeCapture.stdout.toByteArray();
        org.junit.jupiter.api.Assertions.assertArrayEquals(Files.readAllBytes(output), wav);
        assertEquals(wav.length, ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN).getInt(4) + 8);
        assertEquals("", fileCapture.out());
        assertTrue(pipeCapture.err().contains("s of audio"));
    }

    @Test
    void failedStreamingStillClosesTheModelState() {
        var model = new Speech();
        model.fail = true;
        Options o = Options.parse("speak", "-m", "unused", "Hi", "--stream");
        assertThrows(
                IllegalStateException.class,
                () ->
                        Speak.execute(
                                model,
                                "Hi",
                                o,
                                new CliFixtures.Capture("").io,
                                new RecordingPlayback()));
        assertEquals(1, model.closed);
    }

    @Test
    void fileErrorsAndBrokenStdoutAreReportedAfterStateCleanup(@TempDir Path dir) {
        for (Path output : List.of(dir, dir.resolve("missing-parent/file.wav"), Path.of("-"))) {
            var model = new Speech();
            var capture = new CliFixtures.Capture("");
            var broken =
                    new java.io.PrintStream(
                            new java.io.OutputStream() {
                                public void write(int value) throws IOException {
                                    throw new IOException("closed pipe");
                                }
                            });
            Main.IO io =
                    output.toString().equals("-")
                            ? new Main.IO(capture.io.in(), broken, capture.io.err())
                            : capture.io;
            Options o =
                    Options.parse("speak", "-m", "unused", "hello", "--output", output.toString());
            assertThrows(
                    IOException.class,
                    () -> Speak.execute(model, "hello", o, io, new RecordingPlayback()));
            assertEquals(1, model.closed);
        }
    }

    @Test
    void playbackFailureNeverLeavesTheSynthesisStateOpen() {
        var model = new Speech();
        Options o = Options.parse("speak", "-m", "unused", "hello");
        Speak.Playback failing =
                new Speak.Playback() {
                    public void play(Media.Audio audio) throws IOException {
                        throw new IOException("no player available");
                    }

                    public void stream(
                            SpeechSynthesisModel<?, ?, ?> speech,
                            String text,
                            SpeechOptions options,
                            Runnable first) {
                        throw new AssertionError("full playback requested");
                    }
                };
        var failure =
                assertThrows(
                        IOException.class,
                        () ->
                                Speak.execute(
                                        model,
                                        "hello",
                                        o,
                                        new CliFixtures.Capture("").io,
                                        failing));
        assertTrue(failure.getMessage().contains("no player"));
        assertTrue(failure.getMessage().startsWith("cannot play speech\n  no player available"));
        assertEquals("no player available", failure.getCause().getMessage());
        assertEquals(1, model.closed);
    }

    private static final class RecordingPlayback implements Speak.Playback {
        int clips;
        boolean streamed;

        public void play(Media.Audio audio) {
            clips++;
        }

        public void stream(
                SpeechSynthesisModel<?, ?, ?> model,
                String text,
                SpeechOptions options,
                Runnable firstAudio) {
            streamed = true;
            model.speak(
                    text,
                    options,
                    audio -> {
                        if (clips++ == 0) firstAudio.run();
                        return true;
                    });
        }
    }

    static final class Speech implements SpeechSynthesisModel<Void, Void, CliFixtures.State> {
        String text;
        Double speed;
        int closed;
        boolean fail;

        public Void configuration() {
            return null;
        }

        public Void weights() {
            return null;
        }

        public int sampleRate() {
            return 24000;
        }

        public Phonemizer phonemizer() {
            return text -> new int[] {1};
        }

        public CliFixtures.State newState() {
            return new CliFixtures.State(() -> closed++);
        }

        public CliFixtures.State newState(MemoryArena<MemorySegment> arena) {
            return newState();
        }

        public Media.Audio synthesize(
                CliFixtures.State state, int[] phonemes, SpeechOptions options) {
            return new Media.Audio(new float[] {0, 0.25f, -0.25f, 0}, sampleRate(), 1);
        }

        public void speak(
                CliFixtures.State state,
                String text,
                SpeechOptions options,
                Predicate<Media.Audio> sink) {
            this.text = text;
            speed = options.speed();
            assertTrue(state.isAlive());
            if (!sink.test(synthesize(state, new int[] {1}, options))) return;
            if (fail) throw new IllegalStateException("test synthesis failure");
            sink.test(synthesize(state, new int[] {2}, options));
        }
    }

    private static String diagnostics(Path dir) throws IOException {
        return Files.readString(dir.resolve("stderr.txt"));
    }

    private static int run(Path dir, String input, String... args) throws Exception {
        return run(dir, input, null, args);
    }

    private static int run(Path dir, String input, Path players, String... args) throws Exception {
        var command = CliFixtures.javaCommand();
        command.addAll(List.of("-cp", System.getProperty("java.class.path"), Main.class.getName()));
        command.addAll(List.of(args));
        ProcessBuilder builder =
                new ProcessBuilder(command)
                        .redirectOutput(dir.resolve("stdout.wav").toFile())
                        .redirectError(dir.resolve("stderr.txt").toFile());
        if (players != null) {
            builder.environment()
                    .put(
                            "PATH",
                            players
                                    + java.io.File.pathSeparator
                                    + builder.environment().getOrDefault("PATH", ""));
            builder.environment().put("PLAYBACK_CAPTURE", dir.toString());
        }
        Process process = builder.start();
        try {
            try (var stdin = process.getOutputStream()) {
                stdin.write(input.getBytes(StandardCharsets.UTF_8));
            }
            assertTrue(process.waitFor(120, TimeUnit.SECONDS), "CLI timed out");
            assertFalse(diagnostics(dir).contains("Exception in thread"), diagnostics(dir));
            return process.exitValue();
        } finally {
            process.destroyForcibly();
        }
    }
}
