package com.qxotic.jinfer.cli;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.hub.ModelStore;
import com.qxotic.jinfer.media.Media;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.api.parallel.Isolated;

/**
 * Main.run through the real resolver/loader and back, with no learned weights or public network.
 */
@Isolated("Observes the test provider's loaded models and weights arena")
class WorkflowTest {
    @TempDir Path dir;

    @BeforeEach
    @AfterEach
    void resetProvider() {
        CliModelProvider.reset();
    }

    private Path model(String kind, String failure) throws Exception {
        Path path = dir.resolve(kind + " model こんにちは.gguf");
        GGUF.write(
                Builder.newBuilder()
                        .putString("general.architecture", "cli_test_" + kind)
                        .putString("test.failure", failure)
                        .build(),
                path);
        return path;
    }

    private int run(CliFixtures.Capture capture, String... args) {
        return Main.run(args, capture.io, ModelStore.of(dir.resolve("cache")));
    }

    @Test
    void instructAndPromptUseStdinAndModelDefaultsAndReleaseWeights() throws Exception {
        Path path = model("language", "");
        for (String verb : List.of("instruct", "prompt")) {
            var capture = new CliFixtures.Capture("A multilingual prompt: café 日本語.");
            assertEquals(
                    0, run(capture, "-m", path.toString(), verb, "-", "-n", "4"), capture.err());
            assertEquals("xxxx\n", capture.out().replace("\r\n", "\n"));
            assertEquals(
                    "A multilingual prompt: café 日本語.",
                    CliModelProvider.template.conversations.getFirst().messages().getLast().text());
            assertFalse(CliModelProvider.weights.scope().isAlive());
            assertFalse(capture.inputClosed);
        }
        assertEquals(2, CliModelProvider.languageLoads);
    }

    @Test
    void chatAcceptsPipedTurnsAndUsesTheRequestedContext() throws Exception {
        Path path = model("language", "");
        var capture = new CliFixtures.Capture("/context\nfirst\nsecond\n/exit\n");
        assertEquals(
                0,
                run(
                        capture,
                        "chat",
                        "--model=" + path,
                        "-c",
                        "128",
                        "-n",
                        "2",
                        "--system-prompt",
                        "Be brief."),
                capture.err());
        assertEquals("xx\nxx\n", capture.out().replace("\r\n", "\n"));
        assertTrue(capture.err().contains("capacity 128"), capture.err());
        assertEquals(
                "Be brief.",
                CliModelProvider.template.conversations.getLast().messages().getFirst().text());
        assertFalse(CliModelProvider.weights.scope().isAlive());
    }

    @Test
    void modelMaximumAndOverCapacityHaveDifferentOutcomes() throws Exception {
        Path path = model("language", "");
        var capture = new CliFixtures.Capture("/context\n/exit\n");
        assertEquals(0, run(capture, "chat", "-m", path.toString(), "-c", "0"));
        assertTrue(capture.err().contains("capacity 16384"));
        var invalid = new CliFixtures.Capture("");
        assertEquals(1, run(invalid, "instruct", "-m", path.toString(), "hi", "-c", "20000"));
        assertTrue(invalid.err().contains("exceeds"));
        assertFalse(
                CliModelProvider.weights.scope().isAlive(),
                "failed engine construction releases weights");
    }

    @Test
    void voiceAndSpeedSurviveResolutionAndStdoutContainsOnlyWav() throws Exception {
        Path path = model("speech", "");
        Path voice = Files.writeString(dir.resolve("O'Brien voice=heart.gguf"), "voice fixture");
        var capture = new CliFixtures.Capture("Hello from stdin.");
        assertEquals(
                0,
                run(
                        capture,
                        "speak",
                        "-m",
                        path.toString(),
                        "-",
                        "--with",
                        "voice=" + voice,
                        "--speed",
                        "1.25",
                        "-o",
                        "-"),
                capture.err());
        assertEquals(voice, CliModelProvider.attachments.get("voice"));
        assertEquals("Hello from stdin.", CliModelProvider.speech.text);
        assertEquals(1.25, CliModelProvider.speech.speed);
        byte[] wav = capture.stdout.toByteArray();
        assertEquals(wav.length, ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN).getInt(4) + 8);
        assertEquals(1, CliModelProvider.speech.closed);
        assertFalse(CliModelProvider.weights.scope().isAlive());
    }

    @Test
    void failedSpeechLeavesAnExistingOutputUntouched() throws Exception {
        Path path = model("speech", "generate");
        Path output = Files.writeString(dir.resolve("existing.wav"), "keep this");
        var capture = new CliFixtures.Capture("");
        assertEquals(
                1, run(capture, "speak", "-m", path.toString(), "Hello", "-o", output.toString()));
        assertEquals("keep this", Files.readString(output));
        assertEquals(1, CliModelProvider.speech.closed);
        assertFalse(CliModelProvider.weights.scope().isAlive());
        assertEquals("", capture.out());
    }

    @Test
    void audioFilesAndEncodedStdinReachTheSameTranscriptionApplication() throws Exception {
        Path path = model("transcription", "");
        byte[] wav = AudioCodec.wav(new Media.Audio(new float[] {0.2f, -0.2f, 0}, 16000, 1));
        Path audio = Files.write(dir.resolve("recording.wav"), wav);
        for (String input : List.of(audio.toString(), "-")) {
            var capture = new CliFixtures.Capture(wav);
            assertEquals(
                    0, run(capture, "transcribe", "-m", path.toString(), input), capture.err());
            assertEquals("heard\n", capture.out().replace("\r\n", "\n"));
            assertEquals(3, CliModelProvider.transcription.received.length);
            assertEquals(1, CliModelProvider.transcription.closed);
            assertFalse(CliModelProvider.weights.scope().isAlive());
            assertFalse(capture.inputClosed);
        }
    }

    @Test
    void rawStdinWarmsAndClosesItsStatesWhileLegacyInputRemainsCompatible() throws Exception {
        Path path = model("transcription", "");
        for (String[] args :
                new String[][] {
                    {"transcribe", "-m", path.toString(), "-", "--raw-pcm"},
                    {"-m", path.toString(), "--transcribe", "-"}
                }) {
            var capture = new CliFixtures.Capture(new byte[] {0, 0});
            assertEquals(0, run(capture, args), capture.err());
            assertEquals("hello done\n", capture.out().replace("\r\n", "\n"));
            assertEquals(1, CliModelProvider.transcription.finished);
            assertEquals(
                    2,
                    CliModelProvider.transcription.closed,
                    "warm-up and actual transcription each own a state");
            assertFalse(CliModelProvider.weights.scope().isAlive());
        }
    }

    @Test
    void corruptAudioFailsBeforeTheModelLoads() throws Exception {
        Path path = model("transcription", "");
        var capture = new CliFixtures.Capture("not audio");
        assertEquals(1, run(capture, "transcribe", "-m", path.toString(), "-"));
        assertEquals(0, CliModelProvider.transcriptionLoads);
        assertNull(CliModelProvider.weights);
        assertEquals("", capture.out());
        assertTrue(
                capture.err().startsWith("jinfer transcribe: cannot decode audio from stdin"),
                capture.err());
        assertTrue(capture.err().contains("\n  "), "decoder details follow the summary");
        assertFalse(capture.err().contains("\tat "));
    }

    @Test
    void resolutionAndLoadFailuresAreOperationalErrorsWithoutStackTraces() throws Exception {
        Path failing = model("language", "load");
        var load = new CliFixtures.Capture("");
        assertEquals(1, run(load, "instruct", "-m", failing.toString(), "Hello"));
        assertTrue(load.err().contains("fixture load failed"));
        assertFalse(CliModelProvider.weights.scope().isAlive());
        for (String source :
                List.of(dir.resolve("missing.gguf").toString(), "owner/uncached:Q8_0")) {
            var missing = new CliFixtures.Capture("");
            assertEquals(1, run(missing, "chat", "-m", source));
            assertEquals("", missing.out());
            assertFalse(missing.err().contains("Exception in thread"));
            assertFalse(missing.err().contains("\tat "));
        }
    }

    @Test
    void unexpectedFailuresKeepTheirTracesCausesAndSuppressedDetails() throws Exception {
        Path path = model("language", "bug");
        for (String verb : List.of("chat", "instruct")) {
            var capture = new CliFixtures.Capture("hello\nsecond turn\n");
            String[] args =
                    verb.equals("chat")
                            ? new String[] {verb, "-m", path.toString()}
                            : new String[] {verb, "-m", path.toString(), "hello"};
            assertEquals(
                    1,
                    run(capture, args),
                    "a broken chat model must not continue as if a turn was merely refused");
            assertTrue(
                    capture.err()
                            .startsWith(
                                    "jinfer "
                                            + verb
                                            + ": unexpected failure: fixture internal failure"));
            assertTrue(capture.err().contains("java.lang.IllegalStateException"));
            assertTrue(capture.err().contains("\tat "));
            assertTrue(capture.err().contains("Caused by: java.io.IOException: original cause"));
            assertTrue(capture.err().contains("Suppressed: java.io.IOException: cleanup detail"));
            assertFalse(CliModelProvider.weights.scope().isAlive());
        }
    }

    @Test
    void outputFailuresNameTheOperationAndTheDestination() throws Exception {
        Path path = model("speech", "");
        Path destination = dir.resolve("missing-parent/hello.wav");
        var capture = new CliFixtures.Capture("");
        assertEquals(
                1,
                run(
                        capture,
                        "speak",
                        "-m",
                        path.toString(),
                        "hello",
                        "--output",
                        destination.toString()));
        assertTrue(
                capture.err().startsWith("jinfer speak: cannot write WAV to '" + destination + "'"),
                capture.err());
        assertTrue(capture.err().contains("--output"));
        assertFalse(capture.err().contains("\tat "));
    }

    @Test
    void unsupportedCompanionsFailBeforeModelConstruction() throws Exception {
        Path path = model("language", "");
        Path extra = Files.writeString(dir.resolve("extra.gguf"), "extra");
        var capture = new CliFixtures.Capture("");
        assertEquals(1, run(capture, "chat", "-m", path.toString(), "--with", "missing=" + extra));
        assertTrue(capture.err().contains("no 'missing' capability"));
        assertEquals(0, CliModelProvider.languageLoads);
    }

    @Test
    void corruptTokenizerOverrideNamesTheFile() throws Exception {
        Path path = model("language", "");
        for (String content : List.of("bad", "nope")) {
            Path tokenizer = Files.writeString(dir.resolve("bad-tokenizer.gguf"), content);
            var capture = new CliFixtures.Capture("");
            assertEquals(
                    1,
                    run(
                            capture,
                            "chat",
                            "-m",
                            path.toString(),
                            "--with",
                            "tokenizer=" + tokenizer));
            assertTrue(capture.err().contains("bad-tokenizer.gguf"), capture.err());
            assertEquals(0, CliModelProvider.languageLoads);
        }
        Path noVocabulary = dir.resolve("not-a-tokenizer.gguf");
        GGUF.write(
                Builder.newBuilder().putString("general.architecture", "cli_test_speech").build(),
                noVocabulary);
        var wrongKind = new CliFixtures.Capture("");
        assertEquals(
                1,
                run(
                        wrongKind,
                        "chat",
                        "-m",
                        path.toString(),
                        "--with",
                        "tokenizer=" + noVocabulary));
        assertTrue(wrongKind.err().contains("not-a-tokenizer.gguf"), wrongKind.err());
    }

    @Test
    void malformedModelsReturnOperationalErrorsRatherThanEscapingTheCli() throws Exception {
        Path file = dir.resolve("invalid-model.gguf");
        for (String content : List.of("bad", "nope")) {
            Files.writeString(file, content);
            var capture = new CliFixtures.Capture("");
            assertEquals(1, run(capture, "chat", "-m", file.toString()));
            assertTrue(capture.err().contains("not a GGUF"), capture.err());
            assertEquals("", capture.out());
        }
        Path noArchitecture = dir.resolve("missing-architecture.gguf");
        GGUF.write(Builder.newBuilder().build(), noArchitecture);
        var missingArchitecture = new CliFixtures.Capture("");
        assertEquals(1, run(missingArchitecture, "chat", "-m", noArchitecture.toString()));
        assertTrue(missingArchitecture.err().contains("architecture"), missingArchitecture.err());
    }

    @Test
    void zeroOutputBudgetAndRawPromptAreValidOneShotWorkflows() throws Exception {
        Path path = model("language", "");
        var zero = new CliFixtures.Capture("");
        assertEquals(0, run(zero, "instruct", "-m", path.toString(), "hi", "-n", "0"), zero.err());
        assertTrue(zero.out().isBlank());
        var raw = new CliFixtures.Capture("");
        assertEquals(
                0,
                run(
                        raw,
                        "instruct",
                        "-m",
                        path.toString(),
                        "raw input",
                        "--raw-prompt",
                        "--echo",
                        "-n",
                        "3"),
                raw.err());
        assertEquals("xxx\n", raw.out().replace("\r\n", "\n"));
        assertTrue(raw.err().contains("raw input"));
        assertTrue(CliModelProvider.template.conversations.isEmpty());
        assertFalse(CliModelProvider.weights.scope().isAlive());
    }

    @Test
    void promptCacheCanBeWrittenAndReopenedReadOnlyWithoutWeights() throws Exception {
        Path path = model("language", "");
        Path cache = dir.resolve("prompt.jkv");
        var first = new CliFixtures.Capture("");
        assertEquals(
                0,
                run(
                        first,
                        "instruct",
                        "-m",
                        path.toString(),
                        "hello",
                        "-n",
                        "2",
                        "--cache",
                        cache.toString()),
                first.err());
        long size = Files.size(cache);
        var second = new CliFixtures.Capture("");
        assertEquals(
                0,
                run(
                        second,
                        "instruct",
                        "-m",
                        path.toString(),
                        "hello",
                        "-n",
                        "2",
                        "--cache-ro",
                        cache.toString()),
                second.err());
        assertEquals(first.out(), second.out());
        assertEquals(size, Files.size(cache));
        assertTrue(second.err().contains("restored"));
    }

    @Test
    void automaticServerSelectionStartsAndStopsBothModelKinds() throws Exception {
        for (String kind : List.of("language", "transcription")) {
            Path path = model(kind, "");
            var capture = new CliFixtures.Capture("");
            CountDownLatch ready = new CountDownLatch(1);
            var errors =
                    new java.io.PrintStream(
                            capture.stderr, true, java.nio.charset.StandardCharsets.UTF_8) {
                        @Override
                        public java.io.PrintStream printf(String format, Object... args) {
                            var result = super.printf(format, args);
                            if (format.startsWith("listening")) ready.countDown();
                            return result;
                        }
                    };
            var io = new Main.IO(capture.io.in(), capture.io.out(), errors);
            AtomicInteger status = new AtomicInteger(-1);
            AtomicReference<Throwable> failure = new AtomicReference<>();
            Thread server =
                    new Thread(
                            () -> {
                                try {
                                    status.set(
                                            Main.run(
                                                    new String[] {
                                                        "server",
                                                        "-m",
                                                        path.toString(),
                                                        "--port",
                                                        "0"
                                                    },
                                                    io,
                                                    ModelStore.of(dir)));
                                } catch (Throwable error) {
                                    failure.set(error);
                                } finally {
                                    ready.countDown();
                                }
                            },
                            "cli-test-server");
            server.setDaemon(true);
            server.start();
            try {
                assertTrue(ready.await(10, TimeUnit.SECONDS), capture.err());
                assertNull(failure.get());
                assertTrue(capture.err().contains("listening"), capture.err());
                assertTrue(
                        capture.err()
                                .contains(
                                        kind.equals("language")
                                                ? "OpenAI-compatible"
                                                : "/v1/audio/transcriptions"));
                assertEquals("", capture.out());
                var address = java.util.regex.Pattern.compile("http://\\S+").matcher(capture.err());
                assertTrue(address.find(), capture.err());
                String base = address.group();
                try (var client = java.net.http.HttpClient.newHttpClient()) {
                    java.net.http.HttpRequest request;
                    if (kind.equals("language")) {
                        request =
                                java.net.http.HttpRequest.newBuilder(
                                                java.net.URI.create(base + "/health"))
                                        .timeout(java.time.Duration.ofSeconds(5))
                                        .GET()
                                        .build();
                    } else {
                        var body = new java.io.ByteArrayOutputStream();
                        body.write(
                                "--test-boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"clip.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
                                        .getBytes(java.nio.charset.StandardCharsets.UTF_8));
                        body.write(
                                AudioCodec.wav(
                                        new Media.Audio(
                                                new float[] {0, 0.1f, -0.1f, 0}, 16000, 1)));
                        body.write(
                                "\r\n--test-boundary--\r\n"
                                        .getBytes(java.nio.charset.StandardCharsets.UTF_8));
                        request =
                                java.net.http.HttpRequest.newBuilder(
                                                java.net.URI.create(
                                                        base + "/v1/audio/transcriptions"))
                                        .timeout(java.time.Duration.ofSeconds(5))
                                        .header(
                                                "Content-Type",
                                                "multipart/form-data; boundary=test-boundary")
                                        .POST(
                                                java.net.http.HttpRequest.BodyPublishers
                                                        .ofByteArray(body.toByteArray()))
                                        .build();
                    }
                    var response =
                            client.send(
                                    request, java.net.http.HttpResponse.BodyHandlers.ofString());
                    assertEquals(200, response.statusCode(), response.body());
                    if (kind.equals("transcription"))
                        assertEquals(
                                "heard",
                                com.qxotic.format.json.Json.parseMap(response.body()).get("text"));
                }
            } finally {
                server.interrupt();
                server.join(35000);
            }
            assertFalse(server.isAlive(), "server did not stop");
            assertNull(failure.get());
            assertEquals(130, status.get(), capture.err());
            assertFalse(CliModelProvider.weights.scope().isAlive());
        }
    }

    @Test
    void serverSelectsTranscriptionFromTheModelAndReportsLoadFailures() throws Exception {
        Path path = model("transcription", "load");
        var capture = new CliFixtures.Capture("");
        assertEquals(1, run(capture, "serve", "-m", path.toString(), "--port", "0"));
        assertEquals(1, CliModelProvider.languageLoads);
        assertEquals(1, CliModelProvider.transcriptionLoads);
        assertTrue(capture.err().contains("fixture load failed"));
        assertFalse(CliModelProvider.weights.scope().isAlive());
    }

    @Test
    void speechOnlyModelsHaveAClearUnsupportedServerError() throws Exception {
        Path path = model("speech", "");
        var capture = new CliFixtures.Capture("");
        assertEquals(1, run(capture, "server", "-m", path.toString(), "--port", "0"));
        assertTrue(capture.err().contains("cannot be served"));
        assertEquals(0, CliModelProvider.speechLoads, "serving must never synthesize speech");
    }
}
