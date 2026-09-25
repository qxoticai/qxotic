package com.qxotic.jinfer.cli;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.jinfer.Transcription;
import com.qxotic.jinfer.TranscriptionModel;
import com.qxotic.jinfer.TranscriptionStream;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jota.memory.MemoryArena;
import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.MemorySegment;
import java.time.Duration;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class TranscribeTest {
    @Test
    void decodedAudioReportsProgressToStderrAndOnlyTheTranscriptToStdout() throws Exception {
        var model = new Transcriber();
        var capture = new CliFixtures.Capture("");
        Options o = Options.parse("transcribe", "-m", "unused", "input.wav");
        var audio = new Media.Audio(new float[32000], 16000, 1);
        model.onTranscribe =
                () -> assertTrue(capture.err().contains("Transcribing 2.00 s of audio"));
        Transcribe.execute(model, audio, o, capture.io);
        assertArrayEquals(audio.pcm(), model.received);
        assertEquals("heard\n", capture.out().replace("\r\n", "\n"));
        assertTrue(capture.err().contains("Transcribed 2.00 s of audio in "), capture.err());
        assertTrue(capture.err().contains("RTFx "), capture.err());
        assertEquals(1, model.closed);
        assertFalse(capture.inputClosed);
    }

    @Test
    void summaryReportsAudioOverElapsedTimeAndHandlesEmptyAudio() {
        var capture = new CliFixtures.Capture("");
        Transcribe.printSummary(30, 2_000_000_000L, capture.io.err());
        Transcribe.printSummary(0, 0, capture.io.err());
        assertEquals(
                "Transcribed 30.00 s of audio in 2.00 s (RTFx 15.00)\n"
                        + "Transcribed 0.00 s of audio in 0.00 s (RTFx 0.00)\n",
                capture.err().replace("\r\n", "\n"));
    }

    @Test
    void failedTranscriptionDoesNotReportCompletion() {
        var model = new Transcriber();
        model.onTranscribe =
                () -> {
                    throw new IllegalStateException("decode failed");
                };
        var capture = new CliFixtures.Capture("");
        assertThrows(
                IllegalStateException.class,
                () ->
                        Transcribe.execute(
                                model,
                                new Media.Audio(new float[16000], 16000, 1),
                                Options.parse("transcribe", "-m", "unused", "input.wav"),
                                capture.io));
        assertTrue(capture.err().contains("Transcribing 1.00 s of audio"));
        assertFalse(capture.err().contains("Transcribed"));
        assertEquals("", capture.out());
        assertEquals(1, model.closed);
    }

    @Test
    void rawPcmConversionAndEofFinishExactlyOnce() throws Exception {
        var model = new Transcriber();
        var capture = new CliFixtures.Capture(new byte[] {0, (byte) 0x80, 0, 0, (byte) 0xff, 0x7f});
        Options o = Options.parse("transcribe", "-m", "unused", "-", "--raw-pcm");
        Transcribe.execute(model, null, o, capture.io);
        assertArrayEquals(new float[] {-1, 0, 32767 / 32768f}, model.received);
        assertEquals("hello done\n", capture.out().replace("\r\n", "\n"));
        assertEquals(1, model.finished);
        assertEquals(1, model.streamClosed);
        assertEquals(1, model.closed);
        assertFalse(capture.inputClosed);
    }

    @Test
    void streamingTextDoesNotRequireTokenTimestamps() throws Exception {
        var model = new Transcriber();
        model.timestamps = false;
        var capture = new CliFixtures.Capture(new byte[] {0, 0});
        Options o = Options.parse("transcribe", "-m", "unused", "-", "--raw-pcm");
        Transcribe.execute(model, null, o, capture.io);
        assertEquals("hello done\n", capture.out().replace("\r\n", "\n"));
        assertEquals(1, model.finished);
    }

    @Test
    void interruptedInputClosesWithoutStartingAFinalDecode() {
        var model = new Transcriber();
        var capture = new CliFixtures.Capture(new byte[] {0, 0});
        Options o = Options.parse("transcribe", "-m", "unused", "-", "--raw-pcm");
        Thread.currentThread().interrupt();
        try {
            assertThrows(
                    java.io.InterruptedIOException.class,
                    () -> Transcribe.execute(model, null, o, capture.io));
            assertTrue(Thread.currentThread().isInterrupted());
            assertEquals(0, model.finished);
            assertEquals(1, model.streamClosed);
            assertEquals(1, model.closed);
            assertEquals("", capture.out());
            assertFalse(capture.err().contains("Transcribed"));
            assertFalse(capture.inputClosed);
        } finally {
            Thread.interrupted();
        }
    }

    @Test
    void truncatedPcmAndReaderFailureCloseTheStreamWithoutFinishing() {
        for (InputStream input :
                new InputStream[] {
                    new java.io.ByteArrayInputStream(new byte[] {1, 2, 3}),
                    new InputStream() {
                        public int read() throws IOException {
                            throw new IOException("test read failure");
                        }
                    }
                }) {
            var model = new Transcriber();
            var capture = new CliFixtures.Capture("");
            var io = new Main.IO(input, capture.io.out(), capture.io.err());
            Options o = Options.parse("transcribe", "-m", "unused", "-", "--raw-pcm");
            assertThrows(IOException.class, () -> Transcribe.execute(model, null, o, io));
            assertEquals(0, model.finished);
            assertEquals(1, model.streamClosed);
            assertEquals(1, model.closed);
            assertEquals("", capture.out());
            assertFalse(capture.err().contains("Transcribed"));
        }
    }

    @Test
    void liveInputReportsPartialsUntilEofWithoutMixingThemIntoStdout() throws Exception {
        var model = new Transcriber();
        CountDownLatch partial = new CountDownLatch(1);
        model.onPartial = partial::countDown;
        // Half a second of audio, then keep the source open until a partial is requested.
        InputStream live =
                new java.io.ByteArrayInputStream(new byte[16000]) {
                    @Override
                    public synchronized int read(byte[] bytes, int offset, int length) {
                        if (available() == 0) {
                            try {
                                assertTrue(
                                        partial.await(5, TimeUnit.SECONDS),
                                        "no partial while stdin was open");
                            } catch (InterruptedException e) {
                                Thread.currentThread().interrupt();
                                return -1;
                            }
                        }
                        return super.read(bytes, offset, length);
                    }
                };
        var capture = new CliFixtures.Capture("");
        var io = new Main.IO(live, capture.io.out(), capture.io.err());
        Transcribe.execute(
                model, null, Options.parse("transcribe", "-m", "unused", "-", "--raw-pcm"), io);
        assertEquals(1, model.partials);
        assertTrue(capture.err().contains("partial"));
        assertTrue(capture.err().contains("Transcribing raw PCM from stdin"));
        assertTrue(capture.err().contains("Transcribed 0.50 s of audio in "), capture.err());
        assertTrue(capture.err().contains("RTFx "));
        assertFalse(capture.out().contains("partial"));
        assertEquals("hello hello hello hello done\n", capture.out().replace("\r\n", "\n"));
        assertEquals(1, model.finished);
    }

    @Test
    void unsupportedStreamingAndWrongSampleRatesFailWithoutLeakingState() {
        var capture = new CliFixtures.Capture(new byte[] {0, 0});
        Options o = Options.parse("transcribe", "-m", "unused", "-", "--raw-pcm");
        var unsupported = new Transcriber();
        unsupported.streaming = false;
        assertThrows(
                UnsupportedOperationException.class,
                () -> Transcribe.execute(unsupported, null, o, capture.io));
        assertEquals(1, unsupported.closed);
        var wrongRate = new Transcriber();
        wrongRate.rate = 48000;
        assertThrows(
                IllegalArgumentException.class,
                () -> Transcribe.execute(wrongRate, null, o, capture.io));
        assertEquals(0, wrongRate.closed, "format rejection precedes state allocation");
    }

    private static Transcription text(String value) {
        return new Transcription(
                value,
                List.of(new Transcription.Token(value, Duration.ZERO, Duration.ofMillis(100), 1f)));
    }

    static final class Transcriber implements TranscriptionModel<Void, Void, CliFixtures.State> {
        float[] received;
        int closed, finished, streamClosed, partials;
        boolean timestamps = true;
        boolean streaming = true;
        int rate = 16000;
        Runnable onPartial = () -> {};
        Runnable onTranscribe = () -> {};

        private Transcription result(String value) {
            return timestamps ? text(value) : new Transcription(value, List.of());
        }

        public Void configuration() {
            return null;
        }

        public Void weights() {
            return null;
        }

        public int sampleRate() {
            return rate;
        }

        public CliFixtures.State newState() {
            return new CliFixtures.State(() -> closed++);
        }

        public CliFixtures.State newState(MemoryArena<MemorySegment> arena) {
            return newState();
        }

        public Transcription transcribe(CliFixtures.State state, float[] pcm) {
            onTranscribe.run();
            received = pcm.clone();
            return result("heard");
        }

        public TranscriptionStream stream(CliFixtures.State state) {
            if (!streaming) throw new UnsupportedOperationException("streaming is not supported");
            return new TranscriptionStream() {
                public int sampleRate() {
                    return 16000;
                }

                public Transcription feed(float[] pcm, int offset, int length) {
                    received = Arrays.copyOfRange(pcm, offset, offset + length);
                    return result("hello ");
                }

                public Transcription partial() {
                    partials++;
                    onPartial.run();
                    return result("partial");
                }

                public Transcription finish() {
                    finished++;
                    return result("done");
                }

                public void close() {
                    streamClosed++;
                }
            };
        }
    }
}
