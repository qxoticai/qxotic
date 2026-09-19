package com.qxotic.jinfer.models.parakeet;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Transcription;
import com.qxotic.jinfer.TranscriptionStream;
import com.qxotic.jinfer.testkit.TestModels;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/**
 * The streaming contract over the offline model: ragged feeding, mid-stream partials, and the
 * guarantee that {@code finish()} equals the offline transcript - including across window commits.
 */
class ParakeetStreamTest {

    @Test
    void streamedFinishEqualsOfflineTranscription() throws IOException {
        Optional<Path> fixturePath = Fixtures.fixture("tdt-0.6b-v3-f16-jfk.fixture.gguf");
        assumeTrue(fixturePath.isPresent(), "parakeet fixture not checked out");
        Path model = TestModels.require("mudler/parakeet-cpp-gguf/tdt-0.6b-v3-f16.gguf");

        try (FileChannel channel = FileChannel.open(fixturePath.get(), StandardOpenOption.READ);
                Arena arena = Arena.ofShared()) {
            GGUF fixture = GGUF.read(fixturePath.get());
            float[] pcm = Fixtures.floats(channel, fixture, "pcm");
            Parakeet parakeet;
            try (FileChannel modelChannel = FileChannel.open(model, StandardOpenOption.READ)) {
                parakeet =
                        Parakeet.load(
                                modelChannel,
                                com.qxotic.jinfer.kernels.ModelLoader.readGguf(
                                        modelChannel, model.toString()),
                                arena);
            }

            // Single-window path: ragged chunks, a mid-stream partial, exact final equality.
            String offline = parakeet.transcribe(pcm).text();
            try (Parakeet.State state = parakeet.newState()) {
                TranscriptionStream stream = parakeet.stream(state);
                stream.feed(pcm, 0, 30_000);
                stream.feed(pcm, 30_000, 70_000);
                Transcription midway = stream.partial();
                assertFalse(midway.text().isEmpty(), "mid-stream partial is empty");
                stream.feed(pcm, 100_000, pcm.length - 100_000);
                Transcription finished = stream.finish();
                assertEquals(offline, finished.text());
                assertThrows(IllegalStateException.class, () -> stream.feed(pcm, 0, 100));
            }

            // Windowed path: 10 s windows over the 11 s clip force a commit mid-stream. The
            // committed prefix must only ever grow, and always prefix the partial - the contract
            // a live renderer builds on.
            System.setProperty("jinfer.parakeet.chunkSeconds", "10");
            // doubled audio: a 10 s window plus the commit lookahead needs >15 s to commit
            float[] doubled = new float[pcm.length * 2];
            System.arraycopy(pcm, 0, doubled, 0, pcm.length);
            System.arraycopy(pcm, 0, doubled, pcm.length, pcm.length);
            try (Parakeet.State state = parakeet.newState()) {
                String windowed = parakeet.transcribe(doubled).text();
                TranscriptionStream stream = parakeet.stream(state);
                String previousCommitted = "";
                for (int from = 0; from < doubled.length; from += 16_000) {
                    stream.feed(doubled, from, Math.min(16_000, doubled.length - from));
                    String committed = stream.committed().text();
                    assertTrue(committed.startsWith(previousCommitted), "committed text shrank");
                    assertTrue(
                            stream.partial().text().startsWith(committed),
                            "committed is not a prefix of partial");
                    previousCommitted = committed;
                }
                assertFalse(previousCommitted.isEmpty(), "nothing committed across a window");
                assertEquals(windowed, stream.finish().text());
            } finally {
                System.clearProperty("jinfer.parakeet.chunkSeconds");
            }
        }
    }
}
