package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.media.Media;
import com.qxotic.jota.memory.MemoryArena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/** The stream interface's defaults and the audio format check, on fakes. */
class TranscriptionStreamTest {

    private static final int RATE = 16_000;

    /** Records every primitive {@code feed} call as {@code {offset, length}}. */
    private static final class Recording implements TranscriptionStream {
        final List<int[]> feeds = new ArrayList<>();
        float[] lastPcm;

        @Override
        public int sampleRate() {
            return RATE;
        }

        @Override
        public Transcription feed(float[] pcm, int offset, int length) {
            lastPcm = pcm;
            feeds.add(new int[] {offset, length});
            return Transcription.empty();
        }

        @Override
        public Transcription partial() {
            return Transcription.empty();
        }

        @Override
        public Transcription finish() {
            return Transcription.empty();
        }

        @Override
        public void close() {}
    }

    @Test
    void feedingAWholeArrayFeedsAllOfIt() {
        Recording stream = new Recording();
        float[] pcm = new float[123];
        stream.feed(pcm);
        assertSame(pcm, stream.lastPcm);
        assertArrayEquals(new int[] {0, 123}, stream.feeds.getFirst());
    }

    @Test
    void decodedAudioInTheStreamsFormatFeedsItsSamples() {
        Recording stream = new Recording();
        float[] pcm = {0.1f, -0.2f, 0.3f};
        stream.feed(new Media.Audio(pcm, RATE, 1));
        assertSame(pcm, stream.lastPcm);
        assertArrayEquals(new int[] {0, 3}, stream.feeds.getFirst());
    }

    @Test
    void decodedAudioInAnotherFormatIsRefusedBeforeFeeding() {
        Recording stream = new Recording();
        var wrongRate = new Media.Audio(new float[8], 8_000, 1);
        var stereo = new Media.Audio(new float[8], RATE, 2);
        var error = assertThrows(IllegalArgumentException.class, () -> stream.feed(wrongRate));
        assertTrue(error.getMessage().contains("8000 Hz x1"), error.getMessage());
        assertTrue(error.getMessage().contains("mono 16000 Hz"), error.getMessage());
        assertThrows(IllegalArgumentException.class, () -> stream.feed(stereo));
        assertTrue(stream.feeds.isEmpty(), "nothing is fed after a refusal");
    }

    /** A model whose {@code transcribe} records the samples it was given. */
    private static final class Model implements TranscriptionModel<Void, Void, Model.State> {
        static final class State extends RuntimeState {
            @Override
            protected void releaseResources() {}
        }

        float[] transcribed;

        @Override
        public Void configuration() {
            return null;
        }

        @Override
        public Void weights() {
            return null;
        }

        @Override
        public int sampleRate() {
            return RATE;
        }

        @Override
        public State newState() {
            return new State();
        }

        @Override
        public State newState(MemoryArena<MemorySegment> arena) {
            return new State();
        }

        @Override
        public Transcription transcribe(State state, float[] pcm) {
            transcribed = pcm;
            return Transcription.empty();
        }
    }

    @Test
    void theModelChecksDecodedAudioTheSameWay() {
        Model model = new Model();
        float[] pcm = new float[4];
        model.transcribe(new Media.Audio(pcm, RATE, 1));
        assertSame(pcm, model.transcribed);
        model.transcribed = null;
        assertThrows(
                IllegalArgumentException.class,
                () -> model.transcribe(new Media.Audio(new float[4], 44_100, 1)));
        assertEquals(null, model.transcribed);
    }

    @Test
    void aModelThatCannotStreamRefusesToOpenAStream() {
        Model model = new Model();
        try (Model.State state = model.newState()) {
            assertThrows(UnsupportedOperationException.class, () -> model.stream(state));
        }
    }
}
