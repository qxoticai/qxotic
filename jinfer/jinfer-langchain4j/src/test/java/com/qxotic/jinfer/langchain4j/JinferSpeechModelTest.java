package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Config;
import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.SpeechModel;
import com.qxotic.jinfer.SpeechOptions;
import com.qxotic.jinfer.SpeechState;
import com.qxotic.jinfer.testkit.ModelFixture;
import dev.langchain4j.model.audio.TextToSpeechRequest;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Predicate;
import org.junit.jupiter.api.Test;

/** The adapter's own behaviour, over a toy model: no GGUF, no weights, no kernels. */
final class JinferSpeechModelTest {

    @Test
    void aModelIsRequired() {
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class, () -> JinferSpeechModel.builder().build());
        assertEquals("exactly one of model(...) or modelPath(...) is required", e.getMessage());
    }

    @Test
    void anArenaIsForTheLoadingPathOnly() {
        // a model the caller built already sits in an arena; a second one would own nothing
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                JinferSpeechModel.builder()
                                        .model(new ToyModel())
                                        .arena(Arena.ofAuto())
                                        .build());
        assertTrue(e.getMessage().contains("arena"), e.getMessage());
    }

    @Test
    void bothAModelAndAPathIsAmbiguous() {
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        JinferSpeechModel.builder()
                                .model(new ToyModel())
                                .modelPath(Path.of("/nonexistent.gguf"))
                                .build());
    }

    @Test
    void synthesisReturnsWav() {
        try (var speech = JinferSpeechModel.builder().model(new ToyModel()).build()) {
            var audio = speech.synthesize("hello").audio();
            assertEquals("audio/wav", audio.mimeType());
            // 44-byte RIFF header + one 16-bit sample per PCM float
            assertEquals(44 + 2 * ToyModel.SAMPLES, audio.binaryData().length);
        }
    }

    @Test
    void aVoiceThisModelDoesNotHaveIsRefusedRatherThanIgnored() {
        try (var speech = JinferSpeechModel.builder().model(new ToyModel()).build()) {
            UnsupportedOperationException e =
                    assertThrows(
                            UnsupportedOperationException.class,
                            () ->
                                    speech.synthesize(
                                            TextToSpeechRequest.builder("hi")
                                                    .voice("alloy")
                                                    .build()));
            assertTrue(e.getMessage().contains("alloy"), e.getMessage());
        }
    }

    @Test
    void anOversizedRequestIsRejectedBeforeAnySynthesis() {
        try (var speech =
                JinferSpeechModel.builder().model(new ToyModel()).maxInputChars(4).build()) {
            IllegalArgumentException e =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> speech.synthesize("far too long"));
            assertTrue(e.getMessage().contains("over the 4"), e.getMessage());
        }
    }

    @Test
    void everyRequestGetsItsOwnStateAndClosesIt() {
        // the contract is that a state cannot be shared, so the adapter does not share one
        ToyModel model = new ToyModel();
        try (var speech = JinferSpeechModel.builder().model(model).build()) {
            assertEquals(0, model.minted.get(), "no state until a request arrives");
            speech.synthesize("one");
            speech.synthesize("two");
            assertEquals(2, model.minted.get(), "one state per request");
            assertTrue(
                    model.all.stream().allMatch(s -> s.closed),
                    "each request must free the state it minted");
        }
    }

    @Test
    void closeIsIdempotentAndRejectsLaterRequests() {
        var speech = JinferSpeechModel.builder().model(new ToyModel()).build();
        speech.close();
        speech.close();
        assertThrows(IllegalStateException.class, () -> speech.synthesize("hello"));
    }

    @Test
    void aRequestAfterCloseFailsInsteadOfReadingFreedMemory() {
        var speech = JinferSpeechModel.builder().model(new ToyModel()).build();
        speech.close();
        assertThrows(IllegalStateException.class, () -> speech.synthesize("hello"));
    }

    @Test
    void closeIsIdempotentEvenWhenItOwnsTheArena() {
        // no arena given, so the adapter creates and OWNS one. Arena.close() is one-shot, so
        // without the closed flag the second call throws - and a container is not the only caller
        var speech =
                JinferSpeechModel.builder()
                        .modelPath(ModelFixture.INFLECT_NANO_V2_Q8.require())
                        .build();
        speech.close();
        speech.close();
    }

    @Test
    void aCallersArenaOutlivesTheAdapter() {
        try (Arena weights = Arena.ofShared()) {
            JinferSpeechModel.builder()
                    .modelPath(ModelFixture.INFLECT_NANO_V2_Q8.require())
                    .arena(weights)
                    .build()
                    .close();
            assertTrue(weights.scope().isAlive(), "the adapter closed an arena it did not create");
        }
    }

    @Test
    void closeBlocksUntilAnInFlightSynthesisReturns() throws Exception {
        // close frees the arena a running kernel is reading; returning from it is the caller's
        // quiescence certificate, so it must not overtake a synthesis in progress
        CountDownLatch inSynthesis = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        SlowModel model = new SlowModel(inSynthesis, release);
        var speech = JinferSpeechModel.builder().model(model).build();

        Thread request = new Thread(() -> speech.synthesize("hello"));
        request.start();
        assertTrue(inSynthesis.await(5, TimeUnit.SECONDS), "synthesis never started");

        AtomicBoolean closed = new AtomicBoolean();
        Thread closer =
                new Thread(
                        () -> {
                            speech.close();
                            closed.set(true);
                        });
        closer.start();
        Thread.sleep(100);
        assertTrue(!closed.get(), "close returned while a synthesis was still running");

        release.countDown();
        request.join(5000);
        closer.join(5000);
        assertTrue(closed.get(), "close never returned");
        assertTrue(model.state.closed, "the state was not freed");
    }

    /** Blocks inside speak until released, so close() has something in flight to wait for. */
    private static final class SlowModel implements SpeechModel<Config, Void, ToyState> {

        private final CountDownLatch entered;
        private final CountDownLatch release;
        ToyState state;

        SlowModel(CountDownLatch entered, CountDownLatch release) {
            this.entered = entered;
            this.release = release;
        }

        @Override
        public Config config() {
            throw new UnsupportedOperationException();
        }

        @Override
        public Void weights() {
            return null;
        }

        @Override
        public ToyState newState(Arena arena, boolean adopt) {
            return state = new ToyState();
        }

        @Override
        public void speak(
                ToyState state, String text, SpeechOptions options, Predicate<Media.Audio> sink) {
            entered.countDown();
            try {
                release.await();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            sink.test(new Media.Audio(new float[ToyModel.SAMPLES], 24000, 1));
        }
    }

    @Test
    void aFailingStateAllocationSurfacesOnTheRequest() {
        // states are minted per request now, so an allocation failure is a request failure - the
        // model still builds, and the failure does not leave the lifecycle lock held
        try (var speech =
                JinferSpeechModel.builder()
                        .model(
                                new ToyModel() {
                                    @Override
                                    public ToyState newState(Arena arena, boolean adopt) {
                                        throw new IllegalStateException("no scratch");
                                    }
                                })
                        .build()) {
            assertThrows(IllegalStateException.class, () -> speech.synthesize("hello"));
            speech.close(); // would deadlock if the read lock leaked on the failure path
        }
    }

    @Test
    void concurrentRequestsRunInParallelRatherThanSerializing() throws Exception {
        // the point of a per-call state: two callers must overlap. If the adapter serialized on
        // one state, the first request would hold the lock while parked in its sink and the
        // second would never arrive - this times out instead of passing.
        int callers = 4;
        CountDownLatch allInside = new CountDownLatch(callers);
        CountDownLatch release = new CountDownLatch(1);
        ToyModel model =
                new ToyModel() {
                    @Override
                    public void speak(
                            ToyState state,
                            String text,
                            SpeechOptions options,
                            Predicate<Media.Audio> sink) {
                        allInside.countDown();
                        try {
                            release.await(10, TimeUnit.SECONDS);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        }
                        sink.test(new Media.Audio(new float[ToyModel.SAMPLES], 24000, 1));
                    }
                };

        try (var speech = JinferSpeechModel.builder().model(model).build()) {
            for (int i = 0; i < callers; i++) new Thread(() -> speech.synthesize("hi")).start();
            assertTrue(
                    allInside.await(10, TimeUnit.SECONDS),
                    "requests did not overlap - the adapter serialized them");
            release.countDown();
        }
        assertEquals(callers, model.minted.get(), "one state per concurrent request");
    }

    @Test
    void loadsARealModelThroughArchitectureDispatch() {
        Path gguf = ModelFixture.INFLECT_NANO_V2_Q8.require();
        try (var speech = JinferSpeechModel.builder().modelPath(gguf).build()) {
            var audio = speech.synthesize("Speech from a real model.").audio();
            assertEquals("audio/wav", audio.mimeType());
            assertTrue(audio.binaryData().length > 44 + 2 * 8000, "at least a third of a second");
        }
    }

    /** Emits one fixed clip per call and remembers the state it handed out. */
    private static class ToyModel implements SpeechModel<Config, Void, ToyState> {

        static final int SAMPLES = 8;

        ToyState state; // the most recent one
        final java.util.concurrent.atomic.AtomicInteger minted =
                new java.util.concurrent.atomic.AtomicInteger();
        final java.util.List<ToyState> all =
                java.util.Collections.synchronizedList(new java.util.ArrayList<>());

        @Override
        public Config config() {
            throw new UnsupportedOperationException();
        }

        @Override
        public Void weights() {
            return null;
        }

        @Override
        public ToyState newState(Arena arena, boolean adopt) {
            minted.incrementAndGet();
            ToyState s = new ToyState();
            all.add(s);
            return state = s;
        }

        @Override
        public void speak(
                ToyState state, String text, SpeechOptions options, Predicate<Media.Audio> sink) {
            sink.test(new Media.Audio(new float[SAMPLES], 24000, 1));
        }
    }

    private static final class ToyState implements SpeechState {
        boolean closed;

        @Override
        public void close() {
            closed = true;
        }
    }
}
