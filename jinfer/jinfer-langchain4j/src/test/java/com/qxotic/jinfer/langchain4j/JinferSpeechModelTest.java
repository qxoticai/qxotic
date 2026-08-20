package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.SpeechOptions;
import com.qxotic.jinfer.SpeechSynthesisModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jota.memory.MemoryArena;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.audio.TextToSpeechRequest;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Predicate;
import org.junit.jupiter.api.Test;

/** The adapter's own behaviour, over a toy model: no GGUF, no weights, no kernels. */
final class JinferSpeechModelTest {

    @Test
    void aModelIsRequired() {
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class, () -> JinferSpeechModel.builder().build());
        assertEquals(
                "a model is required: model(\"hf.co/owner/repo:Q4_K_M\"), modelPath(...) or"
                        + " model(SpeechSynthesisModel)",
                e.getMessage());
    }

    @Test
    void theLastModelSetterWins() {
        // the setters clear one another: the toy model set LAST wins over the bogus path, which
        // is therefore never opened - a successful build IS the assertion
        JinferSpeechModel.builder()
                .modelPath(Path.of("/nonexistent.gguf"))
                .model(new ToyModel())
                .build()
                .close();
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
            UnsupportedFeatureException e =
                    assertThrows(
                            UnsupportedFeatureException.class,
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
                        .modelPath(
                                TestModels.require(
                                        "hf.co/remixerdec/Inflect-Nano-v2-GGUF/inflect_nano_v2_q8_0.gguf"))
                        .build();
        speech.close();
        speech.close();
    }

    @Test
    void aCallersArenaOutlivesTheAdapter() throws Exception {
        try (Arena weights = Arenas.newCrossThread()) {
            JinferSpeechModel.builder()
                    .model(
                            Models.loadSpeech(
                                    TestModels.require(
                                            "hf.co/remixerdec/Inflect-Nano-v2-GGUF/inflect_nano_v2_q8_0.gguf"),
                                    weights))
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
    private static final class SlowModel implements SpeechSynthesisModel<Void, Void, ToyState> {

        private final CountDownLatch entered;
        private final CountDownLatch release;
        ToyState state;

        SlowModel(CountDownLatch entered, CountDownLatch release) {
            this.entered = entered;
            this.release = release;
        }

        @Override
        public Void configuration() {
            return null;
        }

        @Override
        public Void weights() {
            return null;
        }

        @Override
        public ToyState newState() {
            return state = new ToyState();
        }

        @Override
        public ToyState newState(MemoryArena<MemorySegment> arena) {
            return newState();
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
                                    public ToyState newState() {
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
        Path gguf =
                TestModels.require(
                        "hf.co/remixerdec/Inflect-Nano-v2-GGUF/inflect_nano_v2_q8_0.gguf");
        try (var speech = JinferSpeechModel.builder().modelPath(gguf).build()) {
            var audio = speech.synthesize("Speech from a real model.").audio();
            assertEquals("audio/wav", audio.mimeType());
            assertTrue(audio.binaryData().length > 44 + 2 * 8000, "at least a third of a second");
        }
    }

    /** Emits one fixed clip per call and remembers the state it handed out. */
    private static class ToyModel implements SpeechSynthesisModel<Void, Void, ToyState> {

        static final int SAMPLES = 8;

        ToyState state; // the most recent one
        final AtomicInteger minted = new AtomicInteger();
        final List<ToyState> all = Collections.synchronizedList(new ArrayList<>());

        @Override
        public Void configuration() {
            return null;
        }

        @Override
        public Void weights() {
            return null;
        }

        @Override
        public ToyState newState() {
            minted.incrementAndGet();
            ToyState s = new ToyState();
            all.add(s);
            return state = s;
        }

        @Override
        public ToyState newState(MemoryArena<MemorySegment> arena) {
            return newState();
        }

        @Override
        public void speak(
                ToyState state, String text, SpeechOptions options, Predicate<Media.Audio> sink) {
            sink.test(new Media.Audio(new float[SAMPLES], 24000, 1));
        }
    }

    private static final class ToyState extends RuntimeState {
        boolean closed;

        @Override
        protected void releaseResources() {
            closed = true;
        }
    }
}
