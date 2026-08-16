package com.qxotic.jinfer.models.inflect2;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.PanamaMemoryArena;
import com.qxotic.jinfer.boundary.SpeechOptions;
import com.qxotic.jinfer.testkit.TestModels;
import java.lang.foreign.Arena;
import java.util.ConcurrentModificationException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

/**
 * The lifecycle laws a speech state shares with every {@code RuntimeState}. These are memory-safety
 * properties, not conveniences: every one of them is a use-after-free or a double-free if it stops
 * holding, and the FFM close handshake cannot save a kernel that reads a raw address out of a freed
 * arena.
 *
 * <p>Fixture-gated: the concurrency laws need a synthesis long enough to still be running when the
 * other thread acts, so they need real weights.
 */
final class Inflect2StateLifecycleTest {

    private static final String REF = "hf.co/remixerdec/Inflect-Nano-v2-GGUF:Q8_0";

    private static final String LONG_TEXT =
            "The quick brown fox jumps over the lazy dog. "
                    + "Machine learning models are trained on large datasets, and the results can "
                    + "be surprising. In practice the hardest part is the data pipeline. "
                    + "Teams often underestimate this, repeatedly, and at some length.";

    private static InflectTTS tts() throws Exception {
        return InflectTTS.load(TestModels.require(REF), Arena.ofAuto());
    }

    // ── ownership ─────────────────────────────────────────────────────────

    @Test
    void anOwnedStateIsReleasedByClose() throws Exception {
        InflectTTS tts = tts();
        Inflect2.State state = tts.newState();
        assertTrue(state.isAlive());
        state.close();
        assertFalse(state.isAlive());
    }

    @Test
    void aBorrowedArenaIsNeverTouchedByClose() throws Exception {
        InflectTTS tts = tts();
        try (Arena arena = Arena.ofShared()) {
            Inflect2.State state = tts.newState(new PanamaMemoryArena(arena)); // BORROWED
            state.close();
            assertTrue(arena.scope().isAlive(), "borrow: close must not free an arena it was lent");
        }
    }

    @Test
    void closeIsIdempotent() throws Exception {
        InflectTTS tts = tts();
        Inflect2.State state = tts.newState(); // owned ofShared
        state.close();
        state.close(); // Arena.close is one-shot: without the CAS this throws
        state.close();
        assertFalse(state.isAlive());
    }

    // ── the lock ──────────────────────────────────────────────────────────

    @Test
    void aSynthesisAfterCloseFailsInsteadOfReadingFreedMemory() throws Exception {
        InflectTTS tts = tts();
        Inflect2.State state = tts.newState();
        state.close();
        assertThrows(
                IllegalStateException.class, () -> tts.speak(state, "hello", SpeechOptions.NONE));
    }

    @Test
    void concurrentSynthesisOnOneStateFailsFastRatherThanCorrupting() throws Exception {
        InflectTTS tts = tts();
        try (Inflect2.State state = tts.newState()) {
            CountDownLatch running = new CountDownLatch(1);
            AtomicReference<Throwable> second = new AtomicReference<>();

            Thread first =
                    new Thread(
                            () ->
                                    // the sink runs INSIDE exclusive access, so the first clip
                                    // proves
                                    // the state is held - a latch before speak() proves nothing
                                    tts.speak(
                                            state,
                                            LONG_TEXT,
                                            SpeechOptions.NONE,
                                            clip -> {
                                                running.countDown();
                                                return true;
                                            }));
            first.start();
            assertTrue(running.await(5, TimeUnit.SECONDS));

            // the state is a single serial pipeline: a second thread must be refused, not queued -
            // queueing would hide the bug, and sharing the scratch would corrupt both waveforms
            Thread other =
                    new Thread(
                            () -> {
                                try {
                                    tts.speak(state, "hello", SpeechOptions.NONE);
                                } catch (Throwable t) {
                                    second.set(t);
                                }
                            });
            other.start();
            other.join(10_000);
            first.join(30_000);

            assertTrue(
                    second.get() instanceof ConcurrentModificationException,
                    "expected CME, got " + second.get());
        }
    }

    @Test
    void closeBlocksUntilAnInFlightSynthesisReturns() throws Exception {
        InflectTTS tts = tts();
        Inflect2.State state = tts.newState(); // OWNED: close actually frees, so it MUST wait
        CountDownLatch inSynthesis = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        AtomicBoolean closeReturned = new AtomicBoolean();

        // The sink runs inside exclusive access, so parking there holds the synthesis open for
        // as long as we like - deterministic, unlike racing a real workload.
        Thread worker =
                new Thread(
                        () ->
                                tts.speak(
                                        state,
                                        LONG_TEXT,
                                        SpeechOptions.NONE,
                                        clip -> {
                                            inSynthesis.countDown();
                                            try {
                                                release.await();
                                            } catch (InterruptedException e) {
                                                Thread.currentThread().interrupt();
                                            }
                                            return false; // one clip is enough
                                        }));
        worker.start();
        assertTrue(inSynthesis.await(10, TimeUnit.SECONDS), "synthesis never started");

        Thread closer =
                new Thread(
                        () -> {
                            state.close();
                            closeReturned.set(true);
                        });
        closer.start();
        Thread.sleep(200);
        assertFalse(
                closeReturned.get(),
                "close() returned while a synthesis held the state - it would have freed the arena"
                        + " the kernels are reading");

        release.countDown();
        closer.join(30_000);
        worker.join(30_000);
        assertTrue(closeReturned.get(), "close never returned after the synthesis finished");
        assertFalse(state.isAlive());
    }

    @Test
    void closingFromInsideOwnSynthesisIsRejectedRatherThanSelfFreeing() throws Exception {
        InflectTTS tts = tts();
        try (Inflect2.State state = tts.newState()) {
            AtomicReference<Throwable> thrown = new AtomicReference<>();
            tts.speak(
                    state,
                    "Hello there.",
                    SpeechOptions.NONE,
                    clip -> {
                        try {
                            state.close(); // would free the arena this very callback returns into
                        } catch (Throwable t) {
                            thrown.set(t);
                        }
                        return false;
                    });
            assertTrue(
                    thrown.get() instanceof IllegalStateException,
                    "expected ISE, got " + thrown.get());
            assertTrue(state.isAlive(), "the self-close must not have taken effect");
        }
    }
}
