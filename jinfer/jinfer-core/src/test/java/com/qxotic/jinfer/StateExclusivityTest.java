package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.util.ConcurrentModificationException;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;

/**
 * ONE state, ONE user - across every entry point, in both directions.
 *
 * <p>The contract is a property of the STATE, not of a method: while any entry point holds a state,
 * every other entry point must refuse it too. So this parks a thread inside each entry point in
 * turn and then attacks the state from another thread through every entry point, asserting a
 * {@link ConcurrentModificationException} each time. A gap here is not a race that shows up in a
 * stress test - it is two threads writing the same scratch buffers, which produces wrong numbers
 * silently.
 *
 * <p>The speech side is pinned the same way by {@code SpeechStateLifecycleTest} in jinfer-inflect2,
 * where the state type lives.
 */
final class StateExclusivityTest {

    /** Every public entry point that runs kernels against a state. */
    private enum Entry {
        INGEST {
            @Override
            void invoke(ProbeModel m, ProbeState s) {
                m.ingest(s, Batch.prefill(new int[] {1, 2}));
            }
        },
        LOGITS {
            @Override
            void invoke(ProbeModel m, ProbeState s) {
                m.logits(s, 0);
            }
        },
        EMBEDDING {
            @Override
            void invoke(ProbeModel m, ProbeState s) {
                m.embedding(s, 0);
            }
        },
        EMBED {
            @Override
            void invoke(ProbeModel m, ProbeState s) {
                m.embed(
                        s,
                        new Batch.Input.Sequences(new Batch.Input.Tokens(new int[] {1, 2}), new int[] {2}),
                        tensor -> {});
            }
        };

        abstract void invoke(ProbeModel m, ProbeState s);
    }

    @Test
    void whileAnyEntryPointHoldsTheStateEveryOtherOneIsRefused() throws Exception {
        for (Entry held : Entry.values()) {
            for (Entry attacker : Entry.values()) {
                ProbeModel model = new ProbeModel();
                try (ProbeState state = new ProbeState(Arena.ofShared())) {
                    CountDownLatch inside = new CountDownLatch(1);
                    CountDownLatch release = new CountDownLatch(1);
                    model.park(inside, release);

                    Thread holder = new Thread(() -> held.invoke(model, state));
                    holder.start();
                    assertTrue(
                            inside.await(10, TimeUnit.SECONDS),
                            held + " never entered the kernel");

                    AtomicReference<Throwable> refused = new AtomicReference<>();
                    Thread other =
                            new Thread(
                                    () -> {
                                        try {
                                            attacker.invoke(model, state);
                                        } catch (Throwable t) {
                                            refused.set(t);
                                        }
                                    });
                    other.start();
                    other.join(10_000);

                    assertInstanceOf(
                            ConcurrentModificationException.class,
                            refused.get(),
                            held + " held the state but " + attacker + " was allowed in");
                    assertTrue(
                            refused.get().getMessage().contains("single serial pipeline"),
                            "the message must name the contract: " + refused.get().getMessage());

                    release.countDown();
                    holder.join(10_000);
                }
            }
        }
    }

    @Test
    void embedHoldsTheStateBetweenChunksNotJustInsideThem() throws Exception {
        // The dangerous window for a multi-step operation is BETWEEN its steps. embed() ingests
        // chunk by chunk, and each ingest claims and releases; if that were the only claim, a
        // second caller could interleave its chunks into the same KV context and corrupt both.
        // The sink runs in exactly that window, so parking there is what proves the OUTER claim
        // exists - parking inside a kernel would be covered by ingest's own claim and prove
        // nothing.
        ProbeModel model = new ProbeModel();
        try (ProbeState state = new ProbeState(Arena.ofShared())) {
            CountDownLatch inSink = new CountDownLatch(1);
            CountDownLatch release = new CountDownLatch(1);
            AtomicReference<Throwable> refused = new AtomicReference<>();

            Thread holder =
                    new Thread(
                            () ->
                                    model.embed(
                                            state,
                                            new Batch.Input.Sequences(
                                                    new Batch.Input.Tokens(new int[] {1, 2}),
                                                    new int[] {2}),
                                            tensor -> {
                                                inSink.countDown();
                                                try {
                                                    release.await(30, TimeUnit.SECONDS);
                                                } catch (InterruptedException e) {
                                                    Thread.currentThread().interrupt();
                                                }
                                            }));
            holder.start();
            assertTrue(inSink.await(10, TimeUnit.SECONDS), "embed never reached its sink");

            Thread other =
                    new Thread(
                            () -> {
                                try {
                                    model.ingest(state, Batch.prefill(new int[] {3, 4}));
                                } catch (Throwable e) {
                                    refused.set(e);
                                }
                            });
            other.start();
            other.join(10_000);

            assertInstanceOf(
                    ConcurrentModificationException.class,
                    refused.get(),
                    "embed released the state between chunks - a second caller interleaved");

            release.countDown();
            holder.join(10_000);
        }
    }

    @Test
    void theSameThreadMayReenter() {
        // a generation holds the claim across many forwards, and embed() holds it across chunks -
        // exclusion is between THREADS, never against the holder itself
        ProbeModel model = new ProbeModel();
        try (ProbeState state = new ProbeState(Arena.ofShared())) {
            model.reentrant = true;
            model.ingest(state, Batch.prefill(new int[] {1, 2}));
            assertTrue(model.reenteredOk, "the holding thread must be allowed back in");
        }
    }

    @Test
    void twoStatesOfOneModelDoNotInterfere() {
        // the claim is per STATE: parallel pipelines are separate states, which is exactly what
        // the exception message tells you to build
        ProbeModel model = new ProbeModel();
        try (ProbeState a = new ProbeState(Arena.ofShared());
                ProbeState b = new ProbeState(Arena.ofShared())) {
            model.ingest(a, Batch.prefill(new int[] {1, 2}));
            model.ingest(b, Batch.prefill(new int[] {1, 2}));
            assertEquals(2, model.forwards);
        }
    }

    // ── doubles ───────────────────────────────────────────────────────────

    static final class ProbeState extends BaseState {
        ProbeState(Arena arena) {
            super(arena);
            adoptArena();
        }

        @Override
        public int contextCapacity() {
            return 64;
        }

        @Override
        public int batchCapacity() {
            return 64;
        }

        @Override
        public void reset() {
            resumeAt(0);
        }
    }

    /** Implements both heads, so one state can be attacked through every entry point. */
    static final class ProbeModel
            implements LanguageModel<Config, Void, ProbeState>,
                    EmbeddingModel<Config, Void, ProbeState> {

        private CountDownLatch inside, release;
        boolean reentrant, reenteredOk;
        int forwards;

        void park(CountDownLatch inside, CountDownLatch release) {
            this.inside = inside;
            this.release = release;
        }

        private void body(ProbeState state) {
            forwards++;
            if (reentrant) { // the holding thread comes back through another entry point
                logits(state, 0);
                reenteredOk = true;
                return;
            }
            if (inside == null) return;
            inside.countDown();
            try {
                release.await(30, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
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
        public ProbeState newState(int contextCapacity, int batchCapacity, Arena arena) {
            return new ProbeState(arena);
        }

        @Override
        public void forward(ProbeState state, Batch batch) {
            body(state);
            state.advance(2, Batch.Outputs.LAST);
        }

        @Override
        public FloatTensor head(ProbeState state, int output) {
            if (!reentrant) body(state);
            return F32FloatTensor.allocate(state.arena, 4);
        }

        @Override
        public FloatTensor pool(ProbeState state, int index) {
            if (!reentrant) body(state);
            return F32FloatTensor.allocate(state.arena, 4);
        }
    }
}
