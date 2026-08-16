package com.qxotic.jinfer.boundary;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.PanamaMemoryArena;
import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.ConcurrentModificationException;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

/** Every state-consuming model entry point observes one shared exclusion boundary. */
final class StateExclusivityTest {

    private enum Entry {
        INGEST {
            @Override
            void invoke(ProbeModel model, ProbeState state) {
                model.ingest(state, Batch.prefill(new int[] {1, 2}));
            }
        },
        LOGITS {
            @Override
            void invoke(ProbeModel model, ProbeState state) {
                model.logits(state, 0);
            }
        },
        LOGITS_LAST {
            @Override
            void invoke(ProbeModel model, ProbeState state) {
                model.logits(state);
            }
        },
        PROJECT_EMBEDDING {
            @Override
            void invoke(ProbeModel model, ProbeState state) {
                model.projectEmbedding(state, 0, ignored -> {});
            }
        },
        PROJECT_FINAL_EMBEDDING {
            @Override
            void invoke(ProbeModel model, ProbeState state) {
                model.projectEmbedding(state, ignored -> {});
            }
        },
        EMBED_ALL {
            @Override
            void invoke(ProbeModel model, ProbeState state) {
                model.embedAll(
                        state,
                        new Batch.Input.Sequences(
                                new Batch.Input.Tokens(new int[] {1, 2}), new int[] {2}),
                        ignored -> {});
            }
        };

        abstract void invoke(ProbeModel model, ProbeState state);
    }

    @Test
    void eachEntryPointExcludesEveryOtherEntryPoint() throws Exception {
        for (Entry held : Entry.values()) {
            for (Entry attacker : Entry.values()) {
                ProbeModel model = new ProbeModel();
                try (ProbeState state =
                        new ProbeState(new PanamaMemoryArena(Arena.ofShared()), true)) {
                    model.ingest(state, Batch.step(0));
                    CountDownLatch inside = new CountDownLatch(1);
                    CountDownLatch release = new CountDownLatch(1);
                    model.park(inside, release);

                    Thread holder = new Thread(() -> held.invoke(model, state));
                    holder.start();
                    assertTrue(inside.await(10, TimeUnit.SECONDS));

                    AtomicReference<Throwable> failure = new AtomicReference<>();
                    Thread contender =
                            new Thread(
                                    () -> {
                                        try {
                                            attacker.invoke(model, state);
                                        } catch (Throwable e) {
                                            failure.set(e);
                                        }
                                    });
                    contender.start();
                    contender.join(10_000);

                    assertInstanceOf(ConcurrentModificationException.class, failure.get());
                    release.countDown();
                    holder.join(10_000);
                }
            }
        }
    }

    @Test
    void embedAllKeepsExclusiveAccessWhileCallingTheSink() throws Exception {
        ProbeModel model = new ProbeModel();
        try (ProbeState state = new ProbeState(new PanamaMemoryArena(Arena.ofShared()), true)) {
            CountDownLatch inSink = new CountDownLatch(1);
            CountDownLatch release = new CountDownLatch(1);
            Thread holder =
                    new Thread(
                            () ->
                                    model.embedAll(
                                            state,
                                            new Batch.Input.Sequences(
                                                    new Batch.Input.Tokens(new int[] {1, 2}),
                                                    new int[] {2}),
                                            ignored -> {
                                                inSink.countDown();
                                                try {
                                                    release.await();
                                                } catch (InterruptedException e) {
                                                    Thread.currentThread().interrupt();
                                                }
                                            }));
            holder.start();
            assertTrue(inSink.await(10, TimeUnit.SECONDS));

            AtomicReference<Throwable> failure = new AtomicReference<>();
            Thread contender =
                    new Thread(
                            () -> {
                                try {
                                    model.ingest(state, Batch.step(3));
                                } catch (Throwable e) {
                                    failure.set(e);
                                }
                            });
            contender.start();
            contender.join(10_000);
            assertInstanceOf(ConcurrentModificationException.class, failure.get());

            release.countDown();
            holder.join(10_000);
        }
    }

    @Test
    void projectEmbeddingScopesTheViewToTheSynchronousExclusiveCallback() throws Exception {
        ProbeModel model = new ProbeModel();
        ProbeState state = new ProbeState(new PanamaMemoryArena(Arena.ofShared()), true);
        model.ingest(state, Batch.step(0));
        CountDownLatch inConsumer = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        AtomicReference<Thread> callbackThread = new AtomicReference<>();
        Thread caller =
                new Thread(
                        () ->
                                model.projectEmbedding(
                                        state,
                                        0,
                                        ignored -> {
                                            callbackThread.set(Thread.currentThread());
                                            inConsumer.countDown();
                                            try {
                                                release.await();
                                            } catch (InterruptedException e) {
                                                Thread.currentThread().interrupt();
                                            }
                                        }));
        caller.start();
        assertTrue(inConsumer.await(10, TimeUnit.SECONDS));
        assertSame(caller, callbackThread.get());

        AtomicReference<Throwable> contenderFailure = new AtomicReference<>();
        Thread contender =
                new Thread(
                        () -> {
                            try {
                                model.ingest(state, Batch.step(1));
                            } catch (Throwable e) {
                                contenderFailure.set(e);
                            }
                        });
        contender.start();
        contender.join(10_000);
        assertInstanceOf(ConcurrentModificationException.class, contenderFailure.get());

        CountDownLatch closed = new CountDownLatch(1);
        Thread closer =
                new Thread(
                        () -> {
                            state.close();
                            closed.countDown();
                        });
        closer.start();
        try {
            assertFalse(closed.await(100, TimeUnit.MILLISECONDS));
        } finally {
            release.countDown();
        }
        caller.join(10_000);
        closer.join(10_000);
        assertEquals(0, closed.getCount());
    }

    @Test
    void projectEmbeddingPropagatesConsumerFailureAndReleasesTheState() {
        ProbeModel model = new ProbeModel();
        try (ProbeState state = new ProbeState(new PanamaMemoryArena(Arena.ofShared()), true)) {
            model.ingest(state, Batch.step(0));
            RuntimeException expected = new RuntimeException("consumer failed");
            RuntimeException actual =
                    assertThrows(
                            RuntimeException.class,
                            () ->
                                    model.projectEmbedding(
                                            state,
                                            0,
                                            ignored -> {
                                                throw expected;
                                            }));
            assertSame(expected, actual);

            model.ingest(state, Batch.step(1));
            assertEquals(2, state.position());
        }
    }

    @Test
    void safeEntryPointsMayNestOnTheHoldingThread() {
        ProbeModel model = new ProbeModel();
        try (ProbeState state = new ProbeState(new PanamaMemoryArena(Arena.ofShared()), true)) {
            state.exclusively(() -> model.ingest(state, Batch.prefill(new int[] {1, 2})));
            assertEquals(2, state.position());
        }
    }

    @Test
    void projectEmbeddingWithoutIndexProjectsTheFinalRetainedOutput() {
        ProbeModel model = new ProbeModel();
        try (ProbeState state = new ProbeState(new PanamaMemoryArena(Arena.ofShared()), true)) {
            model.ingest(state, Batch.score(new int[] {1, 2, 3}));
            model.projectEmbedding(state, ignored -> {});
            assertEquals(List.of(2), model.projectedOutputs);
        }
    }

    @Test
    void embedAllProjectsSequenceEndsAcrossBatchBoundaries() {
        ProbeModel model = new ProbeModel();
        try (ProbeState state = model.newState(64, 3)) {
            model.embedAll(
                    state,
                    new Batch.Input.Sequences(
                            new Batch.Input.Tokens(new int[] {1, 2, 3, 4, 5, 6}),
                            new int[] {2, 3, 1}),
                    ignored -> {});

            assertEquals(List.of(1, 1, 2), model.projectedOutputs);
            assertEquals(6, state.position());
        }
    }

    @Test
    void embedAllRejectsIncompleteOrEmptySequences() {
        ProbeModel model = new ProbeModel();
        try (ProbeState state = model.newState(64, 3)) {
            assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            model.embedAll(
                                    state,
                                    new Batch.Input.Sequences(
                                            new Batch.Input.Tokens(new int[] {1, 2, 3}),
                                            new int[] {2}),
                                    ignored -> {}));
            assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            model.embedAll(
                                    state,
                                    new Batch.Input.Sequences(
                                            new Batch.Input.Tokens(new int[0]), new int[] {0}),
                                    ignored -> {}));
        }
    }

    @Test
    void projectEmbeddingWithoutRetainedOutputFailsClearly() {
        ProbeModel model = new ProbeModel();
        try (ProbeState state = new ProbeState(new PanamaMemoryArena(Arena.ofShared()), true)) {
            IllegalStateException error =
                    assertThrows(
                            IllegalStateException.class,
                            () -> model.projectEmbedding(state, ignored -> {}));
            assertEquals("state has no retained outputs", error.getMessage());
        }
    }

    record Configuration(int vocabularySize, int contextLength) implements ContextConfiguration {}

    static final class ProbeState extends ContextState {
        ProbeState(MemoryArena<MemorySegment> arena, boolean ownsArena) {
            this(64, 64, arena, ownsArena);
        }

        ProbeState(
                int contextCapacity,
                int batchCapacity,
                MemoryArena<MemorySegment> arena,
                boolean ownsArena) {
            super(contextCapacity, batchCapacity, arena, ownsArena);
        }

        MemoryView<MemorySegment> floats(int size) {
            return Views.allocateF32(memoryArena(), size);
        }

        void advance(Batch batch) {
            advanceContext(batch.count(), batch.outputs());
        }

        @Override
        protected void clearHistory() {}
    }

    static final class ProbeModel
            implements LanguageModel<Configuration, Void, ProbeState>,
                    EmbeddingModel<Configuration, Void, ProbeState> {
        private CountDownLatch inside;
        private CountDownLatch release;
        private final List<Integer> projectedOutputs = new ArrayList<>();

        void park(CountDownLatch inside, CountDownLatch release) {
            this.inside = inside;
            this.release = release;
        }

        private void body() {
            if (inside == null) return;
            inside.countDown();
            try {
                release.await();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        @Override
        public Configuration configuration() {
            return new Configuration(8, 64);
        }

        @Override
        public Void weights() {
            return null;
        }

        @Override
        public ProbeState newState(int contextCapacity, int batchCapacity) {
            return new ProbeState(
                    contextCapacity, batchCapacity, Arenas.newCrossThreadMemoryArena(), true);
        }

        @Override
        public ProbeState newState(
                int contextCapacity, int batchCapacity, MemoryArena<MemorySegment> arena) {
            return new ProbeState(contextCapacity, batchCapacity, arena, false);
        }

        @Override
        public void ingest(ProbeState state, Batch batch) {
            state.exclusively(
                    () -> {
                        body();
                        state.advance(batch);
                    });
        }

        @Override
        public MemoryView<?> logits(ProbeState state, int output) {
            return state.exclusively(
                    () -> {
                        body();
                        return state.floats(4);
                    });
        }

        @Override
        public void projectEmbedding(
                ProbeState state, int output, java.util.function.Consumer<MemoryView<?>> consumer) {
            state.exclusively(
                    () -> {
                        body();
                        projectedOutputs.add(output);
                        consumer.accept(state.floats(4));
                    });
        }
    }
}
