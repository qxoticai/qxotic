package com.qxotic.jinfer.x.llm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.x.boundary.BaseState;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.boundary.Config;
import com.qxotic.jinfer.x.boundary.LanguageModel;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.time.Duration;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.OptionalInt;
import java.util.Queue;
import java.util.Set;
import org.junit.jupiter.api.Test;

class GeneratorTest {

    private static final Config CONFIG =
            new Config() {
                @Override
                public int vocabularySize() {
                    return 32_000;
                }

                @Override
                public int contextLength() {
                    return 1 << 20;
                }
            };

    /** A scripted sampler: yields queued token ids regardless of logits. */
    private static Sampler scripted(int... tokens) {
        Queue<Integer> queue = new ArrayDeque<>();
        for (int t : tokens) queue.add(t);
        return logits -> {
            Integer t = queue.poll();
            if (t == null) throw new AssertionError("script exhausted");
            return t;
        };
    }

    /** A minimal model: prefill ingests the prompt, decode rows are scripted by the sampler. */
    private static class FakeModel implements LanguageModel<Config, Object, FakeModel.State> {

        final List<int[]> ingested = new ArrayList<>();

        static final class State extends BaseState {
            private final int contextCapacity;
            private final int batchCapacity;

            State(int contextCapacity, int batchCapacity, Arena arena) {
                super(arena);
                this.contextCapacity = contextCapacity;
                this.batchCapacity = batchCapacity;
            }

            @Override
            public int contextCapacity() {
                return contextCapacity;
            }

            @Override
            public int batchCapacity() {
                return batchCapacity;
            }

            @Override
            public void reset() {
                resumeAt(0);
            }
        }

        @Override
        public Config config() {
            return CONFIG;
        }

        @Override
        public Object weights() {
            return new Object();
        }

        @Override
        public State newState(int contextCapacity, int batchCapacity, Arena arena) {
            return new State(contextCapacity, batchCapacity, arena);
        }

        /** Test hook: runs at the top of every forward call. */
        void onForward() {}

        @Override
        public void forward(State state, Batch batch) {
            onForward();
            if (batch.input() instanceof Batch.Input.Tokens t) ingested.add(t.ids());
            state.advance(batch.count(), batch.outputs());
        }

        @Override
        public MemoryView<?> head(State state, int output) {
            return TestLogits.view(CONFIG.vocabularySize());
        }
    }

    private static Generator.Constraints constraints(
            int maxTokens, Duration timeout, Set<Integer> stops) {
        return new Generator.Constraints(maxTokens, timeout, stops);
    }

    private static Generator.GenerationListener recording(
            List<Integer> seen, List<Integer> committed) {
        return new Generator.GenerationListener() {
            @Override
            public boolean onToken(int token) {
                seen.add(token);
                return true;
            }

            @Override
            public void onIngested(int token) {
                committed.add(token);
            }
        };
    }

    @Test
    void generatesUntilAStopToken() {
        FakeModel model = new FakeModel();
        try (FakeModel.State state = model.newState(128, 8, Arena.ofAuto())) {
            List<Integer> seen = new ArrayList<>(), committed = new ArrayList<>();
            var result =
                    Generator.generate(
                            model,
                            state,
                            new int[] {1, 2, 3},
                            scripted(10, 11, 99),
                            constraints(16, Duration.ZERO, Set.of(99)),
                            recording(seen, committed));

            assertEquals(Generator.FinishReason.STOP, result.finishReason());
            assertArrayEquals(new int[] {10, 11}, result.tokens()); // stop token excluded
            assertEquals(OptionalInt.of(99), result.stopToken());
            assertEquals(List.of(10, 11, 99), seen); // listener sees the stop token
            assertEquals(List.of(10, 11), committed); // the stop token is never ingested
            assertEquals(3 + 2, state.position()); // prompt + the two committed decodes
            assertEquals(2, result.completionTokens());
        }
    }

    @Test
    void unlimitedBudgetEndsAtContextExhaustion() {
        FakeModel model = new FakeModel();
        try (FakeModel.State state = model.newState(4, 8, Arena.ofAuto())) {
            var result =
                    Generator.generate(
                            model,
                            state,
                            new int[] {1, 2},
                            scripted(10, 11, 12),
                            constraints(Generator.Constraints.UNLIMITED, Duration.ZERO, Set.of()),
                            recording(new ArrayList<>(), new ArrayList<>()));

            assertEquals(Generator.FinishReason.LENGTH, result.finishReason());
            assertArrayEquals(new int[] {10, 11}, result.tokens()); // 4 - 2 prompt positions
            assertEquals(OptionalInt.empty(), result.stopToken());
        }
    }

    @Test
    void zeroBudgetIsPrefillOnly() {
        FakeModel model = new FakeModel();
        try (FakeModel.State state = model.newState(128, 8, Arena.ofAuto())) {
            var result =
                    Generator.generate(
                            model,
                            state,
                            new int[] {1, 2, 3},
                            scripted(),
                            constraints(0, Duration.ZERO, Set.of()),
                            recording(new ArrayList<>(), new ArrayList<>()));

            assertEquals(Generator.FinishReason.LENGTH, result.finishReason());
            assertArrayEquals(new int[0], result.tokens());
            assertEquals(3, state.position()); // the prompt was still ingested
        }
    }

    @Test
    void abortRecordsButDoesNotIngestTheAbortingToken() {
        FakeModel model = new FakeModel();
        try (FakeModel.State state = model.newState(128, 8, Arena.ofAuto())) {
            List<Integer> committed = new ArrayList<>();
            var result =
                    Generator.generate(
                            model,
                            state,
                            new int[] {1},
                            scripted(10, 11, 12),
                            constraints(16, Duration.ZERO, Set.of()),
                            new Generator.GenerationListener() {
                                private int count;

                                @Override
                                public boolean onToken(int token) {
                                    return ++count < 2; // abort on the second token
                                }

                                @Override
                                public void onIngested(int token) {
                                    committed.add(token);
                                }
                            });

            assertEquals(Generator.FinishReason.ABORT, result.finishReason());
            assertArrayEquals(new int[] {10, 11}, result.tokens()); // aborting token recorded
            assertEquals(List.of(10), committed); // ... but not ingested
            assertEquals(1 + 1, state.position());
        }
    }

    @Test
    void rejectsAnOutOfRangeSampleAndAnOversizedPrompt() {
        FakeModel model = new FakeModel();
        try (FakeModel.State state = model.newState(4, 8, Arena.ofAuto())) {
            assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            Generator.generate(
                                    model,
                                    state,
                                    new int[] {1},
                                    scripted(32_000),
                                    constraints(4, Duration.ZERO, Set.of()),
                                    recording(new ArrayList<>(), new ArrayList<>())));
            assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            Generator.generate(
                                    model,
                                    state,
                                    new int[] {1, 2, 3, 4, 5},
                                    scripted(),
                                    constraints(4, Duration.ZERO, Set.of()),
                                    recording(new ArrayList<>(), new ArrayList<>())));
        }
    }

    @Test
    void deadlinesEndThePassAsTimeoutNotLength() {
        FakeModel model = new FakeModel();
        try (FakeModel.State state = model.newState(128, 8, Arena.ofAuto())) {
            var result =
                    Generator.generate(
                            model,
                            state,
                            new int[] {1},
                            scripted(10, 11, 12, 13),
                            constraints(16, Duration.ofNanos(1), Set.of()),
                            recording(new ArrayList<>(), new ArrayList<>()));

            assertEquals(Generator.FinishReason.TIMEOUT, result.finishReason());
            assertTrue(result.completionTokens() < 4); // the deadline cut the script short
        }
    }

    @Test
    void expiredDeadlineEmitsNoToken() {
        FakeModel model = new FakeModel();
        try (FakeModel.State state = model.newState(128, 8, Arena.ofAuto())) {
            List<Integer> seen = new ArrayList<>(), committed = new ArrayList<>();
            var result =
                    Generator.generate(
                            model,
                            state,
                            new int[] {1},
                            scripted(10, 11, 12),
                            constraints(16, Duration.ofNanos(1), Set.of()),
                            recording(seen, committed));

            assertEquals(Generator.FinishReason.TIMEOUT, result.finishReason());
            assertArrayEquals(new int[0], result.tokens());
            assertEquals(List.of(), seen); // no token is sampled past the deadline
            assertEquals(List.of(), committed);
        }
    }

    @Test
    void deadlineBreaksBetweenPrefillChunks() {
        FakeModel model =
                new FakeModel() {
                    @Override
                    void onForward() {
                        try {
                            Thread.sleep(100); // one chunk outlasts the whole budget
                        } catch (InterruptedException e) {
                            throw new AssertionError(e);
                        }
                    }
                };
        try (FakeModel.State state = model.newState(128, 4, Arena.ofAuto())) {
            List<Integer> seen = new ArrayList<>();
            var result =
                    Generator.generate(
                            model,
                            state,
                            new int[32], // 8 chunks of 4
                            scripted(10, 11, 12),
                            constraints(16, Duration.ofMillis(50), Set.of()),
                            recording(seen, new ArrayList<>()));

            assertEquals(Generator.FinishReason.TIMEOUT, result.finishReason());
            assertArrayEquals(new int[0], result.tokens());
            assertEquals(List.of(), seen);
            assertEquals(4, state.position()); // chunk 0 completed, chunk 1 never started
        }
    }

    @Test
    void generousDeadlineIsTransparent() {
        FakeModel model = new FakeModel();
        try (FakeModel.State state = model.newState(128, 8, Arena.ofAuto())) {
            List<Integer> seen = new ArrayList<>(), committed = new ArrayList<>();
            var result =
                    Generator.generate(
                            model,
                            state,
                            new int[] {1},
                            scripted(10, 11, 12),
                            constraints(3, Duration.ofHours(1), Set.of()),
                            recording(seen, committed));

            assertEquals(Generator.FinishReason.LENGTH, result.finishReason());
            assertArrayEquals(new int[] {10, 11, 12}, result.tokens());
            assertEquals(List.of(10, 11, 12), seen);
            assertEquals(List.of(10, 11), committed); // the final token is not ingested
        }
    }
}
