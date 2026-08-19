package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.ContextConfiguration;
import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.EmbeddingModel;
import com.qxotic.jinfer.Views;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;

final class LoadedEmbedderTest {

    @Test
    void snapshotsFramingAndPacksAgainstTheStateCapacity() {
        int[] prefix = {10};
        int[] suffix = {20};
        ProbeModel model = new ProbeModel(4);
        LoadedEmbedder<ProbeState> loaded =
                loaded(model, 2, IntSequence.wrap(prefix), IntSequence.wrap(suffix));
        prefix[0] = 99;
        suffix[0] = 99;

        try (ProbeState state = model.newState(5, 5)) {
            List<float[]> vectors = new ArrayList<>();
            int tokens = loaded.embedAll(state, List.of("ab", "c"), vectors::add);

            assertEquals(7, tokens);
            assertEquals(2, model.batches.size(), "the two framed sequences do not fit together");
            assertArrayEquals(new int[] {10, 'a', 'b', 20}, model.batches.get(0));
            assertArrayEquals(new int[] {10, 'c', 20}, model.batches.get(1));
            assertArrayEquals(new int[] {4}, model.lengths.get(0));
            assertArrayEquals(new int[] {3}, model.lengths.get(1));
            assertEquals(2, vectors.size());
            assertEquals(1, vectors.get(0)[0]);
            assertEquals(2, vectors.get(1)[0]);
            assertNotSame(vectors.get(0), vectors.get(1));
        }
    }

    @Test
    void dimensionsAreValuesNotOptionProvenance() {
        ProbeModel model = new ProbeModel(4).outputs(new float[] {3, 4, 12, 7});
        LoadedEmbedder<ProbeState> variable = loaded(model, 2);

        try (ProbeState state = model.newState(8, 8)) {
            List<float[]> nativeOutput = new ArrayList<>();
            variable.embedAll(state, List.of("x"), nativeOutput::add);
            assertArrayEquals(new float[] {3, 4, 12, 7}, nativeOutput.get(0));

            List<float[]> reduced = new ArrayList<>();
            variable.embedAll(state, List.of("x"), 2, reduced::add);
            assertArrayEquals(new float[] {0.6f, 0.8f}, reduced.get(0), 1e-6f);

            List<float[]> explicitNative = new ArrayList<>();
            variable.embedAll(state, List.of("x"), 4, explicitNative::add);
            assertArrayEquals(new float[] {3, 4, 12, 7}, explicitNative.get(0));

            assertThrows(
                    IllegalArgumentException.class,
                    () -> variable.embedAll(state, List.of("x"), 1, ignored -> {}));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> variable.embedAll(state, List.of("x"), 5, ignored -> {}));
        }

        ProbeModel fixedModel = new ProbeModel(4);
        LoadedEmbedder<ProbeState> fixed = loaded(fixedModel, 4);
        assertFalse(fixed.supportsCustomDimensions());
        try (ProbeState state = fixedModel.newState(8, 8)) {
            fixed.embedAll(state, List.of("x"), 4, ignored -> {});
            fixed.embedAll(state, List.of("x"), ignored -> {});
            IllegalArgumentException failure =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> fixed.embedAll(state, List.of("x"), 2, ignored -> {}));
            assertTrue(failure.getMessage().contains("fixed embedding dimension"));
        }
    }

    @Test
    void copiedResultsSurviveScratchReuseAndStateClose() {
        ProbeModel model =
                new ProbeModel(4).outputs(new float[] {1, 2, 3, 4}, new float[] {5, 6, 7, 8});
        LoadedEmbedder<ProbeState> loaded = loaded(model, 4);
        List<float[]> vectors = new ArrayList<>();

        ProbeState state = model.newState(8, 8);
        loaded.embedAll(state, List.of("a", "b"), vectors::add);
        state.close();

        assertArrayEquals(new float[] {1, 2, 3, 4}, vectors.get(0));
        assertArrayEquals(new float[] {5, 6, 7, 8}, vectors.get(1));
        assertNotSame(vectors.get(0), vectors.get(1));
    }

    @Test
    void zeroPrefixStaysFiniteAndInvalidViewsFailAtTheOwnedBoundary() {
        ProbeModel zeroModel = new ProbeModel(4).outputs(new float[] {0, 0, 1, 0});
        LoadedEmbedder<ProbeState> zero = loaded(zeroModel, 2);
        try (ProbeState state = zeroModel.newState(8, 8)) {
            List<float[]> vectors = new ArrayList<>();
            zero.embedAll(state, List.of("x"), 2, vectors::add);
            assertArrayEquals(new float[] {0, 0}, vectors.get(0));
        }

        ProbeModel invalidModel = new ProbeModel(4);
        invalidModel.projected =
                Views.wrap(MemorySegment.ofArray(new float[3]), DataType.FP32, Shape.flat(3));
        LoadedEmbedder<ProbeState> invalid = loaded(invalidModel, 4);
        try (ProbeState state = invalidModel.newState(8, 8)) {
            assertThrows(
                    IllegalArgumentException.class,
                    () -> invalid.embedAll(state, List.of("x"), ignored -> {}));
        }

        Arena arena = Arena.ofShared();
        ProbeModel closedModel = new ProbeModel(4);
        closedModel.projected =
                Views.wrap(arena.allocate(4L * Float.BYTES), DataType.FP32, Shape.flat(4));
        arena.close();
        LoadedEmbedder<ProbeState> closed = loaded(closedModel, 4);
        try (ProbeState state = closedModel.newState(8, 8)) {
            assertThrows(
                    IllegalStateException.class,
                    () -> closed.embedAll(state, List.of("x"), ignored -> {}));
        }
    }

    @Test
    void invalidInputsFailBeforeInferenceAndEmptyInputIsARealNoop() {
        ProbeModel model = new ProbeModel(4);
        LoadedEmbedder<ProbeState> loaded = loaded(model, 4);
        try (ProbeState state = model.newState(3, 3)) {
            assertEquals(0, loaded.embedAll(state, List.of(), ignored -> {}));
            assertEquals(0, model.calls);

            assertThrows(
                    IllegalArgumentException.class,
                    () -> loaded.embedAll(state, List.of("abcd"), ignored -> {}));
            assertThrows(
                    NullPointerException.class,
                    () -> loaded.embedAll(state, Arrays.asList("a", null), ignored -> {}));
            assertEquals(0, model.calls);
        }

        LoadedEmbedder<ProbeState> unframed =
                loaded(model, 4, IntSequence.empty(), IntSequence.empty());
        try (ProbeState state = model.newState(3, 3)) {
            assertThrows(
                    IllegalArgumentException.class,
                    () -> unframed.embedAll(state, List.of(""), ignored -> {}));
            assertEquals(0, model.calls);
        }
    }

    @Test
    void consumerRunsAfterExclusiveStateAccessAndFailuresDoNotPoisonTheState() {
        ProbeModel model = new ProbeModel(4);
        LoadedEmbedder<ProbeState> loaded = loaded(model, 4);
        try (ProbeState state = model.newState(8, 8)) {
            AtomicReference<Throwable> concurrentFailure = new AtomicReference<>();
            loaded.embedAll(
                    state,
                    List.of("x"),
                    vector -> {
                        Thread thread =
                                Thread.ofPlatform()
                                        .unstarted(
                                                () -> {
                                                    try {
                                                        state.exclusively(() -> {});
                                                    } catch (Throwable failure) {
                                                        concurrentFailure.set(failure);
                                                    }
                                                });
                        thread.start();
                        try {
                            thread.join();
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            throw new RuntimeException(e);
                        }
                    });
            assertNull(concurrentFailure.get(), "consumer ran while the model held the state");

            RuntimeException expected = new RuntimeException("consumer failed");
            RuntimeException actual =
                    assertThrows(
                            RuntimeException.class,
                            () ->
                                    loaded.embedAll(
                                            state,
                                            List.of("x"),
                                            ignored -> {
                                                throw expected;
                                            }));
            assertSame(expected, actual);
            loaded.embedAll(state, List.of("x"), ignored -> {});
        }
    }

    @Test
    void constructorRejectsInvalidMetadata() {
        ProbeModel model = new ProbeModel(4);
        assertThrows(IllegalArgumentException.class, () -> loaded(model, 0));
        assertThrows(IllegalArgumentException.class, () -> loaded(model, 5));
        assertThrows(NullPointerException.class, () -> loaded(model, 4, null, IntSequence.empty()));
    }

    private static LoadedEmbedder<ProbeState> loaded(ProbeModel model, int minimumDimension) {
        return loaded(model, minimumDimension, IntSequence.of(1), IntSequence.empty());
    }

    private static LoadedEmbedder<ProbeState> loaded(
            ProbeModel model,
            int minimumDimension,
            IntSequence prefixTokens,
            IntSequence suffixTokens) {
        return new LoadedEmbedder<>(
                model,
                tokenizer(),
                prefixTokens,
                suffixTokens,
                model.dimension,
                minimumDimension,
                "test-embedder",
                "",
                "");
    }

    private static Tokenizer tokenizer() {
        return (Tokenizer)
                Proxy.newProxyInstance(
                        Tokenizer.class.getClassLoader(),
                        new Class<?>[] {Tokenizer.class},
                        (proxy, method, args) -> {
                            if (method.getName().equals("encode")) {
                                return IntSequence.wrap(args[0].toString().chars().toArray());
                            }
                            throw new UnsupportedOperationException(method.getName());
                        });
    }

    private record Configuration(int vocabularySize, int contextLength)
            implements ContextConfiguration {}

    private static final class ProbeState extends ContextState {
        private ProbeState(
                int contextCapacity,
                int batchCapacity,
                MemoryArena<MemorySegment> arena,
                boolean ownsArena) {
            super(contextCapacity, batchCapacity, arena, ownsArena);
        }

        @Override
        protected void clearHistory() {}
    }

    private static final class ProbeModel
            implements EmbeddingModel<Configuration, Void, ProbeState> {
        private final int dimension;
        private final float[] scratch;
        private final MemoryView<MemorySegment> scratchView;
        private final List<int[]> batches = new ArrayList<>();
        private final List<int[]> lengths = new ArrayList<>();
        private List<float[]> outputs = List.of();
        private MemoryView<?> projected;
        private int emitted;
        private int calls;

        private ProbeModel(int dimension) {
            this.dimension = dimension;
            scratch = new float[dimension];
            scratchView =
                    Views.wrap(
                            MemorySegment.ofArray(scratch), DataType.FP32, Shape.flat(dimension));
        }

        private ProbeModel outputs(float[]... outputs) {
            this.outputs = List.of(outputs);
            return this;
        }

        @Override
        public Configuration configuration() {
            return new Configuration(256, 1024);
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
            throw new UnsupportedOperationException();
        }

        @Override
        public void projectEmbedding(
                ProbeState state, int outputIndex, Consumer<MemoryView<?>> consumer) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void embedAll(
                ProbeState state,
                Batch.Input.Sequences sequences,
                Consumer<MemoryView<?>> consumer) {
            state.exclusively(
                    () -> {
                        calls++;
                        batches.add(sequences.tokens().ids().clone());
                        lengths.add(sequences.seqLen().clone());
                        for (int ignored : sequences.seqLen()) {
                            if (projected != null) {
                                consumer.accept(projected);
                                continue;
                            }
                            float[] output;
                            if (outputs.isEmpty()) {
                                output = new float[dimension];
                                output[0] = ++emitted;
                            } else {
                                output = outputs.get(emitted++ % outputs.size());
                            }
                            System.arraycopy(output, 0, scratch, 0, dimension);
                            consumer.accept(scratchView);
                            Arrays.fill(scratch, -1);
                        }
                    });
        }
    }
}
