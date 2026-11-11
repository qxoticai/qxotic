package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
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
import com.qxotic.jota.memory.Memories;
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
import org.junit.jupiter.api.function.Executable;

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
    void aTextTemplateOwnsFramingWhileThePackerOwnsLengthsAndInference() {
        ProbeModel model = new ProbeModel(4);
        List<String> rendered = new ArrayList<>();
        LoadedEmbedder<ProbeState> loaded =
                new LoadedEmbedder<>(
                        model,
                        tokenizer(),
                        IntSequence.of(99),
                        IntSequence.of(100),
                        4,
                        4,
                        "templated",
                        "",
                        "",
                        text -> {
                            rendered.add(text);
                            return IntSequence.of(7, text.length(), 8);
                        });
        try (ProbeState state = model.newState(6, 6)) {
            List<float[]> vectors = new ArrayList<>();
            assertEquals(6, loaded.embedAll(state, List.of("x", "yz"), vectors::add));
            assertEquals(List.of("x", "yz"), rendered);
            assertEquals(1, model.calls);
            assertEquals(2, vectors.size());
            assertArrayEquals(new int[] {7, 1, 8, 7, 2, 8}, model.batches.getFirst());
            assertArrayEquals(new int[] {3, 3}, model.lengths.getFirst());
        }
        try (ProbeState state = model.newState(2, 2)) {
            IllegalArgumentException failure =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> loaded.embedAll(state, List.of("x"), 4, 8, ignored -> {}));
            assertTrue(failure.getMessage().startsWith("text 8 is 3 tokens,"));
            assertEquals(1, model.calls, "rendered overflow is refused before inference");
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
    void zeroPrefixStaysFiniteAndClosedViewsFailAtTheOwnedBoundary() {
        ProbeModel zeroModel = new ProbeModel(4).outputs(new float[] {0, 0, 1, 0});
        LoadedEmbedder<ProbeState> zero = loaded(zeroModel, 2);
        try (ProbeState state = zeroModel.newState(8, 8)) {
            List<float[]> vectors = new ArrayList<>();
            zero.embedAll(state, List.of("x"), 2, vectors::add);
            assertArrayEquals(new float[] {0, 0}, vectors.get(0));
        }

        Arena arena = Arena.ofShared();
        ProbeModel closedModel = new ProbeModel(4);
        closedModel.projected =
                Views.wrap(arena.allocate(4L * Float.BYTES), DataType.FP32, Shape.flat(4));
        arena.close();
        LoadedEmbedder<ProbeState> closed = loaded(closedModel, 4);
        try (ProbeState state = closedModel.newState(8, 8)) {
            IllegalStateException failure =
                    assertThrows(
                            IllegalStateException.class,
                            () -> closed.embedAll(state, List.of("x"), ignored -> {}));
            assertTrue(failure.getMessage().contains("backing memory already freed"));
            assertNull(failure.getCause(), "an existing state failure is not wrapped again");
        }
    }

    @Test
    void invalidModelOutputsAreStateFailuresOnBothEmbeddingPaths() {
        List<MemoryView<?>> invalid =
                List.of(
                        Views.wrap(
                                MemorySegment.ofArray(new float[3]), DataType.FP32, Shape.flat(3)),
                        Views.wrap(
                                MemorySegment.ofArray(new double[4]), DataType.FP64, Shape.flat(4)),
                        Views.wrap(
                                        MemorySegment.ofArray(new float[4]),
                                        DataType.FP32,
                                        Shape.flat(2, 2))
                                .transpose(0, 1),
                        MemoryView.rowMajor(
                                Memories.of(new float[4]), DataType.FP32, Shape.flat(4)));
        for (MemoryView<?> view : invalid) {
            ProbeModel model = new ProbeModel(4);
            model.projected = view;
            LoadedEmbedder<ProbeState> loaded = loaded(model, 4);
            try (ProbeState state = model.newState(8, 8)) {
                List<float[]> vectors = new ArrayList<>();
                List<Executable> paths =
                        List.of(
                                () -> loaded.embedAll(state, List.of("x"), vectors::add),
                                () -> loaded.embed(state, List.of("x"), 4, vectors::add));
                for (Executable path : paths) {
                    IllegalStateException failure = assertThrows(IllegalStateException.class, path);
                    if (view.shape().size() != 4) {
                        assertEquals(
                                "embedding: expected 4 values but received 3",
                                failure.getMessage());
                        assertNull(failure.getCause());
                    } else {
                        assertEquals("invalid embedding output", failure.getMessage());
                        assertInstanceOf(IllegalArgumentException.class, failure.getCause());
                        assertTrue(
                                failure.getCause().getMessage().startsWith("embedding: expected "));
                    }
                    assertTrue(vectors.isEmpty(), "invalid output must not reach the consumer");
                }
                model.projected =
                        Views.wrap(
                                MemorySegment.ofArray(new float[4]), DataType.FP32, Shape.flat(4));
                for (Executable path : paths) assertDoesNotThrow(path);
                assertEquals(
                        2,
                        vectors.size(),
                        "both paths remain usable after output validation fails");
            }
        }
    }

    @Test
    void invalidInputsFailBeforeInferenceAndEmptyInputIsARealNoop() {
        ProbeModel model = new ProbeModel(4);
        LoadedEmbedder<ProbeState> loaded = loaded(model, 4);
        try (ProbeState state = model.newState(3, 3)) {
            assertEquals(0, loaded.embedAll(state, List.of(), ignored -> {}));
            assertEquals(0, model.calls);

            IllegalArgumentException failure =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> loaded.embedAll(state, List.of("abcd"), ignored -> {}));
            assertTrue(failure.getMessage().startsWith("text 0 is 5 tokens,"));
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
    void aTextRunRetainsItsOriginalIndexesInValidationErrors() {
        ProbeModel model = new ProbeModel(4);
        LoadedEmbedder<ProbeState> loaded = loaded(model, 4);
        List<float[]> vectors = new ArrayList<>();
        try (ProbeState state = model.newState(3, 3)) {
            IllegalArgumentException tooLong =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> loaded.embedAll(state, List.of("a", "abcd"), 4, 5, vectors::add));
            assertTrue(tooLong.getMessage().startsWith("text 6 is 5 tokens,"));
            NullPointerException missing =
                    assertThrows(
                            NullPointerException.class,
                            () ->
                                    loaded.embedAll(
                                            state, Arrays.asList("a", null), 4, 5, vectors::add));
            assertEquals("texts[6]", missing.getMessage());
            LoadedEmbedder<ProbeState> unframed =
                    loaded(model, 4, IntSequence.empty(), IntSequence.empty());
            IllegalArgumentException empty =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> unframed.embedAll(state, List.of("a", ""), 4, 5, vectors::add));
            assertEquals("text 6 produced no tokens", empty.getMessage());
            assertEquals(0, loaded.embedAll(state, List.of(), 4, 5, vectors::add));
            assertEquals(0, model.calls);
            assertTrue(vectors.isEmpty());
        }
    }

    @Test
    void anInvalidInputIndexRangeIsRefusedBeforeReadingTexts() {
        ProbeModel model = new ProbeModel(4);
        LoadedEmbedder<ProbeState> loaded = loaded(model, 4);
        try (ProbeState state = model.newState(3, 3)) {
            // Null would fail on access: range validation must happen first.
            List<String> texts = Arrays.asList(null, "a");
            for (int firstInputIndex : new int[] {-1, Integer.MAX_VALUE}) {
                IllegalArgumentException failure =
                        assertThrows(
                                IllegalArgumentException.class,
                                () ->
                                        loaded.embedAll(
                                                state, texts, 4, firstInputIndex, ignored -> {}));
                assertTrue(failure.getMessage().startsWith("firstInputIndex " + firstInputIndex));
            }
            assertEquals(0, model.calls);
        }
    }

    @Test
    void theDiagnosticOffsetDoesNotChangePackingTokensOrVectors() {
        ProbeModel model =
                new ProbeModel(4).outputs(new float[] {1, 2, 3, 4}, new float[] {5, 6, 7, 8});
        LoadedEmbedder<ProbeState> loaded = loaded(model, 4);
        try (ProbeState state = model.newState(3, 3)) {
            List<String> texts = List.of("ab", "c");
            List<float[]> original = new ArrayList<>();
            int tokens = loaded.embedAll(state, texts, original::add);
            List<float[]> offset = new ArrayList<>();
            assertEquals(
                    tokens, loaded.embedAll(state, texts, 4, Integer.MAX_VALUE - 2, offset::add));
            assertEquals(5, tokens);
            assertEquals(2, offset.size());
            assertEquals(4, model.calls, "two packed groups on each call");
            for (int i = 0; i < 2; i++) {
                assertArrayEquals(original.get(i), offset.get(i));
                assertArrayEquals(model.batches.get(i), model.batches.get(i + 2));
                assertArrayEquals(model.lengths.get(i), model.lengths.get(i + 2));
            }
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

            IllegalArgumentException expected = new IllegalArgumentException("consumer failed");
            IllegalArgumentException actual =
                    assertThrows(
                            IllegalArgumentException.class,
                            () ->
                                    loaded.embedAll(
                                            state,
                                            List.of("x"),
                                            ignored -> {
                                                throw expected;
                                            }));
            assertSame(expected, actual);
            loaded.embedAll(state, List.of("x"), ignored -> {});
            model.projected =
                    Views.wrap(MemorySegment.ofArray(new float[4]), DataType.FP32, Shape.flat(4));
            assertSame(
                    expected,
                    assertThrows(
                            IllegalArgumentException.class,
                            () ->
                                    loaded.embed(
                                            state,
                                            List.of("x"),
                                            4,
                                            ignored -> {
                                                throw expected;
                                            })));
            loaded.embed(state, List.of("x"), 4, ignored -> {});
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

    private record Configuration(int vocabularySize, int maxContextLength)
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
        public int embed(ProbeState state, List<?> parts, Consumer<MemoryView<?>> consumer) {
            state.exclusively(() -> consumer.accept(projected));
            return 1;
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
