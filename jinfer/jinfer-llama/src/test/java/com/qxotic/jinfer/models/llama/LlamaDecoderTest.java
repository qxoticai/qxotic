package com.qxotic.jinfer.models.llama;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.RoPE;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

final class LlamaDecoderTest {

    private static final Llama.Configuration CONFIG =
            new Llama.Configuration(
                    2, 1, 1, 1, 2, 2, 8, 1e-5f, 10_000f, 2, 2, 1f, 1f, 1f, 0f, 0, 0f, 0);

    @Test
    void optionalBiasesAgreeAcrossBatchIncrementalAndRepeatedLazyTails() {
        try (Arena arena = Arena.ofConfined()) {
            Llama biased = model(MemoryAllocators.ofArena(arena), true);
            float[][] batched = new float[2][];
            try (Llama.State state = biased.newState(8, 2)) {
                biased.ingest(state, Batch.score(new int[] {0, 1}));
                batched[0] = logits(biased, state, 0);
                batched[1] = logits(biased, state, 1);
                assertArrayEquals(batched[0], logits(biased, state, 0));
            }

            try (Llama.State state = biased.newState(8, 2)) {
                biased.ingest(state, Batch.step(0));
                assertArrayEquals(batched[0], logits(biased, state, 0), 1e-6f);
                biased.ingest(state, Batch.step(1));
                assertArrayEquals(batched[1], logits(biased, state, 0), 1e-6f);
            }

            Llama unbiased = model(MemoryAllocators.ofArena(arena), false);
            try (Llama.State state = unbiased.newState(8, 2)) {
                unbiased.ingest(state, Batch.score(new int[] {0, 1}));
                assertFalse(Arrays.equals(batched[1], logits(unbiased, state, 1)));
            }
        }
    }

    @Test
    void logitsFailsFastWhenTheWeightArenaWasClosedAfterIngest() {
        Llama model;
        Llama.State state;
        try (Arena weights = Arena.ofConfined()) {
            model = model(MemoryAllocators.ofArena(weights), true);
            state = model.newState(8, 1);
            model.ingest(state, Batch.step(0));
        }
        try (state) {
            assertThrows(IllegalStateException.class, () -> model.logits(state, 0));
        }
    }

    @Test
    void logitScaleIsADivisor() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryAllocator<MemorySegment> memory = MemoryAllocators.ofArena(arena);
            Llama baseline = model(memory, false, CONFIG);
            Llama scaled =
                    model(
                            memory,
                            false,
                            new Llama.Configuration(
                                    2, 1, 1, 1, 2, 2, 8, 1e-5f, 10_000f, 2, 2, 1f, 1f, 2f, 0f, 0,
                                    0f, 0));
            try (Llama.State baselineState = baseline.newState(8, 1);
                    Llama.State scaledState = scaled.newState(8, 1)) {
                baseline.ingest(baselineState, Batch.step(0));
                scaled.ingest(scaledState, Batch.step(0));
                float[] expected = logits(baseline, baselineState, 0);
                float[] actual = logits(scaled, scaledState, 0);
                for (int i = 0; i < expected.length; i++) {
                    assertEquals(expected[i] / 2f, actual[i]);
                }
            }
        }
    }

    private static float[] logits(Llama model, Llama.State state, int output) {
        return Views.toFloatArray(
                Views.castToSegmentBacked(model.logits(state, output), "logits"), "logits");
    }

    private static Llama model(MemoryAllocator<MemorySegment> memory, boolean withBias) {
        return model(memory, withBias, CONFIG);
    }

    private static Llama model(
            MemoryAllocator<MemorySegment> memory,
            boolean withBias,
            Llama.Configuration configuration) {
        MemoryView<MemorySegment> none = null;
        MemoryView<MemorySegment> bq = withBias ? vector(memory, 0.05f, -0.1f) : none;
        MemoryView<MemorySegment> bk = withBias ? vector(memory, -0.2f, 0.1f) : none;
        MemoryView<MemorySegment> bv = withBias ? vector(memory, 0.3f, -0.15f) : none;
        MemoryView<MemorySegment> bo = withBias ? vector(memory, 0.1f, 0.2f) : none;
        MemoryView<MemorySegment> b1 = withBias ? vector(memory, 0.2f, -0.1f) : none;
        MemoryView<MemorySegment> b2 = withBias ? vector(memory, -0.05f, 0.15f) : none;
        MemoryView<MemorySegment> b3 = withBias ? vector(memory, 0.1f, 0.25f) : none;
        Llama.LayerWeights layer =
                new Llama.LayerWeights(
                        vector(memory, 1f, 1f),
                        identity(memory),
                        bq,
                        identity(memory),
                        bk,
                        identity(memory),
                        bv,
                        identity(memory),
                        bo,
                        vector(memory, 1f, 1f),
                        identity(memory),
                        b1,
                        identity(memory),
                        b2,
                        identity(memory),
                        b3);
        Llama.Weights weights =
                new Llama.Weights(
                        identity(memory),
                        new Llama.LayerWeights[] {layer},
                        vector(memory, 1f, 1f),
                        RoPE.plain(2, 10_000f),
                        identity(memory));
        return new Llama(configuration, null, weights);
    }

    private static MemoryView<MemorySegment> identity(MemoryAllocator<MemorySegment> memory) {
        MemoryView<MemorySegment> value = Views.allocateF32(memory, 2, 2);
        Views.copyFromArray(value, 0, new float[] {1f, 0f, 0f, 1f}, 0, 4, "identity");
        return value;
    }

    private static MemoryView<MemorySegment> vector(
            MemoryAllocator<MemorySegment> memory, float first, float second) {
        return Views.fromFloatArray(memory, new float[] {first, second});
    }
}
