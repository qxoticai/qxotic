package com.qxotic.jinfer.models.llama;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
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

final class GraniteDecoderTest {

    @Test
    void optionalBiasesAgreeAcrossBatchAndIncrementalDecode() {
        try (Arena arena = Arena.ofConfined()) {
            Granite model = model(MemoryAllocators.ofArena(arena), true, true);
            float[][] batched = new float[2][];
            try (Granite.State state = model.newState(8, 2)) {
                model.ingest(state, Batch.score(new int[] {0, 1}));
                batched[0] = logits(model, state, 0);
                batched[1] = logits(model, state, 1);
            }

            try (Granite.State state = model.newState(8, 2)) {
                model.ingest(state, Batch.step(0));
                assertArrayEquals(batched[0], logits(model, state, 0), 1e-5f);
                model.ingest(state, Batch.step(1));
                assertArrayEquals(batched[1], logits(model, state, 0), 1e-5f);
            }

            Granite unbiased = model(MemoryAllocators.ofArena(arena), false, true);
            try (Granite.State state = unbiased.newState(8, 2)) {
                unbiased.ingest(state, Batch.score(new int[] {0, 1}));
                assertFalse(Arrays.equals(batched[1], logits(unbiased, state, 1)));
            }
        }
    }

    @Test
    void ropeCanBeDisabled() {
        try (Arena arena = Arena.ofConfined()) {
            Granite roped = model(MemoryAllocators.ofArena(arena), false, true);
            Granite unroped = model(MemoryAllocators.ofArena(arena), false, false);
            float[] withRope;
            try (Granite.State state = roped.newState(8, 2)) {
                roped.ingest(state, Batch.score(new int[] {0, 1}));
                withRope = logits(roped, state, 1);
            }
            try (Granite.State state = unroped.newState(8, 2)) {
                unroped.ingest(state, Batch.score(new int[] {0, 1}));
                assertFalse(Arrays.equals(withRope, logits(unroped, state, 1)));
            }
        }
    }

    @Test
    void logitsFailsFastWhenTheWeightArenaWasClosedAfterIngest() {
        Granite model;
        Granite.State state;
        try (Arena weights = Arena.ofConfined()) {
            model = model(MemoryAllocators.ofArena(weights), true, true);
            state = model.newState(8, 1);
            model.ingest(state, Batch.step(0));
        }
        try (state) {
            assertThrows(IllegalStateException.class, () -> model.logits(state, 0));
        }
    }

    private static float[] logits(Granite model, Granite.State state, int output) {
        return Views.toFloatArray(
                Views.castToSegmentBacked(model.logits(state, output), "logits"), "logits");
    }

    private static Granite model(
            MemoryAllocator<MemorySegment> memory, boolean withBias, boolean useRope) {
        MemoryView<MemorySegment> none = null;
        Granite.LayerWeights layer =
                new Granite.LayerWeights(
                        vector(memory, 1f, 1f),
                        identity(memory),
                        withBias ? vector(memory, 0.05f, -0.1f) : none,
                        identity(memory),
                        withBias ? vector(memory, -0.2f, 0.1f) : none,
                        identity(memory),
                        withBias ? vector(memory, 0.3f, -0.15f) : none,
                        identity(memory),
                        withBias ? vector(memory, 0.1f, 0.2f) : none,
                        vector(memory, 1f, 1f),
                        identity(memory),
                        withBias ? vector(memory, 0.2f, -0.1f) : none,
                        identity(memory),
                        withBias ? vector(memory, -0.05f, 0.15f) : none,
                        identity(memory),
                        withBias ? vector(memory, 0.1f, 0.25f) : none);
        Granite.Weights weights =
                new Granite.Weights(
                        identity(memory),
                        new Granite.LayerWeights[] {layer},
                        vector(memory, 1f, 1f),
                        RoPE.plain(2, 10_000f),
                        identity(memory));
        return new Granite(configuration(useRope), null, weights);
    }

    private static Granite.Configuration configuration(boolean useRope) {
        return new Granite.Configuration(
                2, 1, 1, 1, 2, 2, 8, 1e-5f, 10_000f, 2, 2, 1f, 1f, 1f, 0f, useRope);
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
