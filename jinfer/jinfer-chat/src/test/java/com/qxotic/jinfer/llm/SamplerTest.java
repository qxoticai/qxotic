package com.qxotic.jinfer.llm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.PanamaMemoryArena;
import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

final class SamplerTest {
    @Test
    void topKSamplingLeavesDenseLogitsUntouched() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> logits = logits(arena, 1, 5, 3, 4, 2);
            Sampler sampler = new Sampling(0.8f, 0.95f, 2, 0.05f, 7L).sampler(5);
            int token = sampler.sampleToken(logits);

            assertTrue(token == 1 || token == 3);
            assertArrayEquals(
                    new float[] {1f, 5f, 3f, 4f, 2f}, Views.toFloatArray(logits, "logits"));
        }
    }

    @Test
    void validatesVocabularyAndLogitsSize() {
        assertThrows(
                IllegalArgumentException.class, () -> new Sampling(1f, 1f, 0, 0f, 7L).sampler(0));

        Sampler sampler = new Sampling(1f, 1f, 0, 0f, 7L).sampler(3);
        try (Arena arena = Arena.ofConfined()) {
            assertThrows(
                    IllegalArgumentException.class,
                    () -> sampler.sampleToken(logits(arena, 1f, 2f)));
        }
    }

    private static MemoryView<MemorySegment> logits(Arena arena, float... values) {
        return Views.fromFloatArray(new PanamaMemoryArena(arena), values);
    }
}
