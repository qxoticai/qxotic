package com.qxotic.jinfer.x.llm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.x.PanamaMemoryArena;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.HashSet;
import java.util.Set;
import org.junit.jupiter.api.Test;

final class SamplerTest {
    @Test
    void filtersComposeOverMemoryViews() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> logits = logits(arena, 1, 5, 3, 4, 2);
            Sampler.withTopK(ignored -> -1, 2).sampleToken(logits);
            float[] values = Views.toFloatArray(logits, "logits");
            assertEquals(Set.of(1, 3), finite(values));

            assertEquals(
                    2,
                    Sampler.banning(Sampler.ARGMAX, Set.of(1)).sampleToken(logits(arena, 1, 5, 3)));
        }
    }

    @Test
    void seededSamplingIsDeterministicAndHonorsTopK() {
        Sampler left = Sampler.select(5, 0.8f, 2, 0.95f, 0.05f, 7);
        Sampler right = Sampler.select(5, 0.8f, 2, 0.95f, 0.05f, 7);
        try (Arena arena = Arena.ofConfined()) {
            for (int i = 0; i < 50; i++) {
                int a = left.sampleToken(logits(arena, 1, 5, 3, 4, 2));
                int b = right.sampleToken(logits(arena, 1, 5, 3, 4, 2));
                assertEquals(a, b);
                assertTrue(a == 1 || a == 3);
            }
        }
    }

    private static MemoryView<MemorySegment> logits(Arena arena, float... values) {
        return Views.fromFloatArray(new PanamaMemoryArena(arena), values);
    }

    private static Set<Integer> finite(float[] values) {
        Set<Integer> ids = new HashSet<>();
        for (int i = 0; i < values.length; i++) if (Float.isFinite(values[i])) ids.add(i);
        return ids;
    }
}
