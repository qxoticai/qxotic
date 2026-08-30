package com.qxotic.jinfer.models.qwen3;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.Builder;
import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryAllocators;
import java.lang.foreign.Arena;
import java.util.Map;
import org.junit.jupiter.api.Test;

final class Qwen3ConfigurationTest {

    @Test
    void readsValidMetadata() {
        Qwen3.Configuration config = read(metadata());

        assertEquals(8, config.headSize());
        assertEquals(16, config.queryDim());
        assertEquals(8, config.kvDim());
        assertEquals(2, config.kvMul());
    }

    @Test
    void rejectsInvalidArchitectureDimensionsAndAttention() {
        assertRejected(metadata().putString("general.architecture", "qwen35"), "architecture");
        assertRejected(metadata().putInteger("qwen3.attention.head_count", 0), "dimensions");
        assertRejected(metadata().putInteger("qwen3.attention.head_count_kv", 3), "dimensions");
        assertRejected(metadata().putInteger("qwen3.attention.value_length", 4), "attention");
        assertRejected(metadata().putInteger("qwen3.rope.dimension_count", 7), "attention");
        assertRejected(metadata().putFloat("qwen3.rope.freq_base", Float.NaN), "attention");
        assertRejected(metadata().putInteger("qwen3.vocab_size", 65), "vocabulary");
        assertRejected(
                metadata()
                        .putInteger("qwen3.attention.head_count", Integer.MAX_VALUE)
                        .putInteger("qwen3.attention.head_count_kv", 1),
                "overflow");
    }

    @Test
    void rejectsIncompatibleWeightShapesAtLoad() {
        Qwen3.Configuration config = read(metadata());
        try (Arena arena = Arena.ofConfined()) {
            var memory = MemoryAllocators.ofArena(arena);
            var tensors = Map.of("token_embd.weight", Views.allocateF32(memory, 63, 16));

            IllegalArgumentException failure =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> Qwen3.loadWeights(tensors, config));

            assertTrue(failure.getMessage().contains("token_embd.weight"), failure::getMessage);
        }
    }

    private static Qwen3.Configuration read(Builder metadata) {
        return Qwen3.readConfiguration(metadata.build(), 64);
    }

    private static void assertRejected(Builder metadata, String message) {
        IllegalArgumentException failure =
                assertThrows(IllegalArgumentException.class, () -> read(metadata));
        assertTrue(failure.getMessage().contains(message), failure::getMessage);
    }

    private static Builder metadata() {
        return Builder.newBuilder()
                .putString("general.architecture", "qwen3")
                .putInteger("qwen3.context_length", 128)
                .putInteger("qwen3.embedding_length", 16)
                .putInteger("qwen3.attention.head_count", 2)
                .putInteger("qwen3.attention.head_count_kv", 1)
                .putInteger("qwen3.block_count", 2)
                .putInteger("qwen3.feed_forward_length", 32)
                .putInteger("qwen3.attention.key_length", 8)
                .putInteger("qwen3.attention.value_length", 8)
                .putInteger("qwen3.rope.dimension_count", 8)
                .putFloat("qwen3.attention.layer_norm_rms_epsilon", 1e-6f)
                .putFloat("qwen3.rope.freq_base", 10_000f);
    }
}
