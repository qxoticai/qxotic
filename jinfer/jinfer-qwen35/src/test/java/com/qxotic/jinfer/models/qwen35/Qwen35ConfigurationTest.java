package com.qxotic.jinfer.models.qwen35;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.Builder;
import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryAllocators;
import java.lang.foreign.Arena;
import java.util.Map;
import org.junit.jupiter.api.Test;

final class Qwen35ConfigurationTest {

    @Test
    void readsExplicitLayerLayoutAndSamplingDefaults() {
        Qwen35.Configuration config = read(metadata());

        assertArrayEquals(new boolean[] {false, true, false, true, true}, config.isFullAttention());
        assertArrayEquals(new int[] {2, 1, 1}, config.ropeDimensionSections());
        assertEquals(1.0f, Qwen35Provider.SAMPLING_DEFAULTS.temperature());
        assertEquals(0.95f, Qwen35Provider.SAMPLING_DEFAULTS.topP());
        assertEquals(20, Qwen35Provider.SAMPLING_DEFAULTS.topK());

        Qwen35Provider provider = new Qwen35Provider();
        assertTrue(provider.supports("qwen35"));
        assertTrue(provider.supports("qwen35moe"));
        assertFalse(provider.supports("qwen3"));
    }

    @Test
    void rejectsInvalidCoreAttentionAndLayerMetadata() {
        assertRejected(metadata().putString("general.architecture", "qwen3"), "architecture");
        assertRejected(metadata().putInteger("qwen35.attention.head_count", 0), "dimensions");
        assertRejected(metadata().putInteger("qwen35.attention.value_length", 4), "attention");
        assertRejected(
                metadata().putArrayOfInteger("qwen35.rope.dimension_sections", new int[] {3, 1, 1}),
                "attention");
        assertRejected(metadata().putFloat("qwen35.rope.freq_base", Float.NaN), "attention");
        assertRejected(metadata().putInteger("qwen35.ssm.time_step_rank", 3), "attention");
        assertRejected(
                metadata()
                        .putArrayOfInteger(
                                "qwen35.attention.recurrent_layers", new int[] {1, 0, 1}),
                "recurrent-layer");
        assertRejected(metadata().putInteger("qwen35.vocab_size", 65), "vocabulary");
        assertRejected(
                metadata()
                        .putInteger("qwen35.attention.head_count", Integer.MAX_VALUE)
                        .putInteger("qwen35.attention.head_count_kv", 1),
                "overflow");
    }

    @Test
    void rejectsArchitectureAndTensorShapeMismatches() {
        assertRejected(
                metadata()
                        .putString("general.architecture", "qwen35moe")
                        .putInteger("qwen35moe.context_length", 128)
                        .putInteger("qwen35moe.embedding_length", 16)
                        .putInteger("qwen35moe.block_count", 5)
                        .putInteger("qwen35moe.nextn_predict_layers", 1)
                        .putInteger("qwen35moe.attention.head_count", 2)
                        .putInteger("qwen35moe.attention.head_count_kv", 1)
                        .putInteger("qwen35moe.attention.key_length", 8)
                        .putInteger("qwen35moe.attention.value_length", 8)
                        .putInteger("qwen35moe.rope.dimension_count", 8)
                        .putArrayOfInteger("qwen35moe.rope.dimension_sections", new int[] {2, 1, 1})
                        .putInteger("qwen35moe.ssm.inner_size", 8)
                        .putInteger("qwen35moe.ssm.group_count", 2)
                        .putInteger("qwen35moe.ssm.time_step_rank", 4)
                        .putInteger("qwen35moe.ssm.state_size", 2)
                        .putInteger("qwen35moe.ssm.conv_kernel", 4),
                "MoE");

        try (Arena arena = Arena.ofConfined()) {
            var rows = Views.allocateF32(MemoryAllocators.ofArena(arena), 63, 16);
            IllegalArgumentException failure =
                    assertThrows(
                            IllegalArgumentException.class,
                            () ->
                                    Qwen35.loadWeights(
                                            Map.of("token_embd.weight", rows), read(metadata())));
            assertTrue(failure.getMessage().contains("token_embd.weight"), failure::getMessage);
        }
    }

    private static Qwen35.Configuration read(Builder metadata) {
        return Qwen35.readConfiguration(metadata.build(), 64);
    }

    private static void assertRejected(Builder metadata, String message) {
        IllegalArgumentException failure =
                assertThrows(IllegalArgumentException.class, () -> read(metadata));
        assertTrue(failure.getMessage().contains(message), failure::getMessage);
    }

    private static Builder metadata() {
        return Builder.newBuilder()
                .putString("general.architecture", "qwen35")
                .putInteger("qwen35.context_length", 128)
                .putInteger("qwen35.embedding_length", 16)
                .putInteger("qwen35.block_count", 5)
                .putInteger("qwen35.nextn_predict_layers", 1)
                .putInteger("qwen35.attention.head_count", 2)
                .putInteger("qwen35.attention.head_count_kv", 1)
                .putInteger("qwen35.attention.key_length", 8)
                .putInteger("qwen35.attention.value_length", 8)
                .putInteger("qwen35.rope.dimension_count", 8)
                .putArrayOfInteger("qwen35.rope.dimension_sections", new int[] {2, 1, 1})
                .putInteger("qwen35.full_attention_interval", 2)
                .putArrayOfInteger("qwen35.attention.recurrent_layers", new int[] {1, 0, 1, 0})
                .putInteger("qwen35.feed_forward_length", 32)
                .putInteger("qwen35.ssm.inner_size", 8)
                .putInteger("qwen35.ssm.group_count", 2)
                .putInteger("qwen35.ssm.time_step_rank", 4)
                .putInteger("qwen35.ssm.state_size", 2)
                .putInteger("qwen35.ssm.conv_kernel", 4)
                .putFloat("qwen35.attention.layer_norm_rms_epsilon", 1e-6f)
                .putFloat("qwen35.rope.freq_base", 1_000_000f);
    }
}
