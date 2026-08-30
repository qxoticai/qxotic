package com.qxotic.jinfer.models.llama;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.RoPE;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

final class GraniteConfigurationTest {

    @Test
    void normalizesZeroScaleSentinelsAndHonorsTheRopeSwitch() {
        GGUF gguf =
                metadata()
                        .putFloat("granite.embedding_scale", 0f)
                        .putFloat("granite.residual_scale", 0f)
                        .putBoolean("granite.rope.scaling.finetuned", false)
                        .build();
        Granite.Configuration config = Granite.readConfiguration(gguf, "granite", 8);
        assertEquals(1f, config.embeddingScale());
        assertEquals(1f, config.residualScale());
        assertFalse(config.useRope());
    }

    @Test
    void rejectsInvalidAndUnsupportedMetadataAtTheBoundary() {
        assertRejected(metadata().putInteger("granite.attention.value_length", 4), "attention");
        assertRejected(metadata().putInteger("granite.expert_count", 8), "MoE");
        assertRejected(
                metadata().putArrayOfInteger("granite.deepstack_mapping", new int[] {-1}),
                "deepstack");
        assertRejected(metadata().putInteger("granite.vocab_size", 9), "vocabulary");
        assertRejected(metadata().putFloat("granite.logit_scale", 0f), "model scaling");
    }

    @Test
    void ropeModesAndFactorsAreExplicit() {
        Granite.Configuration config = Granite.readConfiguration(metadata().build(), "granite", 8);
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Granite.buildRope(
                                metadata().putString("granite.rope.scaling.type", "yarn").build(),
                                "granite",
                                config,
                                Map.of()));

        RoPE.Schedule linear =
                Granite.buildRope(
                        metadata()
                                .putString("granite.rope.scaling.type", "linear")
                                .putFloat("granite.rope.scaling.factor", 2f)
                                .build(),
                        "granite",
                        config,
                        Map.of());
        float[] angles = new float[4];
        linear.angles(2, angles);
        assertEquals(1f, angles[0]);

        RoPE.Schedule none =
                Granite.buildRope(
                        metadata()
                                .putString("granite.rope.scaling.type", "none")
                                .putFloat("granite.rope.scaling.factor", 2f)
                                .build(),
                        "granite",
                        config,
                        Map.of());
        none.angles(2, angles);
        assertEquals(2f, angles[0]);
    }

    @Test
    void loadsOptionalBiasesAndRejectsWrongWidthsAndFusedQkv() {
        Granite.Configuration config = Granite.readConfiguration(metadata().build(), "granite", 8);
        try (Arena arena = Arena.ofConfined()) {
            MemoryAllocator<MemorySegment> memory = MemoryAllocators.ofArena(arena);
            Map<String, MemoryView<MemorySegment>> tensors = denseTensors(memory, config);
            MemoryView<MemorySegment> outputBias =
                    Views.allocateF32(memory, config.embeddingLength());
            tensors.put("blk.0.attn_output.bias", outputBias);
            Granite.Weights weights = Granite.loadWeights(tensors, config, RoPE.plain(8, 10_000f));
            assertSame(outputBias, weights.layers()[0].bo());

            tensors.put(
                    "blk.0.attn_output.bias",
                    Views.allocateF32(memory, config.embeddingLength() + 1));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> Granite.loadWeights(tensors, config, RoPE.plain(8, 10_000f)));

            tensors.remove("blk.0.attn_q.weight");
            tensors.put("blk.0.attn_qkv.weight", Views.allocateF32(memory, 80, 512));
            IllegalArgumentException fused =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> Granite.loadWeights(tensors, config, RoPE.plain(8, 10_000f)));
            assertTrue(fused.getMessage().contains("fused QKV"));
        }
    }

    private static void assertRejected(Builder metadata, String message) {
        IllegalArgumentException failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Granite.readConfiguration(metadata.build(), "granite", 8));
        assertTrue(failure.getMessage().contains(message), failure::getMessage);
    }

    private static Builder metadata() {
        return Builder.newBuilder()
                .putString("general.architecture", "granite")
                .putInteger("granite.context_length", 64)
                .putInteger("granite.embedding_length", 512)
                .putInteger("granite.block_count", 1)
                .putInteger("granite.attention.head_count", 64)
                .putInteger("granite.attention.head_count_kv", 8)
                .putInteger("granite.attention.key_length", 8)
                .putInteger("granite.attention.value_length", 8)
                .putInteger("granite.feed_forward_length", 16)
                .putInteger("granite.rope.dimension_count", 8)
                .putFloat("granite.attention.layer_norm_rms_epsilon", 1e-5f)
                .putFloat("granite.rope.freq_base", 10_000f)
                .putFloat("granite.logit_scale", 1f);
    }

    private static Map<String, MemoryView<MemorySegment>> denseTensors(
            MemoryAllocator<MemorySegment> memory, Granite.Configuration config) {
        Map<String, MemoryView<MemorySegment>> tensors = new HashMap<>();
        tensors.put(
                "token_embd.weight",
                Views.allocateF32(memory, config.vocabularySize(), config.embeddingLength()));
        tensors.put("output_norm.weight", Views.allocateF32(memory, config.embeddingLength()));
        String p = "blk.0.";
        tensors.put(p + "attn_norm.weight", Views.allocateF32(memory, config.embeddingLength()));
        tensors.put(
                p + "attn_q.weight",
                Views.allocateF32(memory, config.queryDim(), config.embeddingLength()));
        tensors.put(
                p + "attn_k.weight",
                Views.allocateF32(memory, config.kvDim(), config.embeddingLength()));
        tensors.put(
                p + "attn_v.weight",
                Views.allocateF32(memory, config.kvDim(), config.embeddingLength()));
        tensors.put(
                p + "attn_output.weight",
                Views.allocateF32(memory, config.embeddingLength(), config.queryDim()));
        tensors.put(p + "ffn_norm.weight", Views.allocateF32(memory, config.embeddingLength()));
        tensors.put(
                p + "ffn_gate.weight",
                Views.allocateF32(memory, config.hiddenDim(), config.embeddingLength()));
        tensors.put(
                p + "ffn_down.weight",
                Views.allocateF32(memory, config.embeddingLength(), config.hiddenDim()));
        tensors.put(
                p + "ffn_up.weight",
                Views.allocateF32(memory, config.hiddenDim(), config.embeddingLength()));
        return tensors;
    }
}
