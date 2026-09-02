package com.qxotic.jinfer.models.llama;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.RoPE;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

final class LlamaConfigurationTest {

    @Test
    void usesReferenceMiniCpmScalesAndHonorsOverrides() {
        Llama.Configuration defaults =
                Llama.readConfiguration(metadata("minicpm").build(), "minicpm", 8);
        assertEquals(12f, defaults.embeddingScale());
        assertEquals(512f / 256f, defaults.logitScale());

        GGUF overridden =
                metadata("minicpm")
                        .putFloat("minicpm.embedding_scale", 2f)
                        .putFloat("minicpm.residual_scale", 3f)
                        .putFloat("minicpm.logit_scale", 4f)
                        .build();
        Llama.Configuration config = Llama.readConfiguration(overridden, "minicpm", 8);
        assertEquals(2f, config.embeddingScale());
        assertEquals(3f, config.residualScale());
        assertEquals(4f, config.logitScale());
    }

    @Test
    void rejectsIncompatibleMetadataAtTheBoundary() {
        assertRejected(
                metadata("llama").putInteger("llama.attention.value_length", 32), "attention");
        assertRejected(metadata("llama").putInteger("llama.expert_count", 8), "MoE");
        assertRejected(metadata("llama").putInteger("llama.vocab_size", 9), "vocabulary");
        assertRejected(
                metadata("llama").putFloat("llama.attention.temperature_scale", 0.1f),
                "attention scaling");
    }

    @Test
    void onlyAllowsNonDivisibleEmbeddingsWithAnExplicitHeadSize() {
        assertRejected(metadata("llama").putInteger("llama.embedding_length", 513), "attention");

        Llama.Configuration explicit =
                Llama.readConfiguration(
                        metadata("llama")
                                .putInteger("llama.embedding_length", 513)
                                .putInteger("llama.attention.key_length", 8)
                                .build(),
                        "llama",
                        8);
        assertEquals(8, explicit.headSize());
    }

    @Test
    void ropeModesAreExplicitAndFrequencyFactorsAreChecked() {
        Llama.Configuration config = Llama.readConfiguration(metadata("llama").build(), "llama", 8);
        // linear is what llama.cpp writes by default: no factor collapses to plain RoPE, a
        // factor stretches positions (angles at pos*f match plain at pos), garbage is refused
        float[] plain = new float[4];
        float[] scaled = new float[4];
        Llama.buildRope(metadata("llama").build(), "llama", config, Map.of()).angles(1, plain);
        Llama.buildRope(
                        metadata("llama").putString("llama.rope.scaling.type", "linear").build(),
                        "llama",
                        config,
                        Map.of())
                .angles(1, scaled);
        assertArrayEquals(plain, scaled);
        Llama.buildRope(
                        metadata("llama")
                                .putString("llama.rope.scaling.type", "linear")
                                .putFloat("llama.rope.scaling.factor", 2f)
                                .build(),
                        "llama",
                        config,
                        Map.of())
                .angles(2, scaled);
        assertArrayEquals(plain, scaled, 1e-6f);
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Llama.buildRope(
                                metadata("llama")
                                        .putString("llama.rope.scaling.type", "linear")
                                        .putFloat("llama.rope.scaling.factor", -2f)
                                        .build(),
                                "llama",
                                config,
                                Map.of()));
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Llama.buildRope(
                                metadata("llama")
                                        .putString("llama.rope.scaling.type", "llama3")
                                        .build(),
                                "llama",
                                config,
                                Map.of()));

        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> factors =
                    Views.fromFloatArray(
                            MemoryAllocators.ofArena(arena), new float[] {1f, 2f, 3f, 4f});
            RoPE.Schedule rope =
                    Llama.buildRope(
                            metadata("llama")
                                    .putString("llama.rope.scaling.type", "llama3")
                                    .build(),
                            "llama",
                            config,
                            Map.of("rope_freqs.weight", factors));
            float[] angles = new float[4];
            rope.angles(1, angles);
            assertTrue(Float.isFinite(angles[3]));
        }
    }

    @Test
    void loadsOptionalBiasesAndRejectsWrongWidthsAndFusedQkv() {
        Llama.Configuration config = Llama.readConfiguration(metadata("llama").build(), "llama", 8);
        try (Arena arena = Arena.ofConfined()) {
            var memory = MemoryAllocators.ofArena(arena);
            Map<String, MemoryView<MemorySegment>> tensors = denseTensors(memory, config);
            MemoryView<MemorySegment> qBias = Views.allocateF32(memory, config.queryDim());
            tensors.put("blk.0.attn_q.bias", qBias);
            Llama.Weights weights = Llama.loadWeights(tensors, config, RoPE.plain(8, 10_000f));
            assertSame(qBias, weights.layers()[0].bq());

            tensors.put("blk.0.attn_q.bias", Views.allocateF32(memory, config.queryDim() + 1));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> Llama.loadWeights(tensors, config, RoPE.plain(8, 10_000f)));

            tensors.remove("blk.0.attn_q.weight");
            tensors.put("blk.0.attn_qkv.weight", Views.allocateF32(memory, 64, 512));
            IllegalArgumentException fused =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> Llama.loadWeights(tensors, config, RoPE.plain(8, 10_000f)));
            assertTrue(fused.getMessage().contains("fused QKV"));
        }
    }

    private static void assertRejected(Builder metadata, String message) {
        IllegalArgumentException failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Llama.readConfiguration(metadata.build(), "llama", 8));
        assertTrue(failure.getMessage().contains(message), failure::getMessage);
    }

    private static Builder metadata(String arch) {
        return Builder.newBuilder()
                .putString("general.architecture", arch)
                .putInteger(arch + ".context_length", 64)
                .putInteger(arch + ".embedding_length", 512)
                .putInteger(arch + ".block_count", 1)
                .putInteger(arch + ".attention.head_count", 64)
                .putInteger(arch + ".attention.head_count_kv", 8)
                .putInteger(arch + ".feed_forward_length", 16)
                .putInteger(arch + ".rope.dimension_count", 8)
                .putFloat(arch + ".attention.layer_norm_rms_epsilon", 1e-5f)
                .putFloat(arch + ".rope.freq_base", 10_000f);
    }

    private static Map<String, MemoryView<MemorySegment>> denseTensors(
            com.qxotic.jota.memory.MemoryAllocator<MemorySegment> memory,
            Llama.Configuration config) {
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
