package com.qxotic.jinfer.models.lfm2;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGMLType;
import com.qxotic.format.gguf.TensorEntry;
import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryAllocators;
import java.lang.foreign.Arena;
import java.util.Map;
import org.junit.jupiter.api.Test;

final class Lfm2ConfigurationTest {

    @Test
    void readsDenseMetadataAndDoesNotExposeMutableArrays() {
        Lfm2.Configuration config = read(metadata("lfm2"));
        assertEquals(8, config.headSize());
        assertEquals(1, config.numberOfKeyValueHeadsPerLayer()[1]);

        int[] widths = config.feedForwardLength();
        int[] kvHeads = config.numberOfKeyValueHeadsPerLayer();
        widths[0] = 1;
        kvHeads[1] = 0;
        assertEquals(32, config.feedForwardLength()[0]);
        assertEquals(1, config.numberOfKeyValueHeadsPerLayer()[1]);
    }

    @Test
    void readsTheMoeArchitectureExplicitly() {
        Lfm2.Configuration config =
                read(
                        metadata("lfm2moe")
                                .putInteger("lfm2moe.expert_count", 8)
                                .putInteger("lfm2moe.expert_used_count", 2)
                                .putInteger("lfm2moe.expert_feed_forward_length", 24)
                                .putInteger("lfm2moe.leading_dense_block_count", 1)
                                .putInteger("lfm2moe.expert_gating_func", 2));

        assertTrue(config.isMoE());
        assertFalse(config.isMoELayer(0));
        assertTrue(config.isMoELayer(1));
    }

    @Test
    void samplingFallbacksFollowThePublishedModelGeneration() {
        var lfm2 =
                Lfm2Provider.samplingDefaults(
                        metadata("lfm2").build(), read(metadata("lfm2")), false);
        assertEquals(0.3f, lfm2.temperature());
        assertEquals(0.15f, lfm2.minP());

        var lfm25 =
                Lfm2Provider.samplingDefaults(
                        metadata("lfm2").build(), read(metadata("lfm2"), 128_000), false);
        assertEquals(0.1f, lfm25.temperature());
        assertEquals(50, lfm25.topK());

        var lfm25Moe =
                Lfm2Provider.samplingDefaults(
                        metadata("lfm2moe").build(),
                        read(
                                metadata("lfm2moe")
                                        .putInteger("lfm2moe.expert_count", 8)
                                        .putInteger("lfm2moe.expert_used_count", 2)
                                        .putInteger("lfm2moe.expert_feed_forward_length", 24)
                                        .putInteger("lfm2moe.leading_dense_block_count", 1),
                                128_000),
                        false);
        assertEquals(0.2f, lfm25Moe.temperature());
        assertEquals(80, lfm25Moe.topK());

        var named350 =
                Lfm2Provider.samplingDefaults(
                        metadata("lfm2").putString("general.name", "LFM2.5-350M").build(),
                        read(metadata("lfm2")),
                        false);
        assertEquals(0.1f, named350.temperature());
        assertEquals(50, named350.topK());

        var vl450 =
                Lfm2Provider.samplingDefaults(
                        metadata("lfm2").putString("general.name", "LFM2.5-VL-450M").build(),
                        read(metadata("lfm2")),
                        true);
        assertEquals(0.1f, vl450.temperature());
        assertEquals(0.15f, vl450.minP());

        var vl3b =
                Lfm2Provider.samplingDefaults(
                        metadata("lfm2").putString("general.name", "LFM2.5-VL-3B").build(),
                        read(metadata("lfm2"), 128_000),
                        true);
        assertEquals(0.2f, vl3b.temperature());
        assertEquals(50, vl3b.topK());
    }

    @Test
    void providerClaimsOnlyArchitecturesTheLoaderUnderstands() {
        Lfm2Provider provider = new Lfm2Provider();
        assertTrue(provider.supports("lfm2"));
        assertTrue(provider.supports("lfm2moe"));
        assertFalse(provider.supports("lfm"));
        assertFalse(provider.supports("lfm3"));
    }

    @Test
    void rejectsInvalidCoreAndLayerMetadata() {
        assertRejected(metadata("lfm3"), "architecture");
        assertRejected(metadata("lfm2").putInteger("lfm2.attention.head_count", 3), "dimensions");
        assertRejected(
                metadata("lfm2").putArrayOfInteger("lfm2.feed_forward_length", new int[] {32}),
                "feed-forward");
        assertRejected(
                metadata("lfm2")
                        .putTensor(
                                TensorEntry.create(
                                        "blk.1.attn_k.weight",
                                        new long[] {16, 7},
                                        GGMLType.F32,
                                        0)),
                "K projection");
    }

    @Test
    void rejectsIncoherentMoeAndNumericMetadata() {
        assertRejected(metadata("lfm2").putInteger("lfm2.expert_count", 8), "MoE");
        assertRejected(
                metadata("lfm2").putInteger("lfm2.shortconv.l_cache", 0), "short-convolution");
        assertRejected(metadata("lfm2").putFloat("lfm2.rope.freq_base", Float.NaN), "RoPE");
        assertRejected(metadata("lfm2").putInteger("lfm2.vocab_size", 65), "vocabulary");
    }

    @Test
    void rejectsIncompatibleWeightShapesAtLoad() {
        Lfm2.Configuration config = read(metadata("lfm2"));
        try (Arena arena = Arena.ofConfined()) {
            var memory = MemoryAllocators.ofArena(arena);
            var tensors = Map.of("token_embd.weight", Views.allocateF32(memory, 63, 16));

            IllegalArgumentException failure =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> Lfm2.loadWeights(tensors, config));

            assertTrue(failure.getMessage().contains("token_embd.weight"), failure::getMessage);
        }
    }

    private static Lfm2.Configuration read(Builder metadata) {
        return read(metadata, 64);
    }

    private static Lfm2.Configuration read(Builder metadata, int vocabularySize) {
        var gguf = metadata.build();
        return Lfm2.readConfiguration(gguf, vocabularySize);
    }

    private static void assertRejected(Builder metadata, String message) {
        IllegalArgumentException failure =
                assertThrows(IllegalArgumentException.class, () -> read(metadata));
        assertTrue(failure.getMessage().contains(message), failure::getMessage);
    }

    private static Builder metadata(String arch) {
        String prefix = arch.equals("lfm3") ? "lfm2" : arch;
        return Builder.newBuilder()
                .putString("general.architecture", arch)
                .putInteger(prefix + ".context_length", 128)
                .putInteger(prefix + ".embedding_length", 16)
                .putInteger(prefix + ".attention.head_count", 2)
                .putInteger(prefix + ".block_count", 2)
                .putInteger(prefix + ".feed_forward_length", 32)
                .putFloat(prefix + ".attention.layer_norm_rms_epsilon", 1e-5f)
                .putFloat(prefix + ".rope.freq_base", 10_000f)
                .putTensor(
                        TensorEntry.create(
                                "blk.1.attn_k.weight", new long[] {16, 8}, GGMLType.F32, 0));
    }
}
