package com.qxotic.jinfer.models.laguna;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.ModelProvider;
import java.util.ServiceLoader;
import org.junit.jupiter.api.Test;

final class LagunaConfigurationTest {

    @Test
    void providerIsRegistered() {
        assertEquals(
                1,
                ServiceLoader.load(ModelProvider.class).stream()
                        .filter(provider -> provider.type().equals(LagunaProvider.class))
                        .count());
    }

    @Test
    void readsVariableHeadsAndHybridRopeMetadata() {
        GGUF gguf =
                metadata()
                        .putArrayOfInteger(
                                "laguna.attention.head_count", new int[] {48, 64, 64, 64})
                        .build();

        Laguna.Configuration config = Laguna.loadConfiguration(gguf, 100_352, "laguna");

        assertArrayEquals(new int[] {48, 64, 64, 64}, config.headCount());
        assertArrayEquals(new boolean[] {false, true, true, true}, config.isSwa());
        assertEquals(6_144, config.queryDim(0));
        assertEquals(8_192, config.queryDim(1));
        assertEquals(64, config.ropeDimensionCount());
        assertEquals(128, config.ropeDimensionCountSwa());
        assertEquals(500_000d, config.ropeTheta());
        assertEquals(10_000d, config.ropeThetaSwa());
        assertEquals(2.5f, config.expertWeightsScale());
        assertEquals(1_024, config.kvCachePositions(0, 1_024));
        assertEquals(512, config.kvCachePositions(1, 1_024));
        assertEquals(7, config.kvCacheIndex(1, 519));
    }

    @Test
    void rejectsUnsupportedExpertSemantics() {
        GGUF gguf =
                metadata()
                        .putInteger("laguna.attention.head_count", 48)
                        .putInteger("laguna.expert_gating_func", 1)
                        .build();

        assertThrows(
                IllegalArgumentException.class,
                () -> Laguna.loadConfiguration(gguf, 100_352, "laguna"));
    }

    @Test
    void readsSlidingWindowPeriod() {
        GGUF gguf =
                metadata()
                        .putInteger("laguna.attention.head_count", 48)
                        .putInteger("laguna.attention.sliding_window_pattern", 2)
                        .build();

        Laguna.Configuration config = Laguna.loadConfiguration(gguf, 100_352, "laguna");

        assertArrayEquals(new boolean[] {false, true, false, true}, config.isSwa());

        GGUF allSwa =
                metadata()
                        .putInteger("laguna.attention.head_count", 48)
                        .putInteger("laguna.attention.sliding_window_pattern", 0)
                        .build();
        assertArrayEquals(
                new boolean[] {true, true, true, true},
                Laguna.loadConfiguration(allSwa, 100_352, "laguna").isSwa());
    }

    @Test
    void doesNotApplyLagunasFinalYarnMagnitudeTwice() {
        float mscale = (float) (1.0 + 0.1 * Math.log(32.0));
        assertEquals(1f, Laguna.yarnKernelAttentionFactor(32f, mscale), 1e-6f);
    }

    private static Builder metadata() {
        return Builder.newBuilder()
                .putInteger("laguna.block_count", 4)
                .putInteger("laguna.context_length", 262_144)
                .putInteger("laguna.embedding_length", 2_048)
                .putInteger("laguna.feed_forward_length", 8_192)
                .putInteger("laguna.attention.head_count_kv", 8)
                .putInteger("laguna.attention.key_length", 128)
                .putInteger("laguna.attention.value_length", 128)
                .putInteger("laguna.attention.sliding_window", 512)
                .putFloat("laguna.attention.layer_norm_rms_epsilon", 1e-6f)
                .putInteger("laguna.leading_dense_block_count", 1)
                .putInteger("laguna.expert_count", 256)
                .putInteger("laguna.expert_used_count", 8)
                .putInteger("laguna.expert_feed_forward_length", 512)
                .putInteger("laguna.expert_shared_feed_forward_length", 512)
                .putFloat("laguna.expert_weights_scale", 2.5f)
                .putBoolean("laguna.expert_weights_norm", true)
                .putInteger("laguna.vocab_size", 100_352)
                .putInteger("laguna.rope.dimension_count", 64)
                .putInteger("laguna.rope.dimension_count_swa", 128)
                .putFloat("laguna.rope.freq_base", 500_000f)
                .putFloat("laguna.rope.freq_base_swa", 10_000f)
                .putString("laguna.rope.scaling.type", "yarn")
                .putFloat("laguna.rope.scaling.factor", 32f)
                .putInteger("laguna.rope.scaling.original_context_length", 8_192)
                .putFloat("laguna.rope.scaling.yarn_beta_fast", 64f)
                .putFloat("laguna.rope.scaling.yarn_beta_slow", 1f)
                .putFloat("laguna.rope.scaling.yarn_attn_factor", 1f);
    }
}
