package com.qxotic.jinfer.models.mellum;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.ModelProvider;
import java.util.ServiceLoader;
import org.junit.jupiter.api.Test;

final class MellumConfigurationTest {

    @Test
    void providerIsRegistered() {
        assertEquals(
                1,
                ServiceLoader.load(ModelProvider.class).stream()
                        .filter(provider -> provider.type().equals(MellumProvider.class))
                        .count());
    }

    @Test
    void readsThePerLayerWindowPatternAndYarnMetadata() {
        GGUF gguf =
                metadata()
                        .putArrayOfBoolean(
                                "mellum.attention.sliding_window_pattern",
                                new boolean[] {true, true, true, false})
                        .build();

        Mellum.Configuration config = Mellum.loadConfiguration(gguf, 98_304);

        assertArrayEquals(new boolean[] {true, true, true, false}, config.isSwa());
        assertEquals(4_096, config.queryDim());
        assertEquals(512, config.kvDim());
        assertEquals(128, config.ropeDimensionCount());
        assertEquals(500_000d, config.ropeTheta());
        assertEquals(500_000d, config.ropeThetaSwa());
        assertEquals(16f, config.ropeScalingFactor());
        assertEquals(8_192, config.ropeOriginalContext());
        assertEquals(
                1f, config.ropeAttentionFactor(), "the converter's derived magnitude is not read");
        assertEquals(1_024, config.kvCachePositions(0, 4_096));
        assertEquals(4_096, config.kvCachePositions(3, 4_096));
        assertEquals(7, config.kvCacheIndex(0, 1_031));
        assertEquals(1_031, config.kvCacheIndex(3, 1_031));
        assertEquals(8, config.attentionStart(0, 1_031));
        assertEquals(0, config.attentionStart(3, 1_031));
    }

    @Test
    void readsAWindowPeriodLikeLlamaCpp() {
        GGUF period = metadata().putInteger("mellum.attention.sliding_window_pattern", 2).build();
        assertArrayEquals(
                new boolean[] {false, true, false, true},
                Mellum.loadConfiguration(period, 98_304).isSwa());

        GGUF defaulted = metadata().build();
        assertArrayEquals(
                new boolean[] {false, true, true, true},
                Mellum.loadConfiguration(defaulted, 98_304).isSwa(),
                "no pattern key: llama.cpp's default period of 4");

        GGUF everyLayer =
                metadata().putInteger("mellum.attention.sliding_window_pattern", 0).build();
        assertArrayEquals(
                new boolean[] {true, true, true, true},
                Mellum.loadConfiguration(everyLayer, 98_304).isSwa());

        GGUF noWindow = metadata().putInteger("mellum.attention.sliding_window", 0).build();
        assertArrayEquals(
                new boolean[] {false, false, false, false},
                Mellum.loadConfiguration(noWindow, 98_304).isSwa());
    }

    @Test
    void rejectsWhatThePortCannotRun() {
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Mellum.loadConfiguration(
                                metadata()
                                        .putInteger("mellum.attention.sliding_window", 1_000)
                                        .build(),
                                98_304),
                "a ring needs a power-of-two window");
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Mellum.loadConfiguration(
                                metadata().putString("mellum.rope.scaling.type", "linear").build(),
                                98_304));
        assertThrows(
                IllegalArgumentException.class,
                () -> Mellum.loadConfiguration(metadata().build(), 98_305),
                "tokenizer and model vocabularies must agree");
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Mellum.loadConfiguration(
                                metadata().putInteger("mellum.expert_used_count", 65).build(),
                                98_304));
    }

    private static Builder metadata() {
        return Builder.newBuilder()
                .putInteger("mellum.block_count", 4)
                .putInteger("mellum.context_length", 131_072)
                .putInteger("mellum.embedding_length", 2_304)
                .putInteger("mellum.feed_forward_length", 7_168)
                .putInteger("mellum.attention.head_count", 32)
                .putInteger("mellum.attention.head_count_kv", 4)
                .putInteger("mellum.attention.key_length", 128)
                .putInteger("mellum.attention.value_length", 128)
                .putInteger("mellum.attention.sliding_window", 1_024)
                .putFloat("mellum.attention.layer_norm_rms_epsilon", 1e-6f)
                .putInteger("mellum.expert_count", 64)
                .putInteger("mellum.expert_used_count", 8)
                .putInteger("mellum.expert_feed_forward_length", 896)
                .putInteger("mellum.vocab_size", 98_304)
                .putFloat("mellum.rope.freq_base", 500_000f)
                .putFloat("mellum.rope.freq_base_swa", 500_000f)
                .putString("mellum.rope.scaling.type", "yarn")
                .putFloat("mellum.rope.scaling.factor", 16f)
                .putInteger("mellum.rope.scaling.original_context_length", 8_192)
                .putFloat("mellum.rope.scaling.yarn_attn_factor", 1.2772588f)
                .putFloat("mellum.rope.scaling.yarn_beta_fast", 32f)
                .putFloat("mellum.rope.scaling.yarn_beta_slow", 1f);
    }
}
