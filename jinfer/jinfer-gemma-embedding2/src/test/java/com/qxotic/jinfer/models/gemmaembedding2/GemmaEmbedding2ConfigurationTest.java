package com.qxotic.jinfer.models.gemmaembedding2;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.format.gguf.Builder;
import com.qxotic.jinfer.chat.ModelProvider;
import java.util.ServiceLoader;
import org.junit.jupiter.api.Test;

final class GemmaEmbedding2ConfigurationTest {

    @Test
    void providerIsRegistered() {
        assertEquals(
                1,
                ServiceLoader.load(ModelProvider.class).stream()
                        .filter(provider -> provider.type().equals(GemmaEmbedding2Provider.class))
                        .count());
    }

    @Test
    void readsPerLayerHeadsAndWindowFlags() {
        GemmaEmbedding2.Configuration c =
                GemmaEmbedding2.loadConfiguration(metadata().build(), 262_144);
        assertArrayEquals(new int[] {2, 2, 1}, c.numberOfKeyValueHeads());
        assertArrayEquals(new boolean[] {true, true, false}, c.isSwa());
        assertEquals(256, c.headSize(0));
        assertEquals(512, c.headSize(2));
        assertEquals(1_024, c.slidingWindow());
        assertEquals(768, c.embeddingLengthOut());
    }

    @Test
    void readsAWindowPeriodLikeLlamaCpp() {
        Builder period = metadata().putInteger(KEY + "attention.sliding_window_pattern", 2);
        assertArrayEquals(
                new boolean[] {true, false, true},
                GemmaEmbedding2.loadConfiguration(period.build(), 262_144).isSwa());
        Builder every = metadata().putInteger(KEY + "attention.sliding_window_pattern", 0);
        assertArrayEquals(
                new boolean[] {true, true, true},
                GemmaEmbedding2.loadConfiguration(every.build(), 262_144).isSwa());
    }

    @Test
    void refusesWhatTheGraphDoesNotImplement() {
        for (Builder unsupported :
                new Builder[] {
                    metadata().putBoolean(KEY + "attention.causal", true),
                    metadata().putInteger(KEY + "pooling_type", 2),
                    metadata().putInteger(KEY + "attention.shared_kv_layers", 1),
                    metadata().putInteger(KEY + "rope.dimension_count", 128),
                    metadata()
                            .putArrayOfInteger(KEY + "attention.head_count_kv", new int[] {3, 2, 1})
                })
            assertThrows(
                    IllegalArgumentException.class,
                    () -> GemmaEmbedding2.loadConfiguration(unsupported.build(), 262_144));
    }

    private static final String KEY = "gemma-embedding2.";

    private static Builder metadata() {
        return Builder.newBuilder()
                .putInteger(KEY + "block_count", 3)
                .putInteger(KEY + "context_length", 262_144)
                .putInteger(KEY + "embedding_length", 512)
                .putInteger(KEY + "feed_forward_length", 2_048)
                .putInteger(KEY + "attention.head_count", 4)
                .putArrayOfInteger(KEY + "attention.head_count_kv", new int[] {2, 2, 1})
                .putFloat(KEY + "rope.freq_base", 1_000_000f)
                .putFloat(KEY + "rope.freq_base_swa", 10_000f)
                .putFloat(KEY + "attention.layer_norm_rms_epsilon", 1e-6f)
                .putInteger(KEY + "attention.key_length", 512)
                .putInteger(KEY + "attention.value_length", 512)
                .putInteger(KEY + "attention.key_length_swa", 256)
                .putInteger(KEY + "attention.value_length_swa", 256)
                .putInteger(KEY + "rope.dimension_count", 512)
                .putInteger(KEY + "rope.dimension_count_swa", 256)
                .putInteger(KEY + "attention.sliding_window", 1_024)
                .putArrayOfBoolean(
                        KEY + "attention.sliding_window_pattern", new boolean[] {true, true, false})
                .putInteger(KEY + "embedding_length_per_layer_input", 512)
                .putInteger(KEY + "embedding_length_out", 768)
                .putBoolean(KEY + "attention.causal", false)
                .putInteger(KEY + "pooling_type", 1);
    }
}
