package com.qxotic.jinfer.models.kokoro;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.Builder;
import org.junit.jupiter.api.Test;

final class KokoroConfigurationTest {

    @Test
    void readsTheSupportedSchemaWithoutExposingMetadataArrays() {
        Kokoro.Configuration config = Kokoro.readConfig(metadata().build());
        assertEquals(24_000, config.sampleRate());
        assertEquals(510, config.plbertMaxPositions() - 2);

        int[] rates = config.upsampleRates();
        rates[0] = 1;
        assertEquals(10, config.upsampleRates()[0]);

        String[] tokens = config.tokens();
        tokens[0] = "changed";
        assertEquals("$", config.tokens()[0]);
    }

    @Test
    void rejectsArchitecturesAndLayoutsThisPortDoesNotImplement() {
        assertRejected(metadata().putString("general.architecture", "kokoro2"), "architecture");
        assertRejected(metadata().putInteger("kokoro.n_token", 177), "symbol table");
        assertRejected(metadata().putInteger("kokoro.istft.n_fft", 32), "audio layout");
        assertRejected(
                metadata().putArrayOfInteger("kokoro.istft.upsample_rates", new int[] {5, 12}),
                "ISTFTNet layout");
        assertRejected(
                metadata().putInteger("kokoro.plbert.num_hidden_layers", 6), "PL-BERT layout");
    }

    @Test
    void namesMissingMetadata() {
        assertRejected(metadata().removeKey("kokoro.sample_rate"), "kokoro.sample_rate");
    }

    private static void assertRejected(Builder builder, String message) {
        IllegalArgumentException failure =
                assertThrows(
                        IllegalArgumentException.class, () -> Kokoro.readConfig(builder.build()));
        assertTrue(failure.getMessage().contains(message), failure::getMessage);
    }

    static Builder metadata() {
        String[] tokens = new String[178];
        tokens[0] = "$";
        for (int i = 1; i < tokens.length; i++) tokens[i] = "t" + i;
        return Builder.newBuilder()
                .putString("general.architecture", "kokoro")
                .putInteger("kokoro.dim_in", 64)
                .putInteger("kokoro.hidden_dim", 512)
                .putInteger("kokoro.style_dim", 128)
                .putInteger("kokoro.max_conv_dim", 512)
                .putInteger("kokoro.max_dur", 50)
                .putInteger("kokoro.n_token", 178)
                .putInteger("kokoro.n_mels", 80)
                .putInteger("kokoro.n_layer", 3)
                .putInteger("kokoro.sample_rate", 24_000)
                .putInteger("kokoro.text_encoder_kernel_size", 5)
                .putInteger("kokoro.istft.init_channel", 512)
                .putInteger("kokoro.istft.n_fft", 20)
                .putInteger("kokoro.istft.hop_size", 5)
                .putArrayOfInteger("kokoro.istft.upsample_rates", new int[] {10, 6})
                .putArrayOfInteger("kokoro.istft.upsample_kernel_sizes", new int[] {20, 12})
                .putArrayOfInteger("kokoro.istft.resblock_kernel_sizes", new int[] {3, 7, 11})
                .putArrayOfInteger(
                        "kokoro.istft.resblock_dilation_sizes",
                        new int[] {1, 3, 5, 1, 3, 5, 1, 3, 5})
                .putInteger("kokoro.plbert.embedding_size", 128)
                .putInteger("kokoro.plbert.hidden_size", 768)
                .putInteger("kokoro.plbert.num_hidden_layers", 12)
                .putInteger("kokoro.plbert.num_attention_heads", 12)
                .putInteger("kokoro.plbert.intermediate_size", 2048)
                .putInteger("kokoro.plbert.max_position_embeddings", 512)
                .putArrayOfString("tokenizer.ggml.tokens", tokens);
    }
}
