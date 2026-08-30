package com.qxotic.jinfer.models.inflect2;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.Builder;
import org.junit.jupiter.api.Test;

final class Inflect2ConfigurationTest {

    @Test
    void readsAValidConfigurationAndDoesNotExposeMutableMetadataArrays() {
        Inflect2.Configuration config = Inflect2.readConfig(metadata().build());
        assertEquals(178, config.symbolCount());
        assertEquals(36, config.hiddenChannels() / config.nHeads());

        int[] rates = config.upsampleRates();
        rates[0] = 1;
        assertEquals(8, config.upsampleRates()[0]);
    }

    @Test
    void rejectsIncompatibleCoreMetadata() {
        assertRejected(metadata().putString("general.architecture", "inflect-v3"), "architecture");
        assertRejected(metadata().putInteger("inflect.v2.symbol_count", 177), "symbol table");
        assertRejected(metadata().putInteger("inflect.v2.inter_channels", 127), "dimensions");
        assertRejected(metadata().putInteger("inflect.v2.n_heads", 5), "dimensions");
        assertRejected(metadata().putInteger("inflect.v2.kernel_size", 2), "dimensions");
    }

    @Test
    void rejectsLayoutsTheForwardPathCannotRepresent() {
        assertRejected(
                metadata()
                        .putArrayOfInteger("inflect.v2.resblock_dilation_sizes", new int[] {1, 3}),
                "resblock layout");
        assertRejected(
                metadata().putArrayOfInteger("inflect.v2.upsample_kernel_sizes", new int[] {16}),
                "upsample layout");
        assertRejected(
                metadata()
                        .putArrayOfInteger(
                                "inflect.v2.upsample_kernel_sizes", new int[] {15, 16, 4, 4}),
                "upsample stage 0");
        assertRejected(metadata().putBoolean("inflect.v2.add_blank", false), "blank-interspersed");
        assertRejected(metadata().putString("inflect.v2.activation", "relu"), "activation");
    }

    private static void assertRejected(Builder metadata, String message) {
        IllegalArgumentException failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Inflect2.readConfig(metadata.build()));
        assertTrue(failure.getMessage().contains(message), failure::getMessage);
    }

    private static Builder metadata() {
        return Builder.newBuilder()
                .putString("general.architecture", "inflect-v2")
                .putInteger("inflect.v2.symbol_count", 178)
                .putInteger("inflect.v2.inter_channels", 128)
                .putInteger("inflect.v2.hidden_channels", 72)
                .putInteger("inflect.v2.filter_channels", 384)
                .putInteger("inflect.v2.n_heads", 2)
                .putInteger("inflect.v2.n_layers", 3)
                .putInteger("inflect.v2.kernel_size", 3)
                .putInteger("inflect.v2.sample_rate", 24_000)
                .putInteger("inflect.v2.upsample_initial_channel", 192)
                .putArrayOfInteger("inflect.v2.resblock_kernel_sizes", new int[] {3, 7, 11})
                .putArrayOfInteger(
                        "inflect.v2.resblock_dilation_sizes", new int[] {1, 3, 5, 1, 3, 5, 1, 3, 5})
                .putArrayOfInteger("inflect.v2.upsample_rates", new int[] {8, 8, 2, 2})
                .putArrayOfInteger("inflect.v2.upsample_kernel_sizes", new int[] {16, 16, 4, 4})
                .putBoolean("inflect.v2.add_blank", true)
                .putString("inflect.v2.activation", "leaky_relu");
    }
}
