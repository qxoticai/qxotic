package com.qxotic.jinfer.models.parakeet;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.testkit.TestModels;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/**
 * Layer-by-layer parity of the FastConformer encoder against a parakeet.cpp-generated fixture
 * ({@code test-fixtures/parakeet/}, generator {@code dump_fixture.cpp} there), running the real F16
 * checkpoint. Tolerances budget from the measured mel noise floor (~3e-4) and grow with depth as
 * float accumulation differences compound over 24 blocks.
 */
class ParakeetEncoderTest {

    @Test
    void relativePositionTableMatchesReference() throws IOException {
        Optional<Path> fixture = Fixtures.fixture("tdt-0.6b-v3-f16-jfk.fixture.gguf");
        assumeTrue(fixture.isPresent(), "parakeet fixture not checked out");
        try (FileChannel channel = FileChannel.open(fixture.get(), StandardOpenOption.READ)) {
            GGUF gguf = GGUF.read(fixture.get());
            float[] expected = Fixtures.floats(channel, gguf, "pos_emb");
            int dModel = 1024;
            int frames = (expected.length / dModel + 1) / 2;
            float[] table =
                    ParakeetEncoder.relativePositions(
                            frames, dModel, new float[(2 * frames - 1) * dModel]);
            assertEquals(expected.length, table.length);
            double maxAbs = 0;
            for (int i = 0; i < table.length; i++)
                maxAbs = Math.max(maxAbs, Math.abs(expected[i] - table[i]));
            assertTrue(maxAbs < 1e-5, "pos_emb max abs diff: " + maxAbs);
        }
    }

    @Test
    void encoderMatchesParakeetCppLayerDumps() throws IOException {
        for (String tag : Fixtures.MODELS) encoderMatchesLayerDumps(tag);
    }

    private void encoderMatchesLayerDumps(String tag) throws IOException {
        Optional<Path> fixture = Fixtures.fixture(tag + "-f16-jfk.fixture.gguf");
        assumeTrue(fixture.isPresent(), "parakeet fixture not checked out");
        Path model = TestModels.require("mudler/parakeet-cpp-gguf/" + tag + "-f16.gguf");

        try (FileChannel channel = FileChannel.open(fixture.get(), StandardOpenOption.READ);
                Arena arena = Arena.ofShared()) {
            GGUF gguf = GGUF.read(fixture.get());
            float[] pcm = Fixtures.floats(channel, gguf, "pcm");
            ParakeetEncoder encoder = Fixtures.load(model, arena).weights().encoder();
            int layers = encoder.config().layers();

            Map<Integer, String> taps = new HashMap<>();
            taps.put(0, "enc_layer_0");
            taps.put(1, "enc_layer_1");
            taps.put(layers / 2, "enc_layer_mid");
            taps.put(layers - 1, "enc_layer_last");
            Map<Integer, float[]> captured = new HashMap<>();
            float[] output =
                    Fixtures.onScratch(
                            workspace ->
                                    Views.toFloatArray(
                                            encoder.encode(
                                                    pcm,
                                                    0,
                                                    pcm.length,
                                                    (data, layer) -> {
                                                        if (taps.containsKey(layer))
                                                            captured.put(layer, data);
                                                    },
                                                    workspace),
                                            "encoder output"));

            int frames = encoder.frames(pcm.length);
            int dim = encoder.config().dModel();
            // Fixture layer dumps are frame-major [T', d]; encoder_out is channels-first [d, T'].
            for (Map.Entry<Integer, String> tap : taps.entrySet()) {
                float[] expected = Fixtures.floats(channel, gguf, tap.getValue());
                float[] got = captured.get(tap.getKey());
                assertEquals(expected.length, got.length, tap.getValue() + " size");
                double maxAbs = 0;
                for (int i = 0; i < got.length; i++)
                    maxAbs = Math.max(maxAbs, Math.abs(expected[i] - got[i]));
                System.out.printf("%s %s maxAbs=%.3e%n", tag, tap.getValue(), maxAbs);
                // Intermediate layers carry the residual stream's large dynamic range, and ggml's
                // F16 matmul quantizes activations where jinfer keeps F32, so these are structural
                // tripwires, not precision bounds (measured ~0.15-0.5 on values of hundreds).
                assertTrue(maxAbs < 1.0, tap.getValue() + " max abs diff: " + maxAbs);
            }
            float[] expected = Fixtures.floats(channel, gguf, "encoder_out");
            assertEquals(expected.length, (long) frames * dim, "encoder_out size");
            double maxAbs = 0;
            for (int c = 0; c < dim; c++)
                for (int t = 0; t < frames; t++)
                    maxAbs =
                            Math.max(
                                    maxAbs,
                                    Math.abs(expected[c * frames + t] - output[t * dim + c]));
            System.out.printf("%s encoder_out maxAbs=%.3e%n", tag, maxAbs);
            // Post-norm output on the O(1) scale the joint consumes. Measured floors: 3.7e-4 for
            // the 24-layer v3, 8.1e-3 for the 42-layer 1.1b (more accumulation of the same
            // ggml-F16-activation asymmetry; its decode traces are still integer-exact).
            double bound = tag.equals("tdt-0.6b-v3") ? 5e-3 : 2e-2;
            assertTrue(maxAbs < bound, tag + " encoder_out max abs diff: " + maxAbs);
        }
    }
}
