package com.qxotic.jinfer.models.parakeet;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
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
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * TDT decoder parity against the parakeet.cpp fixture: SOS prediction output, joint logits with
 * that state, the exact greedy decode trace, the transcript, and finally the full pipeline from raw
 * PCM through jinfer's own encoder.
 */
@Tag("integration")
class ParakeetTdtTest {

    @Test
    void decoderMatchesFixtureTrace() throws IOException {
        for (String tag : Fixtures.MODELS) decoderMatchesTrace(tag);
    }

    private void decoderMatchesTrace(String tag) throws IOException {
        Optional<Path> fixturePath = Fixtures.fixture(tag + "-f16-jfk.fixture.gguf");
        assumeTrue(fixturePath.isPresent(), "parakeet fixture not checked out");
        Path model = TestModels.require("mudler/parakeet-cpp-gguf/" + tag + "-f16.gguf");

        try (FileChannel channel = FileChannel.open(fixturePath.get(), StandardOpenOption.READ);
                Arena arena = Arena.ofShared()) {
            GGUF fixture = GGUF.read(fixturePath.get());
            ParakeetTdt tdt = Fixtures.load(model, arena).weights().decoder();
            int vPlus = fixture.getValue(int.class, "fixture.v_plus");
            assertEquals(vPlus, tdt.config().vPlus());
            float[] sosExpected = Fixtures.floats(channel, fixture, "pred_sos");
            float[] encoderOut = Fixtures.floats(channel, fixture, "encoder_out");
            float[] logitsExpected = Fixtures.floats(channel, fixture, "joint_logits_sos");
            int[] tokens = Fixtures.ints(channel, fixture, "tdt_tokens");
            int[] tokenFrames = Fixtures.ints(channel, fixture, "tdt_frames");
            int[] durations = Fixtures.ints(channel, fixture, "tdt_durations");
            String transcript = fixture.getValue(String.class, "fixture.transcript");

            Fixtures.onScratch(
                    workspace -> {
                        // 1. SOS prediction output (all-F32 weights: expect near-exact agreement).
                        float[] sos = tdt.probeSos(workspace);
                        assertTrue(
                                maxAbs(sosExpected, sos) < 1e-4,
                                "pred_sos diff " + maxAbs(sosExpected, sos));

                        // 2. Joint logits for the first frames with the SOS state.
                        int dModel = 1024;
                        int frames = encoderOut.length / dModel;
                        float[] frameMajor = new float[encoderOut.length];
                        for (int c = 0; c < dModel; c++)
                            for (int t = 0; t < frames; t++)
                                frameMajor[t * dModel + c] = encoderOut[c * frames + t];
                        float[] projected =
                                tdt.encProjection(
                                        Views.fromFloatArray(workspace, frameMajor),
                                        frames,
                                        workspace);
                        int probes = logitsExpected.length / vPlus;
                        double logitDiff = 0;
                        for (int t = 0; t < probes; t++) {
                            float[] logits = tdt.probeJointLogits(projected, t, sos, workspace);
                            for (int v = 0; v < vPlus; v++)
                                logitDiff =
                                        Math.max(
                                                logitDiff,
                                                Math.abs(
                                                        logitsExpected[t * vPlus + v] - logits[v]));
                        }
                        System.out.printf("%s joint_logits_sos maxAbs=%.3e%n", tag, logitDiff);
                        // Measured 1.6e-2 on logits spanning tens: the F16 joint projections
                        // quantize activations in ggml but not in jinfer, the same asymmetry as
                        // the encoder layers.
                        assertTrue(logitDiff < 5e-2, "joint logits diff " + logitDiff);

                        // 3. The greedy trace, integer-exact, and the transcript, byte-identical.
                        List<ParakeetTdt.Emission> emissions =
                                tdt.decode(projected, frames, workspace);
                        assertEquals(tokens.length, emissions.size(), "emission count");
                        assertArrayEquals(
                                tokens,
                                emissions.stream().mapToInt(ParakeetTdt.Emission::token).toArray());
                        assertArrayEquals(
                                tokenFrames,
                                emissions.stream().mapToInt(ParakeetTdt.Emission::frame).toArray());
                        assertArrayEquals(
                                durations,
                                emissions.stream()
                                        .mapToInt(ParakeetTdt.Emission::duration)
                                        .toArray());
                        assertEquals(transcript, text(tdt, emissions));
                        return null;
                    });
        }
    }

    @Test
    void fullPipelineTranscribesLikeParakeetCpp() throws IOException {
        for (String tag : Fixtures.MODELS) fullPipeline(tag);
    }

    private void fullPipeline(String tag) throws IOException {
        for (String quant : new String[] {"f16", "q8_0"}) {
            Path model =
                    TestModels.require("mudler/parakeet-cpp-gguf/" + tag + "-" + quant + ".gguf");
            try (Arena arena = Arena.ofShared()) {
                Parakeet parakeet = Fixtures.load(model, arena);
                ParakeetEncoder encoder = parakeet.weights().encoder();
                ParakeetTdt tdt = parakeet.weights().decoder();
                for (String clip : new String[] {"jfk", "speech"}) {
                    Optional<Path> fixturePath =
                            Fixtures.fixture(tag + "-" + quant + "-" + clip + ".fixture.gguf");
                    if (fixturePath.isEmpty()) continue; // not every model has every clip
                    try (FileChannel fixtureChannel =
                            FileChannel.open(fixturePath.get(), StandardOpenOption.READ)) {
                        GGUF fixture = GGUF.read(fixturePath.get());
                        float[] pcm = Fixtures.floats(fixtureChannel, fixture, "pcm");
                        int frames = encoder.frames(pcm.length);
                        String transcript =
                                Fixtures.onScratch(
                                        workspace ->
                                                text(
                                                        tdt,
                                                        tdt.decode(
                                                                tdt.encProjection(
                                                                        encoder.encode(
                                                                                pcm,
                                                                                0,
                                                                                pcm.length,
                                                                                null,
                                                                                workspace),
                                                                        frames,
                                                                        workspace),
                                                                frames,
                                                                workspace)));
                        assertEquals(
                                fixture.getValue(String.class, "fixture.transcript"),
                                transcript,
                                tag + "/" + quant + "/" + clip);
                    }
                }
            }
        }
    }

    private static double maxAbs(float[] expected, float[] got) {
        assertEquals(expected.length, got.length);
        double max = 0;
        for (int i = 0; i < got.length; i++) max = Math.max(max, Math.abs(expected[i] - got[i]));
        return max;
    }

    /** The transcript of {@code emissions}: their pieces joined, specials dropped. */
    private static String text(ParakeetTdt tdt, List<ParakeetTdt.Emission> emissions) {
        StringBuilder text = new StringBuilder();
        for (ParakeetTdt.Emission emission : emissions) {
            String piece = tdt.config().pieces()[emission.token()];
            if (!ParakeetTdt.isSpecial(piece)) text.append(piece);
        }
        return text.toString().replace('▁', ' ').strip();
    }
}
