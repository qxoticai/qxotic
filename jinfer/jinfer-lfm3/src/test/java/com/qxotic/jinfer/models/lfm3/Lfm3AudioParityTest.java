package com.qxotic.jinfer.models.lfm3;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * The LFM3-Omni audio encoder against llama.cpp's mtmd: the mel, every block's normed input, the
 * stack's final norm and the projected rows, on 16 kHz clips read as miniaudio reads them.
 */
@Tag("integration")
final class Lfm3AudioParityTest {

    static final String TEXT = "hf.co/LiquidAI/LFM3-Omni-2B-GGUF/LFM3-Omni-2B-Q8_0.gguf";
    static final String MMPROJ = "hf.co/LiquidAI/LFM3-Omni-2B-GGUF/mmproj-LFM3-Omni-2B-F16.gguf";
    static final String MMPROJ_F32 =
            "hf.co/LiquidAI/LFM3-Omni-2B-GGUF/mmproj-LFM3-Omni-2B-F32.gguf";

    /**
     * F32 weights leave the order of float sums; below 64 rows ggml's flash attention also leaves
     * its tiled kernel for one that accumulates in F16. The F16 mmproj adds ggml's F16 rounding of
     * every matmul's activations.
     */
    @ParameterizedTest
    @CsvSource({
        "a32-en0, parakeet/en0.wav, F32, 1e-4, 0.9999999",
        "a32-odd, audio/odd-1p37s.wav, F32, 1e-3, 0.9999999",
        "a-en0, parakeet/en0.wav, F16, 1e-3, 0.999999",
        "a-ls1, parakeet/ls1.wav, F16, 1e-3, 0.999999",
        "a-odd, audio/odd-1p37s.wav, F16, 1e-3, 0.999999"
    })
    void theEncoderMatchesLlamaCpp(
            String dump, String clip, String precision, double layerBound, double cosBound)
            throws Exception {
        Path dir = Oracle.dir(dump);
        Media.Audio audio = wav(TestModels.fixture(clip));
        try (Arena arena = Arena.ofShared()) {
            Lfm3 model =
                    Lfm3.loadModel(
                            TestModels.require(TEXT),
                            TestModels.require(precision.equals("F32") ? MMPROJ_F32 : MMPROJ),
                            arena);
            Lfm3Audio encoder = (Lfm3Audio) model.projector(Media.Audio.class).orElseThrow();

            Oracle.Tensor mel = Oracle.tensor(dir, "inp_raw"); // [frames, nMel], time fastest
            int frames = mel.width(), bands = mel.rows();
            float[] ours = encoder.mel(Lfm3Audio.mono(audio)), planar = new float[ours.length];
            for (int t = 0; t < frames; t++)
                for (int m = 0; m < bands; m++) planar[m * frames + t] = ours[t * bands + m];
            Oracle.Score melScore = Oracle.score(mel.data(), planar, frames);
            System.out.printf("%s mel %dx%d %s%n", dump, frames, bands, melScore);
            // near-silent bins (log10 power about -9) carry the float FFT's rounding
            assertTrue(melScore.maxRel() < 1e-4, "mel " + melScore);

            float[] expected = Oracle.floats(dir.resolve("embd.f32"));
            int dim = model.configuration().embeddingLength();
            assertEquals(expected.length / dim, encoder.positions(audio));
            MemoryArena<MemorySegment> scratch = Arenas.newCrossThreadMemoryArena();
            try {
                double[] worst = {0};
                MemoryView<MemorySegment> rows =
                        encoder.encode(
                                audio,
                                scratch,
                                (normed, layer) -> {
                                    try {
                                        Oracle.Tensor ref =
                                                Oracle.tensor(
                                                        dir,
                                                        layer < 17
                                                                ? "operator_norm-" + layer
                                                                : "embedding_norm");
                                        Oracle.Score s =
                                                Oracle.score(
                                                        ref.data(),
                                                        Oracle.rows(
                                                                normed, ref.rows(), ref.width()),
                                                        ref.width());
                                        System.out.printf("  norm-%-2d %s%n", layer, s);
                                        worst[0] = Math.max(worst[0], s.maxRel());
                                    } catch (IOException e) {
                                        throw new UncheckedIOException(e);
                                    }
                                });
                Oracle.Score out =
                        Oracle.score(expected, Oracle.rows(rows, expected.length / dim, dim), dim);
                System.out.printf("  embd %s%n", out);
                assertTrue(worst[0] < layerBound, "layer drift " + worst[0]);
                assertTrue(out.minCos() > cosBound, "rows " + out);
            } finally {
                Arenas.close(scratch);
            }
        }
    }

    /** 16-bit PCM WAV as miniaudio decodes it: {@code s / 32768}. */
    static Media.Audio wav(Path file) throws IOException {
        ByteBuffer b = ByteBuffer.wrap(Files.readAllBytes(file)).order(ByteOrder.LITTLE_ENDIAN);
        int channels = b.getShort(22), rate = b.getInt(24);
        int at = 12;
        while (b.getInt(at) != 0x61746164) at += 8 + b.getInt(at + 4); // "data"
        int size = b.getInt(at + 4);
        float[] pcm = new float[size / 2];
        for (int i = 0; i < pcm.length; i++) pcm[i] = b.getShort(at + 8 + 2 * i) / 32768f;
        return new Media.Audio(pcm, rate, channels);
    }
}
