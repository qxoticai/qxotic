package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/**
 * Parity of the Parakeet mel configuration against a parakeet.cpp-generated fixture ({@code
 * test-fixtures/parakeet/}, see {@code dump_fixture.cpp} there): NeMo preprocessing is pre-emphasis
 * 0.97, a centered 400-sample Hann in a 512 FFT, power spectrum, additive log guard, and
 * per-feature normalization over the valid frames.
 */
class LogMelParakeetTest {
    private static final int N_FFT = 512;
    private static final int HOP = 160;
    private static final int N_MELS = 128;
    private static final int WIN_LENGTH = 400;
    private static final float PREEMPHASIS = 0.97f;
    private static final float LOG_ZERO_GUARD = 5.9604645E-8f;

    @Test
    void matchesParakeetCppMelFixture() throws IOException {
        Optional<Path> fixture = fixture("tdt-0.6b-v3-f16-jfk.fixture.gguf");
        assumeTrue(fixture.isPresent(), "parakeet fixture not checked out");

        try (FileChannel channel = FileChannel.open(fixture.get(), StandardOpenOption.READ)) {
            GGUF gguf = GGUF.read(fixture.get());
            float[] pcm = floats(channel, gguf, "pcm");
            float[] window = floats(channel, gguf, "featurizer_window");
            float[] filterbank = floats(channel, gguf, "featurizer_fb");
            float[] expected = floats(channel, gguf, "mel");
            assertEquals(WIN_LENGTH, window.length);
            assertEquals(N_MELS * (N_FFT / 2 + 1), filterbank.length);

            // torch.stft centers a shorter window inside the FFT frame.
            float[] centered = new float[N_FFT];
            System.arraycopy(window, 0, centered, (N_FFT - WIN_LENGTH) / 2, WIN_LENGTH);
            LogMel logMel =
                    new LogMel(
                            new LogMel.Spec(
                                    N_FFT,
                                    HOP,
                                    N_MELS,
                                    centered,
                                    filterbank,
                                    PREEMPHASIS,
                                    2f,
                                    0f,
                                    LOG_ZERO_GUARD));

            int frames = 1 + pcm.length / HOP;
            int valid = pcm.length / HOP;
            assertEquals(expected.length, frames * N_MELS, "fixture frame count");
            float[] features = logMel.frames(pcm, 0, pcm.length, N_FFT / 2, frames);
            LogMel.normalizePerFeature(features, N_MELS, frames, valid);

            // Fixture layout is feat-major [nMels, T]; ours is frame-major [T, nMels].
            double maxAbs = 0;
            for (int m = 0; m < N_MELS; m++)
                for (int t = 0; t < frames; t++)
                    maxAbs =
                            Math.max(
                                    maxAbs,
                                    Math.abs(expected[m * frames + t] - features[t * N_MELS + m]));
            assertTrue(maxAbs < 1e-3, "max abs diff vs parakeet.cpp mel: " + maxAbs);
        }
    }

    private static float[] floats(FileChannel channel, GGUF gguf, String name) throws IOException {
        TensorEntry tensor =
                gguf.getTensors().stream()
                        .filter(entry -> entry.name().equals(name))
                        .findFirst()
                        .orElseThrow(() -> new IllegalStateException("fixture misses " + name));
        ByteBuffer bytes =
                ByteBuffer.allocate(Math.toIntExact(tensor.byteSize()))
                        .order(ByteOrder.LITTLE_ENDIAN);
        channel.read(bytes, gguf.getTensorDataOffset() + tensor.offset());
        bytes.flip();
        float[] values = new float[Math.toIntExact(tensor.totalNumberOfElements())];
        bytes.asFloatBuffer().get(values);
        return values;
    }

    private static Optional<Path> fixture(String name) {
        for (Path directory = Path.of("").toAbsolutePath();
                directory != null;
                directory = directory.getParent()) {
            Path candidate = directory.resolve("test-fixtures").resolve("parakeet").resolve(name);
            if (Files.isRegularFile(candidate)) return Optional.of(candidate);
        }
        return Optional.empty();
    }
}
