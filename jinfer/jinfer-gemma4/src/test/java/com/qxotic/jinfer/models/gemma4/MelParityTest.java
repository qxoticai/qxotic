package com.qxotic.jinfer.models.gemma4;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * Mel front-end parity against llama.cpp dumps (test-fixtures/audio/oracle, generated per its
 * README with llama-mtmd-debug). Inputs are the oracle's own synthetic formulas, reproduced
 * bit-exactly here; skipped when the fixture directory is absent.
 */
class MelParityTest {

    private static final Path ORACLE =
            Path.of("../../test-fixtures/audio/oracle").toAbsolutePath().normalize();

    private static float[] input(String kind, int n) {
        float[] pcm = new float[n];
        switch (kind) {
            case "440" -> {
                float pi = 3.14159265f;
                for (int i = 0; i < n; i++) {
                    pcm[i] = (float) Math.sin(2 * pi * 440.0f * i / 16000.0f);
                }
            }
            case "half" -> java.util.Arrays.fill(pcm, 0.5f);
            case "zero" -> {}
            default -> throw new IllegalArgumentException(kind);
        }
        return pcm;
    }

    /** Parses "mel[chunk][m=..][t=..] = value" lines into time-major chunks. */
    private static List<AudioPreprocess.MelChunk> parseDump(Path file) throws IOException {
        Pattern val = Pattern.compile("mel\\[(\\d+)\\]\\[m=(\\d+)\\]\\[t=(\\d+)\\] = (-?[0-9.]+)");
        Pattern dims = Pattern.compile("chunk (\\d+) has n_len=(\\d+), n_mel=(\\d+)");
        java.util.Map<Integer, int[]> shapes = new java.util.HashMap<>();
        java.util.Map<Integer, float[]> data = new java.util.HashMap<>();
        for (String line : Files.readAllLines(file)) {
            Matcher d = dims.matcher(line);
            if (d.find()) {
                shapes.put(
                        Integer.parseInt(d.group(1)),
                        new int[] {Integer.parseInt(d.group(2)), Integer.parseInt(d.group(3))});
                continue;
            }
            Matcher m = val.matcher(line);
            if (m.find()) {
                int c = Integer.parseInt(m.group(1));
                int[] shape = shapes.get(c);
                float[] cell = data.computeIfAbsent(c, k -> new float[shape[0] * shape[1]]);
                int mel = Integer.parseInt(m.group(2));
                int t = Integer.parseInt(m.group(3));
                cell[t * shape[1] + mel] = Float.parseFloat(m.group(4));
            }
        }
        List<AudioPreprocess.MelChunk> out = new java.util.ArrayList<>();
        for (int c = 0; c < shapes.size(); c++) {
            out.add(new AudioPreprocess.MelChunk(data.get(c), shapes.get(c)[0]));
        }
        return out;
    }

    @ParameterizedTest
    @CsvSource({
        "440, 48000, mel-440-48000.txt",
        "half, 16000, mel-half-16000.txt",
        "zero, 16000, mel-zero-16000.txt",
        "440, 560000, mel-440-560000.txt",
        "440, 21920, mel-440-21920.txt",
    })
    void melMatchesLlamaCpp(String kind, int n, String dumpFile) throws IOException {
        Path dump = ORACLE.resolve(dumpFile);
        Assumptions.assumeTrue(Files.exists(dump), "oracle dump missing: " + dump);
        List<AudioPreprocess.MelChunk> oracle = parseDump(dump);
        List<AudioPreprocess.MelChunk> ours = new AudioPreprocess(128).logMel(input(kind, n));
        assertEquals(oracle.size(), ours.size(), "chunk count");
        for (int c = 0; c < ours.size(); c++) {
            AudioPreprocess.MelChunk expectedChunk = oracle.get(c);
            float[] expected = expectedChunk.data();
            AudioPreprocess.MelChunk chunk = ours.get(c);
            assertEquals(expectedChunk.frames(), chunk.frames(), "frames, chunk " + c);
            double worst = 0;
            double sum = 0;
            int worstAt = -1;
            for (int i = 0; i < expected.length; i++) {
                double diff = Math.abs(expected[i] - chunk.data()[i]);
                sum += diff;
                if (diff > worst) {
                    worst = diff;
                    worstAt = i;
                }
            }
            double mean = sum / expected.length;
            // The mean pins structure (a recipe error shifts everything); libm ulps (C sinf
            // vs Java's rounded double sine, in both inputs and twiddle tables) set the floor.
            // Constant inputs
            // (zero/half) are bit-identical on both sides and hold a tight bound; the sine
            // inputs are generated with C sinf there and Java's rounded double-sine here, so
            // the INPUT signals differ by result-ulps before the pipeline runs - the residual
            // (~2e-3 worst) lives only in floor-adjacent leakage bins where ln amplifies it.
            // Embedding-level parity is the final gate.
            boolean exactInput = !kind.equals("440");
            assertTrue(mean < (exactInput ? 5e-6 : 5e-5), "chunk " + c + " mean |diff| " + mean);
            assertTrue(
                    worst < (exactInput ? 1e-4 : 5e-3),
                    "chunk " + c + " worst |diff| " + worst + " at flat index " + worstAt);
        }
    }
}
