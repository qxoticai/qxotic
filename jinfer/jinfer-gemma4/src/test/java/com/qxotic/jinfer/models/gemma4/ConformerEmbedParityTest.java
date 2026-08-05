package com.qxotic.jinfer.models.gemma4;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.Media;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * End-to-end tower parity: the sine fixture through the full Conformer versus the values llama.cpp
 * prints in its MTMD_DEBUG_EMBEDDINGS block (test-fixtures/audio/oracle/trace-sine440.txt).
 * Matching first/last-16 of token 0 plus global stats pins every stage at once; on mismatch, bisect
 * with the per-tensor trace in the same file. Skipped when the E2B mmproj or fixtures are absent.
 */
class ConformerEmbedParityTest {

    private static final Path WAV =
            Path.of("../../test-fixtures/audio/sine440-3s.wav").toAbsolutePath().normalize();

    // llama.cpp MTMD_DEBUG_EMBEDDINGS for sine440-3s.wav: shape [1536, 75]
    private static final float[] FIRST16 = {
        -0.767661f, -1.006043f, -0.502152f, -2.082058f, -2.236480f, -0.367943f, 1.253600f,
        0.464692f, -1.339543f, 1.050981f, -2.194752f, -0.962332f, 4.301184f, -1.847641f,
        -2.763173f, 1.130458f
    };
    private static final float[] LAST16 = {
        -0.278815f, 2.637283f, 0.147154f, -7.050886f, 3.151919f, 1.775534f, -0.302658f,
        -1.131007f, 1.461215f, -3.042179f, 0.259548f, -2.660585f, 2.901028f, -2.949881f,
        -5.865742f, 1.738980f
    };
    private static final float MEAN = -0.019500f;
    private static final float STD = 2.656330f;

    static float[] readWav16kMono(Path wav) throws IOException {
        byte[] all = Files.readAllBytes(wav);
        ByteBuffer bb = ByteBuffer.wrap(all).order(ByteOrder.LITTLE_ENDIAN);
        // minimal RIFF walk to the data chunk (fixtures are PCM16 mono 16k)
        bb.position(12);
        while (bb.remaining() > 8) {
            int id = bb.getInt();
            int size = bb.getInt();
            if (id == 0x61746164) { // "data"
                float[] pcm = new float[size / 2];
                for (int i = 0; i < pcm.length; i++) {
                    pcm[i] = bb.getShort() / 32768.0f;
                }
                return pcm;
            }
            bb.position(bb.position() + size);
        }
        throw new IOException("no data chunk in " + wav);
    }

    private static double worst(FloatTensor rows, long offset, float[] reference) {
        double worst = 0;
        for (int i = 0; i < reference.length; i++) {
            worst = Math.max(worst, Math.abs(rows.getFloat(offset + i) - reference[i]));
        }
        return worst;
    }

    @Test
    void towerMatchesLlamaCppEmbeddings() throws IOException {
        Assumptions.assumeTrue(
                Files.exists(TestModels.E2B_MMPROJ), "mmproj missing: " + TestModels.E2B_MMPROJ);
        Assumptions.assumeTrue(Files.exists(WAV), "fixture missing: " + WAV);
        Gemma4Conformer tower = Gemma4Conformer.loadModel(TestModels.E2B_MMPROJ, Arena.ofAuto());
        float[] pcm = readWav16kMono(WAV);
        Media.Audio audio = new Media.Audio(pcm, 16000, 1);

        assertEquals(75, tower.positions(audio), "token count");
        FloatTensor rows = tower.encode(audio);
        assertEquals(75L * 1536, rows.size(), "rows size");

        double sum = 0;
        double sumSq = 0;
        for (long i = 0; i < rows.size(); i++) {
            float v = rows.getFloat(i);
            sum += v;
            sumSq += v * v;
        }
        double mean = sum / rows.size();
        double std = Math.sqrt(sumSq / rows.size() - mean * mean);

        double worstFirst = worst(rows, 0, FIRST16);
        double worstLast = worst(rows, 1536 - 16, LAST16);
        String report =
                String.format(
                        "mean %.6f (ref %.6f)  std %.6f (ref %.6f)  worstFirst16 %.4f  worstLast16"
                                + " %.4f",
                        mean, MEAN, std, STD, worstFirst, worstLast);
        System.out.println("[conformer parity] " + report);
        // the reference ran a Q4 text model but an F32 mmproj; the audio tower is pure F32 on
        // both sides, so agreement should be tight
        assertTrue(Math.abs(mean - MEAN) < 5e-3, report);
        assertTrue(Math.abs(std - STD) < 2e-2, report);
        assertTrue(worstFirst < 5e-2 && worstLast < 5e-2, report);
    }
}
