package com.qxotic.jinfer.models.parakeet;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * The GEMM relative attention against the per-pair definition it replaced, in double precision:
 * {@code softmax_k((qU·k + qV·p[T-1-q+k]) / sqrt(d)) · v} over the valid keys, padded query rows
 * zero. Model-free, so it runs on every checkout.
 */
class ParakeetAttentionTest {

    @ParameterizedTest(name = "frames={0} valid={1} dim={2} heads={3} gain={4}")
    @CsvSource({
        "1, 1, 8, 2, 1",
        "7, 7, 16, 4, 1",
        "37, 35, 60, 3, 1", // headDim 20 and odd lengths: every vector tail
        "64, 1, 32, 2, 1", // a single valid frame under 63 padded ones
        "40, 39, 32, 2, 12", // peaked softmax: scores in the hundreds
        "200, 199, 1024, 8, 1", // the model's head shape
    })
    void matchesThePerPairDefinition(int frames, int valid, int dim, int heads, float gain) {
        Random random = new Random(frames * 31L + valid);
        int positionRows = 2 * frames - 1;
        float[] queryU = gaussian(random, frames * dim, gain);
        float[] queryV = gaussian(random, frames * dim, gain);
        float[] key = gaussian(random, frames * dim, 1);
        float[] value = gaussian(random, frames * dim, 1);
        float[] position = gaussian(random, positionRows * dim, 1);
        double[] expected =
                reference(queryU, queryV, key, value, position, frames, valid, dim, heads);

        // Everything the valid block must never read is poisoned: padded rows of every input and
        // the relative offsets beyond +-(valid-1).
        for (float[] rows : new float[][] {queryU, queryV, key, value})
            Arrays.fill(rows, valid * dim, frames * dim, Float.NaN);
        Arrays.fill(position, 0, (frames - valid) * dim, Float.NaN);
        Arrays.fill(position, (frames + valid - 1) * dim, positionRows * dim, Float.NaN);

        MemoryArena<MemorySegment> arena = Arenas.newCrossThreadMemoryArena();
        try {
            ParakeetEncoder.Scratch work = ParakeetEncoder.Scratch.allocate(arena, frames, dim, 4);
            // Scratch is reused across layers: no result may lean on it starting out zeroed.
            for (MemoryView<MemorySegment> stale :
                    List.of(
                            work.scores(),
                            work.positionScores(),
                            work.valueT(),
                            work.attentionT(),
                            work.attention()))
                fill(stale, nans(Math.toIntExact(stale.shape().size())));
            fill(work.queryU(), queryU);
            fill(work.queryV(), queryV);
            fill(work.key(), key);
            fill(work.value(), value);
            fill(work.position(), position);
            ParakeetEncoder.relativeAttention(work, frames, valid, dim, heads);
            float[] actual = Views.toFloatArray(work.attention(), "attention");

            double worst = 0;
            for (int i = 0; i < valid * dim; i++) {
                assertTrue(Float.isFinite(actual[i]), "non-finite output at " + i);
                worst = Math.max(worst, Math.abs(actual[i] - expected[i]));
            }
            assertTrue(worst < 1e-4, "max abs error " + worst);
            for (int i = valid * dim; i < frames * dim; i++)
                assertEquals(0f, actual[i], "padded query row " + i / dim + " must be zero");
        } finally {
            Arenas.close(arena);
        }
    }

    /** The replaced loop's math, per (query, head, key), in double. */
    private static double[] reference(
            float[] queryU,
            float[] queryV,
            float[] key,
            float[] value,
            float[] position,
            int frames,
            int valid,
            int dim,
            int heads) {
        int headDim = dim / heads;
        double scale = 1 / Math.sqrt(headDim);
        double[] out = new double[frames * dim];
        for (int q = 0; q < valid; q++) {
            for (int h = 0; h < heads; h++) {
                int base = h * headDim;
                double[] scores = new double[valid];
                double maximum = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < valid; k++) {
                    int relative = frames - 1 - q + k;
                    double content = 0, pos = 0;
                    for (int c = 0; c < headDim; c++) {
                        content += (double) queryU[q * dim + base + c] * key[k * dim + base + c];
                        pos +=
                                (double) queryV[q * dim + base + c]
                                        * position[relative * dim + base + c];
                    }
                    scores[k] = (content + pos) * scale;
                    maximum = Math.max(maximum, scores[k]);
                }
                double sum = 0;
                for (int k = 0; k < valid; k++) sum += scores[k] = Math.exp(scores[k] - maximum);
                for (int k = 0; k < valid; k++)
                    for (int c = 0; c < headDim; c++)
                        out[q * dim + base + c] += scores[k] / sum * value[k * dim + base + c];
            }
        }
        return out;
    }

    private static float[] gaussian(Random random, int size, float gain) {
        float[] values = new float[size];
        for (int i = 0; i < size; i++) values[i] = (float) random.nextGaussian() * gain;
        return values;
    }

    private static float[] nans(int size) {
        float[] values = new float[size];
        Arrays.fill(values, Float.NaN);
        return values;
    }

    private static void fill(MemoryView<MemorySegment> view, float[] values) {
        Views.copyFromArray(view, 0, values, 0, values.length, "attention input");
    }
}
