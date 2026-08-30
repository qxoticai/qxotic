package com.qxotic.jinfer.models.llama;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Views;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

final class GraniteIntegrationTest {

    private static final String MODEL_PROPERTY = "jinfer.granite.model";
    private static final String LLAMA_LOGITS_PROPERTY = "jinfer.granite.llamaLogits";
    private static final String PROMPT = "The quick brown fox jumps over the lazy dog.";

    @Test
    @Tag("integration")
    void loadsAndDecodesTheRealCheckpoint() throws Exception {
        try (Arena arena = Arena.ofShared()) {
            Granite model = Granite.loadModel(requiredFile(MODEL_PROPERTY), arena);
            int token = model.tokenizer().encodeToArray(PROMPT)[0];
            try (Granite.State state = model.newState(64, 1)) {
                model.ingest(state, Batch.step(token));
                float[] logits = logits(model, state, 0);
                assertEquals(model.configuration().vocabularySize(), logits.length);
                for (float value : logits) assertTrue(Float.isFinite(value));
            }
        }
    }

    @Test
    @Tag("integration")
    void everyIncrementalAndBatchedRowMatchesLlamaCpp() throws Exception {
        float[] expected = readFloats(requiredFile(LLAMA_LOGITS_PROPERTY));
        try (Arena arena = Arena.ofShared()) {
            Granite model = Granite.loadModel(requiredFile(MODEL_PROPERTY), arena);
            int[] tokens = model.tokenizer().encodeToArray(PROMPT);
            int vocabulary = model.configuration().vocabularySize();
            assertEquals(tokens.length * vocabulary, expected.length);

            try (Granite.State state = model.newState(64, 1)) {
                for (int row = 0; row < tokens.length; row++) {
                    model.ingest(state, Batch.step(tokens[row]));
                    assertParity(expected, row, vocabulary, logits(model, state, 0));
                }
            }
            try (Granite.State state = model.newState(64, tokens.length)) {
                model.ingest(state, Batch.score(tokens));
                for (int row = 0; row < tokens.length; row++) {
                    assertParity(expected, row, vocabulary, logits(model, state, row));
                }
            }
        }
    }

    private static void assertParity(float[] matrix, int row, int columns, float[] actual) {
        float[] expected = Arrays.copyOfRange(matrix, row * columns, (row + 1) * columns);
        assertEquals(argmax(expected), argmax(actual), "row " + row + " argmax");
        assertTrue(rmse(expected, actual) < .2, "row " + row + " RMSE");
        assertTrue(maxAbsDifference(expected, actual) < 1f, "row " + row + " max difference");
        assertTrue(cosine(expected, actual) > .995, "row " + row + " cosine");
    }

    private static Path requiredFile(String property) {
        String configured = System.getProperty(property, "");
        assumeTrue(!configured.isBlank(), "set -D" + property + "=/path/to/file");
        Path path = Path.of(configured);
        assumeTrue(Files.isRegularFile(path), path + " is not a file");
        return path;
    }

    private static float[] readFloats(Path path) throws Exception {
        byte[] bytes = Files.readAllBytes(path);
        assertEquals(0, bytes.length % Float.BYTES, "reference logits must be raw FP32");
        float[] values = new float[bytes.length / Float.BYTES];
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(values);
        return values;
    }

    private static float[] logits(Granite model, Granite.State state, int output) {
        return Views.toFloatArray(
                Views.castToSegmentBacked(model.logits(state, output), "logits"), "logits");
    }

    private static int argmax(float[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
        return best;
    }

    private static float maxAbsDifference(float[] left, float[] right) {
        float max = 0f;
        for (int i = 0; i < left.length; i++) max = Math.max(max, Math.abs(left[i] - right[i]));
        return max;
    }

    private static double rmse(float[] left, float[] right) {
        double square = 0;
        for (int i = 0; i < left.length; i++) {
            double difference = left[i] - right[i];
            square += difference * difference;
        }
        return Math.sqrt(square / left.length);
    }

    private static double cosine(float[] left, float[] right) {
        double dot = 0, aa = 0, bb = 0;
        for (int i = 0; i < left.length; i++) {
            dot += (double) left[i] * right[i];
            aa += (double) left[i] * left[i];
            bb += (double) right[i] * right[i];
        }
        return dot / Math.sqrt(aa * bb);
    }
}
