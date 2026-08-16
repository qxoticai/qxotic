package com.qxotic.jinfer.x.models.lfm2;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.chat.Models;
import com.qxotic.jinfer.x.kernels.ModelLoader;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** LFM2.5-Embedding parity against llama.cpp on the same Q8_0 checkpoint. */
final class XLfm2EmbeddingTest {

    private static final List<String> TEXTS =
            List.of(
                    "query: What is panda?",
                    "document: The giant panda (Ailuropoda melanoleuca), sometimes called a panda"
                            + " bear or simply panda, is a bear species endemic to China.",
                    "document: it is a bear",
                    "document: hi",
                    "query: Wie funktioniert ein Kernkraftwerk?",
                    "document: A nuclear power plant heats water into steam that drives a turbine"
                            + " connected to a generator.");

    private static Lfm2 model;
    private static Tokenizer tokenizer;
    private static int bos;

    @BeforeAll
    static void loadModel() throws Exception {
        Path path = TestModels.require("hf.co/LiquidAI/LFM2.5-Embedding-350M-GGUF:Q8_0");
        try (FileChannel channel = FileChannel.open(path)) {
            GGUF gguf = ModelLoader.readGguf(channel, "lfm2.5-embedding");
            tokenizer = GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
            bos = gguf.getValue(int.class, "tokenizer.ggml.bos_token_id");
        }
        var loaded = Models.loadEmbedder(path, Arena.ofAuto(), tokenizer);
        assertSame(tokenizer, loaded.tokenizer());
        model = (Lfm2) loaded.model();
    }

    @Test
    void matchesLlamaCppEmbeddings() throws Exception {
        assertTrue(model.configuration().isEmbedder());
        assertEquals(1024, model.configuration().embeddingLength());
        List<double[]> golden = golden();
        List<float[]> actual = new ArrayList<>();

        try (var state = model.newState(2048, 64)) {
            model.embedAll(
                    state,
                    sequences(TEXTS),
                    view ->
                            actual.add(
                                    Views.toFloatArray(
                                            Views.castToSegmentBacked(view, "embedding"),
                                            "embedding")));
        }

        assertEquals(golden.size(), actual.size());
        for (int i = 0; i < actual.size(); i++) {
            assertTrue(
                    cosine(actual.get(i), golden.get(i)) > 0.999,
                    "vector " + i + " diverged from llama.cpp");
            double norm = 0;
            for (float value : actual.get(i)) norm += value * value;
            assertEquals(1.0, Math.sqrt(norm), 1e-3, "vector " + i + " is not normalized");
        }
    }

    @Test
    void refusesASequenceLargerThanTheBatch() {
        try (var state = model.newState(2048, 16)) {
            IllegalArgumentException error =
                    assertThrows(
                            IllegalArgumentException.class,
                            () ->
                                    model.embedAll(
                                            state,
                                            sequences(List.of("word ".repeat(64))),
                                            v -> {}));
            assertTrue(error.getMessage().contains("whole"), error.getMessage());
        }
    }

    private static Batch.Input.Sequences sequences(List<String> texts) {
        int[][] sequences = new int[texts.size()][];
        for (int i = 0; i < texts.size(); i++) {
            int[] text = tokenizer.encodeToArray(texts.get(i));
            sequences[i] = new int[text.length + 1];
            sequences[i][0] = bos;
            System.arraycopy(text, 0, sequences[i], 1, text.length);
        }
        return (Batch.Input.Sequences) Batch.pack(sequences).input();
    }

    private static List<double[]> golden() throws Exception {
        List<double[]> vectors = new ArrayList<>();
        try (BufferedReader in =
                new BufferedReader(
                        new InputStreamReader(
                                XLfm2EmbeddingTest.class.getResourceAsStream(
                                        "/lfm25-embedding-350m-q8-golden.csv"),
                                StandardCharsets.UTF_8))) {
            for (String line; (line = in.readLine()) != null; ) {
                if (!line.isBlank()) {
                    vectors.add(
                            Arrays.stream(line.split(","))
                                    .mapToDouble(Double::parseDouble)
                                    .toArray());
                }
            }
        }
        return vectors;
    }

    private static double cosine(float[] a, double[] b) {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return dot / Math.sqrt(na * nb);
    }
}
