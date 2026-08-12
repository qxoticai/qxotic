package com.qxotic.jinfer.x.models.gemma4;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jinfer.x.PanamaMemoryArena;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.Media;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class Gemma4AudioParityTest {
    @Test
    void matchesLegacyUnifiedAudioProjector() throws Exception {
        Path path = TestModels.require("hf.co/unsloth/gemma-4-12b-it-GGUF/mmproj-F32.gguf");
        float[] pcm = new float[10_000];
        for (int i = 0; i < pcm.length; i++) pcm[i] = (float) Math.sin(i * 0.07);

        try (Arena oldArena = Arena.ofShared();
                Arena xArena = Arena.ofShared()) {
            FloatTensor expected =
                    com.qxotic.jinfer.models.gemma4.Gemma4Audio.loadModel(path, oldArena)
                            .encode(new com.qxotic.jinfer.Media.Audio(pcm, 16_000, 1));
            Gemma4Audio actual = Gemma4Audio.loadModel(path, xArena);
            int[] offset = {0};
            actual.embed(
                    new Media.Audio(pcm, 16_000, 1),
                    3,
                    rows -> {
                        var view = Views.castToSegmentBacked(rows, "audio output");
                        float maxError = 0;
                        for (int i = 0; i < view.shape().size(); i++)
                            maxError =
                                    Math.max(
                                            maxError,
                                            Math.abs(
                                                    expected.getFloat(offset[0] + i)
                                                            - Views.getFloat(
                                                                    view, i, "audio output")));
                        assertTrue(maxError < 2e-3f, "maximum absolute error: " + maxError);
                        offset[0] += Math.toIntExact(view.shape().size());
                    });
            assertEquals(expected.size(), offset[0]);
        }
    }

    @Test
    void matchesLegacyDecoderAcrossChunkedCausalAudio() throws Exception {
        Path text = TestModels.require("hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0");
        Path mmproj = TestModels.require("hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf");
        float[] pcm = new float[10_000];
        for (int i = 0; i < pcm.length; i++) pcm[i] = (float) Math.sin(i * 0.07);
        var oldAudio = new com.qxotic.jinfer.Media.Audio(pcm, 16_000, 1);
        var xAudio = new Media.Audio(pcm, 16_000, 1);

        try (Arena oldArena = Arena.ofShared();
                Arena xArena = Arena.ofShared()) {
            var old =
                    com.qxotic.jinfer.models.gemma4.Gemma4.loadModel(text, oldArena)
                            .attachMediaEncoders(mmproj, oldArena);
            var x = Gemma4.loadModel(text, mmproj, xArena);
            int[] prefix = x.tokenizer().encodeToArray("Listen:");
            int rows = x.embedder(Media.Audio.class).orElseThrow().positions(xAudio);
            int context = prefix.length + rows + 2;
            try (var oldState = old.newState(context, rows);
                    var xState = x.newState(context, 4)) {
                old.ingest(oldState, com.qxotic.jinfer.Batch.prefill(prefix));
                x.ingest(xState, com.qxotic.jinfer.x.boundary.Batch.prefill(prefix));
                old.embedder(com.qxotic.jinfer.Media.Audio.class)
                        .orElseThrow()
                        .embed(
                                oldAudio,
                                rows,
                                projected -> {
                                    old.ingest(
                                            oldState,
                                            com.qxotic.jinfer.Batch.embeddings(
                                                    projected, rows, false));
                                    int dim = x.config().embeddingLength();
                                    for (int first = 0; first < rows; first += 4) {
                                        int count = Math.min(4, rows - first);
                                        MemoryView<MemorySegment> chunk =
                                                Views.allocateF32(
                                                        new PanamaMemoryArena(xArena), count, dim);
                                        float[] values = new float[count * dim];
                                        for (int i = 0; i < values.length; i++)
                                            values[i] = projected.getFloat((long) first * dim + i);
                                        Views.copyFromArray(
                                                chunk, 0, values, 0, values.length, "audio rows");
                                        x.ingest(
                                                xState,
                                                com.qxotic.jinfer.x.boundary.Batch.embeddings(
                                                        chunk, count, false));
                                    }
                                });
                assertLogits(
                        old.logits(oldState),
                        Views.castToSegmentBacked(x.logits(xState), "logits"));
            }
        }
    }

    private static void assertLogits(FloatTensor expected, MemoryView<MemorySegment> actual) {
        assertEquals(expected.size(), actual.shape().size());
        float maxError = 0;
        for (int i = 0; i < expected.size(); i++)
            maxError =
                    Math.max(
                            maxError,
                            Math.abs(expected.getFloat(i) - Views.getFloat(actual, i, "logits")));
        assertTrue(maxError < 1e-2f, "maximum absolute error: " + maxError);
    }
}
