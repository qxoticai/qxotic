package com.qxotic.jinfer.models.gemma4;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.PanamaMemoryArena;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.boundary.Media;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import org.junit.jupiter.api.Test;

class Gemma4AudioTest {
    private static final int FRAME_SIZE = 640;

    @Test
    void downmixesAndLinearlyResamplesPcm() {
        Media.Audio stereo8k = new Media.Audio(new float[] {1, 1, 1, -1}, 8_000, 2);
        assertEquals(4, AudioPreprocess.mono16kLength(stereo8k));
        assertArrayEquals(new float[] {1, 0.5f, 0, 0}, AudioPreprocess.toMono16k(stereo8k), 0);
    }

    @Test
    void streamsCausalRowsAndPositionsMatchesEveryEdgeCase() {
        try (Arena weightsArena = Arena.ofConfined()) {
            MemoryView<MemorySegment> projection = projection(new PanamaMemoryArena(weightsArena));
            Gemma4Audio audio = new Gemma4Audio(2, 1e-6f, projection);
            float[] pcm = new float[FRAME_SIZE + 1];
            for (int i = 0; i < FRAME_SIZE; i++) pcm[i] = 1;
            pcm[FRAME_SIZE] = 0.5f;
            List<float[]> rows = assertEmission(audio, new Media.Audio(pcm, 16_000, 1), 1, 2);
            assertEquals((float) (1 / Math.sqrt(1 + 1e-6f)), rows.get(0)[0], 1e-6f);
            assertEquals(
                    (float) (0.5 / Math.sqrt(0.25 / FRAME_SIZE + 1e-6f)), rows.get(1)[0], 1e-5f);
            assertArrayEquals(
                    new float[] {0, 0},
                    assertEmission(audio, new Media.Audio(new float[0], 16_000, 1), 3, 1)
                            .getFirst(),
                    0);
            assertArrayEquals(
                    new float[] {0, 0},
                    assertEmission(audio, new Media.Audio(new float[0], 8_000, 1), 3, 1).getFirst(),
                    0);
        }
    }

    @Test
    void parsedLoaderChecksProjectorShapeAndDtype() {
        GGUF gguf =
                Builder.newBuilder()
                        .putString("clip.audio.projector_type", "gemma4ua")
                        .putInteger("clip.audio.projection_dim", 2)
                        .build();
        try (Arena arena = Arena.ofConfined()) {
            PanamaMemoryArena memory = new PanamaMemoryArena(arena);
            MemoryView<MemorySegment> valid = projection(memory);
            Gemma4Audio.loadModel(
                    Path.of("synthetic.gguf"), gguf, Map.of("mm.a.input_projection.weight", valid));
            MemoryView<MemorySegment> wrongShape = Views.allocateF32(memory, 3, 639);
            assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            Gemma4Audio.loadModel(
                                    Path.of("synthetic.gguf"),
                                    gguf,
                                    Map.of("mm.a.input_projection.weight", wrongShape)));
            MemoryView<MemorySegment> wrongType = Views.allocateF16(memory, 2, 640);
            assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            Gemma4Audio.loadModel(
                                    Path.of("synthetic.gguf"),
                                    gguf,
                                    Map.of("mm.a.input_projection.weight", wrongType)));
        }
        GGUF wrongProjector =
                Builder.newBuilder()
                        .putString("clip.audio.projector_type", "gemma4a")
                        .putInteger("clip.audio.projection_dim", 2)
                        .build();
        assertThrows(
                IllegalArgumentException.class,
                () -> Gemma4Audio.loadModel(wrongProjector, Map.of()));
    }

    private static List<float[]> assertEmission(
            Gemma4Audio embedder, Media.Audio source, int maxChunkSize, int expectedRows) {
        List<float[]> output = new CopyOnWriteArrayList<>();
        int[] rows = {0};
        embedder.project(
                source,
                maxChunkSize,
                chunk -> {
                    int count = Math.toIntExact(chunk.shape().flatAt(0));
                    assertEquals(2, chunk.shape().flatAt(1));
                    assertEquals(Math.min(maxChunkSize, expectedRows - rows[0]), count);
                    MemoryView<MemorySegment> segmentChunk =
                            Views.castToSegmentBacked(chunk, "audio output");
                    for (int row = 0; row < count; row++)
                        output.add(
                                new float[] {
                                    Views.getFloat(segmentChunk, (long) row * 2, "audio output"),
                                    Views.getFloat(segmentChunk, (long) row * 2 + 1, "audio output")
                                });
                    rows[0] += count;
                });
        assertEquals(expectedRows, embedder.positions(source));
        assertEquals(expectedRows, rows[0]);
        assertEquals(expectedRows, output.size());
        return output;
    }

    private static MemoryView<MemorySegment> projection(PanamaMemoryArena arena) {
        MemoryView<MemorySegment> value = Views.allocateF32(arena, 2, FRAME_SIZE);
        float[] values = new float[2 * FRAME_SIZE];
        values[0] = 1;
        values[FRAME_SIZE] = 1;
        Views.copyFromArray(value, 0, values, 0, values.length, "projection");
        return value;
    }
}
