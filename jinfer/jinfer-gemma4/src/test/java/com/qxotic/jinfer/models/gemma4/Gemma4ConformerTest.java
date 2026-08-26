package com.qxotic.jinfer.models.gemma4;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class Gemma4ConformerTest {
    private static final int DIM = 32;
    private static final int OUTPUT_DIM = 3;

    @Test
    void preservesPositionGeometryAndRelativePositionOrdering() {
        assertEquals(1, Gemma4Conformer.tokensForFrames(1));
        assertEquals(1, Gemma4Conformer.tokensForFrames(4));
        assertEquals(2, Gemma4Conformer.tokensForFrames(5));
        assertEquals(3, Gemma4Conformer.tokensForFrames(9));

        float[] positions = Gemma4Conformer.buildPositionEmbedding(4);
        assertEquals(Gemma4Conformer.RPE * 4, positions.length);
        assertEquals((float) Math.sin(12), positions[0], 0f);
        assertEquals((float) Math.cos(12f / 10_000f), positions[3], 1e-7f);
        assertArrayEquals(new float[] {0f, 0f, 1f, 1f}, slice(positions, 12 * 4, 4), 0f);
        assertThrows(
                IllegalArgumentException.class,
                () -> Gemma4Conformer.validateArchitecture(Path.of("bad.gguf"), 31, 1, 4));
    }

    @Test
    void computesWholeMelTowerThenStreamsFinalRows() {
        try (Arena weightsArena = Arena.ofConfined()) {
            MemoryArena<MemorySegment> weights = MemoryAllocators.ofArena(weightsArena);
            Gemma4Conformer conformer = synthetic(weights);
            Media.Audio audio = new Media.Audio(new float[1_000], 16_000, 1);
            List<float[]> output = new ArrayList<>();
            List<MemoryView<?>> borrowed = new ArrayList<>();

            conformer.project(
                    audio,
                    1,
                    chunk -> {
                        assertEquals(1, chunk.shape().flatAt(0));
                        assertEquals(OUTPUT_DIM, chunk.shape().flatAt(1));
                        MemoryView<MemorySegment> rows =
                                Views.castToSegmentBacked(chunk, "conformer rows");
                        output.add(values(rows));
                        borrowed.add(chunk);
                    });

            assertEquals(2, conformer.positions(audio));
            assertEquals(2, output.size());
            float rms = (float) Math.sqrt((1 + 4 + 9) / 3f + 1e-6f);
            float[] expected = {1 / rms, 2 / rms, 3 / rms};
            assertArrayEquals(expected, output.get(0), 0f);
            assertArrayEquals(expected, output.get(1), 0f);
            assertFalse(((MemorySegment) borrowed.getFirst().memory().base()).scope().isAlive());
        }
    }

    @Test
    void parsedLoaderRequiresGemma4aAndChecksEveryTensorShape() {
        GGUF gguf = metadata("gemma4a");
        try (Arena arena = Arena.ofConfined()) {
            MemoryArena<MemorySegment> memory = MemoryAllocators.ofArena(arena);
            Map<String, MemoryView<MemorySegment>> tensors = tensors(memory);
            Gemma4Conformer.loadModel(Path.of("synthetic.gguf"), gguf, tensors, arena);

            Map<String, MemoryView<MemorySegment>> wrongShape = new HashMap<>(tensors);
            wrongShape.put("a.pre_encode.out.bias", Views.allocateF32(memory, OUTPUT_DIM + 1));
            assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            Gemma4Conformer.loadModel(
                                    Path.of("synthetic.gguf"), gguf, wrongShape, arena));

            Map<String, MemoryView<MemorySegment>> missing = new HashMap<>(tensors);
            missing.remove("a.conv1d.1.norm.weight");
            assertThrows(
                    IllegalStateException.class,
                    () ->
                            Gemma4Conformer.loadModel(
                                    Path.of("synthetic.gguf"), gguf, missing, arena));
        }
        assertThrows(
                IllegalArgumentException.class,
                () -> Gemma4Conformer.loadModel(metadata("gemma4ua"), Map.of(), Arena.ofAuto()));
    }

    private static Gemma4Conformer synthetic(MemoryArena<MemorySegment> memory) {
        return new Gemma4Conformer(
                DIM,
                1,
                8,
                4,
                OUTPUT_DIM,
                1e-6f,
                new float[9 * 128],
                filled(memory, new long[] {128}, 1f),
                new float[9 * 128 * 32],
                filled(memory, new long[] {32}, 1f),
                unclamped(identity(memory, DIM, DIM)),
                new Gemma4Conformer.Block[0],
                unclamped(filled(memory, new long[] {OUTPUT_DIM, DIM}, 0f)),
                tensor(memory, new long[] {OUTPUT_DIM}, 1f, 2f, 3f),
                unclamped(identity(memory, OUTPUT_DIM, OUTPUT_DIM)));
    }

    private static GGUF metadata(String projectorType) {
        return Builder.newBuilder()
                .putString("clip.audio.projector_type", projectorType)
                .putInteger("clip.audio.embedding_length", DIM)
                .putInteger("clip.audio.attention.head_count", 1)
                .putInteger("clip.audio.feed_forward_length", 8)
                .putInteger("clip.audio.block_count", 0)
                .putInteger("clip.audio.num_mel_bins", 4)
                .putInteger("clip.audio.projection_dim", OUTPUT_DIM)
                .build();
    }

    private static Map<String, MemoryView<MemorySegment>> tensors(
            MemoryArena<MemorySegment> memory) {
        Map<String, MemoryView<MemorySegment>> tensors = new HashMap<>();
        tensors.put("a.conv1d.0.weight", filled(memory, new long[] {128, 1, 3, 3}, 0f));
        tensors.put("a.conv1d.0.norm.weight", filled(memory, new long[] {128}, 1f));
        tensors.put("a.conv1d.1.weight", filled(memory, new long[] {32, 128, 3, 3}, 0f));
        tensors.put("a.conv1d.1.norm.weight", filled(memory, new long[] {32}, 1f));
        tensors.put("a.input_projection.weight", identity(memory, DIM, DIM));
        tensors.put("a.pre_encode.out.weight", filled(memory, new long[] {OUTPUT_DIM, DIM}, 0f));
        tensors.put("a.pre_encode.out.bias", tensor(memory, new long[] {OUTPUT_DIM}, 1f, 2f, 3f));
        tensors.put("mm.a.input_projection.weight", identity(memory, OUTPUT_DIM, OUTPUT_DIM));
        return tensors;
    }

    private static Clamped unclamped(MemoryView<MemorySegment> weight) {
        return new Clamped(
                weight, -Float.MAX_VALUE, Float.MAX_VALUE, -Float.MAX_VALUE, Float.MAX_VALUE);
    }

    private static MemoryView<MemorySegment> identity(
            MemoryArena<MemorySegment> memory, int rows, int columns) {
        MemoryView<MemorySegment> value = Views.allocateF32(memory, rows, columns);
        float[] values = new float[rows * columns];
        for (int i = 0; i < Math.min(rows, columns); i++) values[i * columns + i] = 1f;
        Views.copyFromArray(value, 0, values, 0, values.length, "identity");
        return value;
    }

    private static MemoryView<MemorySegment> filled(
            MemoryArena<MemorySegment> memory, long[] shape, float fill) {
        MemoryView<MemorySegment> value = Views.allocateF32(memory, shape);
        float[] values = new float[Math.toIntExact(value.shape().size())];
        Arrays.fill(values, fill);
        Views.copyFromArray(value, 0, values, 0, values.length, "filled tensor");
        return value;
    }

    private static MemoryView<MemorySegment> tensor(
            MemoryArena<MemorySegment> memory, long[] shape, float... values) {
        MemoryView<MemorySegment> value = Views.allocateF32(memory, shape);
        assertEquals(value.shape().size(), values.length);
        Views.copyFromArray(value, 0, values, 0, values.length, "test tensor");
        return value;
    }

    private static float[] values(MemoryView<MemorySegment> value) {
        return Views.toFloatArray(value, "test tensor");
    }

    private static float[] slice(float[] values, int offset, int length) {
        float[] result = new float[length];
        System.arraycopy(values, offset, result, 0, length);
        return result;
    }
}
